import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import miniharness as mh  # noqa: E402


class FakeLLM(BaseHTTPRequestHandler):
    script = []      # replies, popped in order: {"content": str} or {"tool": (name, args)}
    requests = []
    n_ctx = 2048
    ollama = False   # behave like Ollama: no /props, /api/ps + /v1/models, tool calls without index

    def log_message(self, *a):
        pass

    def reply_json(self, obj, code=200):
        self.send_response(code)
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_GET(self):
        if not self.ollama:
            return self.reply_json({"default_generation_settings": {"n_ctx": self.n_ctx}})
        if self.path == "/api/ps":
            return self.reply_json({"models": [{"name": "qwen:7b", "context_length": 8192}]})
        if self.path == "/v1/models":
            return self.reply_json({"data": [{"id": "qwen:7b"}]})
        self.reply_json({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            return self.reply_json({"parameters": "num_ctx                        6144"})
        FakeLLM.requests.append(req)
        step = FakeLLM.script.pop(0) if FakeLLM.script else {"content": "ok"}
        delta = {"content": step["content"]} if "content" in step else {"tool_calls": [{
            **({} if self.ollama else {"index": 0}), "id": "c1", "function": {
                "name": step["tool"][0], "arguments": json.dumps(step["tool"][1])}}]}
        chunks = [{"choices": [{"delta": delta, "finish_reason": None}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            self.wfile.write(("data: %s\n\n" % json.dumps(c)).encode())
        self.wfile.write(b"data: [DONE]\n\n")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.asked = []
        self.tools = mh.Tools(self.root, lambda t, p: self.asked.append(t) or True, lambda: 2000)

    def tearDown(self):
        self.tmp.cleanup()


class ToolTests(Base):
    def test_sandbox(self):
        with self.assertRaises(mh.ToolError):
            self.tools.read_file("../etc/passwd")
        with self.assertRaises(mh.ToolError):
            self.tools.write_file(os.path.abspath(os.sep) + "x.txt", "no")

    def test_read_chunks(self):
        (self.root / "a.txt").write_text("\n".join("line %d" % i for i in range(1, 500)))
        t = mh.Tools(self.root, lambda *a: True, lambda: 300)
        out = t.read_file("a.txt")
        self.assertIn("start_line=", out)
        self.assertLess(len(out), 500)
        self.assertIn("line 1\n", out)
        self.assertIn("line 499", t.read_file("a.txt", start_line=499))

    def test_edit_and_crlf(self):
        (self.root / "b.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
        self.assertTrue(self.tools.edit_file("b.txt", "two", "2").startswith("OK"))
        self.assertEqual((self.root / "b.txt").read_bytes(), b"one\r\n2\r\nthree\r\n")
        with self.assertRaises(mh.ToolError):
            self.tools.edit_file("b.txt", "nothing", "x")
        (self.root / "c.txt").write_text("x x")
        with self.assertRaises(mh.ToolError):
            self.tools.edit_file("c.txt", "x", "y")

    def test_write_append_binary_search(self):
        self.tools.write_file("sub/d.txt", "alpha\nbeta")
        self.tools.write_file("sub/d.txt", "gamma\n", append=True)
        self.assertEqual((self.root / "sub" / "d.txt").read_text(), "alpha\nbeta\ngamma\n")
        (self.root / "bin.dat").write_bytes(b"\x00\x01\x02")
        with self.assertRaises(mh.ToolError):
            self.tools.read_file("bin.dat")
        self.assertIn("sub/d.txt:2: beta", self.tools.search("BETA"))
        self.assertIn("sub/", self.tools.list_files("."))

    def test_declined(self):
        t = mh.Tools(self.root, lambda *a: False, lambda: 2000)
        self.assertEqual(t.write_file("e.txt", "x"), "Declined by user.")
        self.assertFalse((self.root / "e.txt").exists())


class AgentTests(Base):
    def make_agent(self, n_ctx, **kw):
        args = mh.parse_args(["--dir", str(self.root)])
        for k, v in kw.items():
            setattr(args, k, v)
        client = mh.Client("http://127.0.0.1:1/v1")
        return mh.Agent(args, client, self.tools, n_ctx)

    def test_trimming_keeps_budget_and_structure(self):
        a = self.make_agent(1536, summarize=False)
        for i in range(12):
            a.msgs.append({"role": "user", "content": "question %d " % i + "x" * 300})
            a.msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": "t%d" % i, "type": "function",
                 "function": {"name": "read_file", "arguments": json.dumps({"path": "a"})}}]})
            a.msgs.append({"role": "tool", "tool_call_id": "t%d" % i, "content": "y" * 400})
            a.msgs.append({"role": "assistant", "content": "answer " + "z" * 200})
        a.fit()
        self.assertLessEqual(a.est(), a.budget())
        self.assertEqual(a.msgs[0]["role"], "user")
        ids = [t["id"] for m in a.msgs for t in m.get("tool_calls", [])]
        tool_ids = [m["tool_call_id"] for m in a.msgs if m["role"] == "tool"]
        self.assertEqual(ids, tool_ids)

    def test_oversized_tool_result_is_cut(self):
        a = self.make_agent(1536, summarize=False)
        a.msgs.append({"role": "user", "content": "hi"})
        a.msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "t", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        a.msgs.append({"role": "tool", "tool_call_id": "t", "content": "y" * 20000})
        a.fit()
        self.assertLessEqual(a.est(), a.budget())

    def test_text_call_parsing(self):
        calls = mh.parse_text_calls('ok\nTOOL: {"name": "read_file", "arguments": {"path": "a"}}')
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertIn("error", mh.parse_text_calls('TOOL: {"name": "x", "argu')[0])


class EndToEnd(Base):
    def setUp(self):
        super().setUp()
        FakeLLM.requests, FakeLLM.script = [], []
        self.srv = HTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/v1" % self.srv.server_port

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        super().tearDown()

    def test_server_ctx(self):
        self.assertEqual(mh.Client(self.url).server_ctx(), (2048, "llama.cpp /props"))

    def test_ollama(self):
        FakeLLM.ollama = True
        self.addCleanup(setattr, FakeLLM, "ollama", False)
        c = mh.Client(self.url)
        self.assertEqual(c.pick_model(), "qwen:7b")
        c.model = "qwen:7b"
        self.assertEqual(c.server_ctx(), (8192, "Ollama /api/ps"))
        c.model = "other:1b"      # not loaded -> falls back to the Modelfile's num_ctx
        self.assertEqual(c.server_ctx(), (6144, "Ollama model parameters"))
        (self.root / "o.txt").write_text("x\n")
        FakeLLM.script = [{"tool": ("read_file", {"path": "o.txt"})}, {"content": "ok"}]
        a = mh.Agent(mh.parse_args(["--dir", str(self.root)]), c, self.tools, 4096)
        a.ctx_guess = True
        a.run_turn("read")
        self.assertEqual(a.n_ctx, 6144)
        self.assertEqual(FakeLLM.requests[1]["messages"][-1]["role"], "tool")

    def test_tool_loop_native(self):
        (self.root / "n.txt").write_text("hello world\n")
        FakeLLM.script = [{"tool": ("edit_file", {"path": "n.txt", "old": "world", "new": "there"})},
                          {"content": "done"}]
        a = mh.Agent(mh.parse_args(["--dir", str(self.root), "-y"]),
                     mh.Client(self.url), self.tools, 2048)
        a.run_turn("change world to there")
        self.assertEqual((self.root / "n.txt").read_text(), "hello there\n")
        self.assertEqual(len(FakeLLM.requests), 2)
        self.assertEqual(FakeLLM.requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(a.msgs[-1]["content"], "done")
        self.assertEqual(a.used, 110)

    def test_tool_loop_text(self):
        (self.root / "n.txt").write_text("abc\n")
        FakeLLM.script = [{"content": 'TOOL: {"name": "read_file", "arguments": {"path": "n.txt"}}'},
                          {"content": "fertig"}]
        a = mh.Agent(mh.parse_args(["--dir", str(self.root), "--protocol", "text"]),
                     mh.Client(self.url), self.tools, 2048)
        a.run_turn("read it")
        self.assertIn("abc", FakeLLM.requests[1]["messages"][-1]["content"])
        self.assertNotIn("tools", FakeLLM.requests[0])


class ConfigTests(Base):
    def write_ini(self, text):
        path = self.root / "miniharness.ini"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_precedence_and_relative_dir(self):
        ini = self.write_ini("[harness]\ndir = work\nctx = 3000\nprotocol = text\nauto_approve = true\n")
        a = mh.parse_args(["--config", ini])
        self.assertEqual((a.ctx, a.protocol, a.yes), (3000, "text", True))
        self.assertEqual(Path(a.dir), self.root.resolve() / "work")
        self.assertEqual(mh.parse_args(["--config", ini, "--ctx", "999"]).ctx, 999)

    def test_server_command(self):
        def cmd(body, ctx=4096, proto="native", url=""):
            conf = mh.Config(Path(self.write_ini(body)))
            return mh.server_command(mh.server_settings(conf, url, ctx, proto))
        argv, env = cmd("[server]\nstart = true\nmodel = m.gguf\nargs = --n-gpu-layers 99\n", ctx=8192)
        self.assertEqual(argv, ["llama-server", "-m", "m.gguf", "-c", "8192", "--host", "127.0.0.1",
                                "--port", "8080", "--jinja", "-np", "1", "--n-gpu-layers", "99"])
        argv, _ = cmd("[server]\nstart = true\nhf_repo = a/b\n", proto="text", url="http://x:9000/v1")
        self.assertIn("-hf", argv)
        self.assertNotIn("--jinja", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "9000")
        argv, env = cmd("[server]\nstart = true\ntype = ollama\nhost = 0.0.0.0\n", ctx=6000)
        self.assertEqual(argv, ["ollama", "serve"])
        self.assertEqual(env, {"OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_CONTEXT_LENGTH": "6000"})
        argv, _ = cmd("[server]\nstart = true\ntype = custom\ncommand = srv --p {port} -c {ctx}\n")
        self.assertEqual(argv, ["srv", "--p", "8080", "-c", "4096"])
        self.assertIsNone(mh.server_settings(mh.Config(Path(self.write_ini("[server]\nstart = false\n"))),
                                             "", 1, "native"))

    def test_start_and_stop_server(self):
        import socket
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        script = self.root / "fake_server.py"
        script.write_text("import sys, json\nfrom http.server import HTTPServer, BaseHTTPRequestHandler\n"
                          "class H(BaseHTTPRequestHandler):\n"
                          "    def do_GET(self):\n"
                          "        self.send_response(200); self.end_headers()\n"
                          "        self.wfile.write(json.dumps({'data': [{'id': 'm'}]}).encode())\n"
                          "    def log_message(self, *a): pass\n"
                          "HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n")
        conf = mh.Config(Path(self.write_ini("[server]\nstart = true\ntype = custom\ncommand = \"%s\" \"%s\" {port}\n"
                                             "timeout = 30\nlog = srv.log\n" % (sys.executable, script))))
        srv = mh.ServerProc(mh.server_settings(conf, "", 4096, "native"))
        srv.c["port"] = port
        srv.c["command"] = '"%s" "%s" %d' % (sys.executable, script, port)
        client = mh.Client("http://127.0.0.1:%d/v1" % port)
        self.addCleanup(srv.stop)
        srv.ensure(client)
        self.assertEqual(client.pick_model(), "m")
        self.assertTrue(srv.listening())
        srv.stop()
        self.assertFalse(srv.listening())


class HardwareTests(Base):
    def test_parsing(self):
        g = mh.parse_nvidia("NVIDIA RTX 4070, 1024, 12282, 37\nbroken line\n")
        self.assertEqual(g, [{"name": "NVIDIA RTX 4070", "used": 1024.0, "total": 12282.0, "util": 37.0}])
        info = mh.offload_info("offloaded 0/33 layers to GPU\noffloaded 33/33 layers to GPU\n"
                               "using device CUDA0 (NVIDIA RTX 4070) - 11000 MiB free")
        self.assertIn("offloaded 33/33 layers to GPU", info)
        self.assertIn("CUDA0", info)

    def test_gpu_and_thread_flags(self):
        def cmd(body, **kw):
            conf = mh.Config(Path(self._ini(body)))
            return mh.server_command(mh.server_settings(conf, "", 4096, "native"))[0]
        a = cmd("[server]\nstart = true\nmodel = m.gguf\ngpu_layers = all\nthreads = 6\n")
        self.assertEqual(a[a.index("-ngl") + 1], "999")
        self.assertEqual(a[a.index("-t") + 1], "6")
        self.assertEqual(cmd("[server]\nstart = true\nmodel = m.gguf\ngpu_layers = 0\n")[-2:], ["-ngl", "0"])
        a = cmd("[server]\nstart = true\nmodel = m.gguf\ngpu_layers = 5\nargs = -ngl 20\n")
        self.assertEqual(a.count("-ngl"), 1)       # explicit args win
        self.assertNotIn("-ngl", cmd("[server]\nstart = true\nmodel = m.gguf\n"))

    def _ini(self, body):
        path = self.root / "miniharness.ini"
        path.write_text(body, encoding="utf-8")
        return str(path)

    @unittest.skipIf(os.name == "nt", "uses a shell script as fake nvidia-smi")
    def test_monitor_with_fake_nvidia_smi(self):
        exe = self.root / "nvidia-smi"
        exe.write_text("#!/bin/sh\necho 'Fake GPU, 2048, 8192, 66'\n")
        exe.chmod(0o755)
        old = os.environ["PATH"]
        os.environ["PATH"] = str(self.root) + os.pathsep + old
        self.addCleanup(os.environ.__setitem__, "PATH", old)
        mon = mh.HwMonitor()
        self.assertEqual(len(mon.gpus), 1)
        mon.start()
        out = mon.stop()
        self.assertIn("GPU 66% peak, VRAM 2.0/8.0 GB", out)
        self.assertTrue(any("Fake GPU" in ln for ln in mon.describe()))


if __name__ == "__main__":
    unittest.main()

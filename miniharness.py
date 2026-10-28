#!/usr/bin/env python3
"""miniharness - small agent harness for local LLMs (llama.cpp / OpenAI-compatible).

Reads and edits text files inside one folder, built for SMALL context windows.
Only the Python standard library is used (Python >= 3.8, Linux / macOS / Windows).
"""
import argparse
import codecs
import difflib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:  # line editing + history on Linux/macOS; harmless if missing (Windows)
    import readline  # noqa: F401
except ImportError:
    pass

DEFAULT_URL = "http://127.0.0.1:8080/v1"
DEFAULT_CTX = 4096
MAX_FILE_BYTES = 4_000_000
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv"}


# ----------------------------------------------------------------- terminal
class Term:
    color = False

    @staticmethod
    def setup():
        for s in (sys.stdout, sys.stderr):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
        Term.color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        if Term.color and os.name == "nt":
            os.system("")  # enables ANSI escape processing in the Windows console


def paint(code, s):
    return "\033[%sm%s\033[0m" % (code, s) if Term.color else s


def dim(s):
    return paint("2", s)


def say(s="", end="\n"):
    print(s, end=end, flush=True)


# ------------------------------------------------------------------- tools
class ToolError(Exception):
    pass


def _spec(name, desc, params, required=()):
    props = {k: {"type": t, **({"description": d} if d else {})} for k, (t, d) in params.items()}
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


TOOL_SPECS = [
    _spec("list_files", "List files/folders (path default '.').",
          {"path": ("string", ""), "recursive": ("boolean", "")}),
    _spec("read_file", "Read a text file in chunks; continue with start_line.",
          {"path": ("string", ""), "start_line": ("integer", ""), "max_lines": ("integer", "")},
          ["path"]),
    _spec("search", "Regex search in a file/folder; returns path:line: text.",
          {"pattern": ("string", ""), "path": ("string", "")}, ["pattern"]),
    _spec("edit_file", "Replace one exact, unique snippet 'old' with 'new'.",
          {"path": ("string", ""), "old": ("string", ""), "new": ("string", "")},
          ["path", "old", "new"]),
    _spec("write_file", "Create/overwrite a file, or append=true to append.",
          {"path": ("string", ""), "content": ("string", ""), "append": ("boolean", "")},
          ["path", "content"]),
]


def make_diff(old, new, name, limit=40):
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                      "a/" + name, "b/" + name, lineterm="", n=2))
    out = []
    for ln in lines[:limit]:
        if ln.startswith("+") and not ln.startswith("+++"):
            ln = paint("32", ln)
        elif ln.startswith("-") and not ln.startswith("---"):
            ln = paint("31", ln)
        else:
            ln = dim(ln)
        out.append(ln)
    if len(lines) > limit:
        out.append(dim("... (%d more diff lines)" % (len(lines) - limit)))
    return "\n".join(out)


class Tools:
    def __init__(self, root, confirm, cap):
        self.root = Path(root).resolve()
        self.confirm = confirm      # confirm(title, preview) -> bool
        self.cap = cap              # cap() -> max chars of one tool result
        self.table = {s["function"]["name"]: getattr(self, s["function"]["name"])
                      for s in TOOL_SPECS}

    # -- helpers
    def resolve(self, p):
        full = (self.root / str(p or ".").strip()).resolve()
        try:
            full.relative_to(self.root)
        except ValueError:
            raise ToolError("path is outside the project folder")
        return full

    def rel(self, full):
        r = full.relative_to(self.root).as_posix()
        return r if r != "." else "."

    @staticmethod
    def load(full):
        if full.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("file too large")
        data = full.read_bytes()
        if b"\0" in data[:8192]:
            raise ToolError("binary file")
        bom = data.startswith(codecs.BOM_UTF8)
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ToolError("not a UTF-8 text file")
        return text.replace("\r\n", "\n"), "\r\n" in text, bom

    @staticmethod
    def save(full, text, crlf, bom):
        if crlf:
            text = text.replace("\n", "\r\n")
        data = text.encode("utf-8")
        if bom:
            data = codecs.BOM_UTF8 + data
        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_name(full.name + ".mh-tmp")
        tmp.write_bytes(data)
        os.replace(tmp, full)

    def walk(self, base):
        for dirpath, dirs, files in os.walk(str(base)):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(files):
                yield Path(dirpath) / f

    # -- tools
    def list_files(self, path=".", recursive=False):
        base = self.resolve(path)
        if not base.is_dir():
            raise ToolError("not a folder: " + self.rel(base))
        out = []
        if recursive:
            for f in self.walk(base):
                out.append("%s (%s)" % (self.rel(f), _size(f)))
                if len(out) >= 200:
                    break
        else:
            for e in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if e.name in SKIP_DIRS:
                    continue
                out.append(e.name + "/" if e.is_dir() else "%s (%s)" % (e.name, _size(e)))
                if len(out) >= 200:
                    break
            if len(out) >= 200:
                out.append("... (list cut at 200 entries)")
        return "\n".join(out) or "[empty folder]"

    def read_file(self, path, start_line=1, max_lines=None):
        full = self.resolve(path)
        if not full.is_file():
            raise ToolError("no such file: " + str(path))
        text = self.load(full)[0]
        if not text:
            return "[%s is empty]" % self.rel(full)
        lines = text.split("\n")
        if lines[-1] == "":
            lines.pop()
        total = len(lines)
        start = max(1, _int(start_line, 1))
        limit = _int(max_lines, 0) or None
        if start > total:
            raise ToolError("start_line %d is beyond end of file (%d lines)" % (start, total))
        cap, used, out, i = self.cap(), 0, [], start - 1
        while i < total and (limit is None or len(out) < limit):
            ln = lines[i]
            if len(ln) > cap:
                ln = ln[:cap] + " ...[line cut]"
            if out and used + len(ln) + 1 > cap:
                break
            out.append(ln)
            used += len(ln) + 1
            i += 1
        tail = ("[more: call read_file with start_line=%d]" % (i + 1)) if i < total else "[end of file]"
        return "[%s lines %d-%d of %d]\n%s\n%s" % (self.rel(full), start, i, total, "\n".join(out), tail)

    def search(self, pattern, path="."):
        base = self.resolve(path)
        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I)
        files = [base] if base.is_file() else self.walk(base)
        out, scanned = [], 0
        for f in files:
            scanned += 1
            if scanned > 2000 or len(out) >= 40:
                break
            try:
                text = self.load(f)[0]
            except (ToolError, OSError):
                continue
            for n, ln in enumerate(text.split("\n"), 1):
                if rx.search(ln):
                    out.append("%s:%d: %s" % (self.rel(f), n, ln.strip()[:160]))
                    if len(out) >= 40:
                        out.append("... (stopped at 40 matches)")
                        break
        return "\n".join(out) or "[no matches]"

    def edit_file(self, path, old, new):
        full = self.resolve(path)
        if not full.is_file():
            raise ToolError("no such file: " + str(path))
        text, crlf, bom = self.load(full)
        old, new = str(old).replace("\r\n", "\n"), str(new).replace("\r\n", "\n")
        if not old:
            raise ToolError("'old' must not be empty (use write_file to create content)")
        n = text.count(old)
        if n == 0:
            first = old.split("\n")[0].strip()
            at = next((i for i, ln in enumerate(text.split("\n"), 1) if first and first in ln), None)
            hint = (" Its first line appears at line %d, but the full text differs; "
                    "read_file that region and copy it exactly." % at) if at else ""
            raise ToolError("'old' text not found." + hint)
        if n > 1:
            raise ToolError("'old' occurs %d times; include more surrounding text to make it unique" % n)
        result = text.replace(old, new, 1)
        if not self.confirm("edit " + self.rel(full), make_diff(text, result, self.rel(full))):
            return "Declined by user."
        self.save(full, result, crlf, bom)
        return "OK: edited %s" % self.rel(full)

    def write_file(self, path, content, append=False):
        full = self.resolve(path)
        if full.is_dir():
            raise ToolError("is a folder: " + str(path))
        old, crlf, bom = (self.load(full) if full.exists() else ("", False, False))
        content = str(content).replace("\r\n", "\n")
        if append and old and not old.endswith("\n"):
            old_part = old + "\n"
        else:
            old_part = old
        new = old_part + content if append else content
        title = ("append to " if append else "write ") + self.rel(full)
        if not full.exists():
            preview = dim("(new file, %d lines)\n" % len(new.splitlines())) + \
                "\n".join(paint("32", "+" + ln) for ln in new.splitlines()[:20])
        else:
            preview = make_diff(old, new, self.rel(full))
        if not self.confirm(title, preview):
            return "Declined by user."
        self.save(full, new, crlf, bom)
        return "OK: %s now has %d lines" % (self.rel(full), len(new.splitlines()))


def _int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _size(p):
    try:
        n = p.stat().st_size
    except OSError:
        return "?"
    return "%d B" % n if n < 1024 else "%.1f KB" % (n / 1024) if n < 1 << 20 else "%.1f MB" % (n / (1 << 20))


# ------------------------------------------------------------------ client
class LLMError(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status

    @property
    def is_context(self):
        m = str(self).lower()
        return self.status == 400 and ("context" in m or "exceed" in m or "too long" in m)


class Client:
    def __init__(self, base, key="", model="", timeout=600):
        self.base = base.rstrip("/")
        self.key, self.model, self.timeout = key, model or "local", timeout
        host = urllib.parse.urlparse(self.base).hostname or ""
        local = host in ("localhost", "127.0.0.1", "::1")
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}) if local else urllib.request.ProxyHandler())

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def _open(self, url, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers())
        try:
            return self.opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                err = json.loads(raw).get("error", raw)
                raw = err.get("message", raw) if isinstance(err, dict) else str(err)
            except (ValueError, AttributeError):
                pass
            raise LLMError("HTTP %d: %s" % (e.code, raw.strip()[:300]), e.code)
        except urllib.error.URLError as e:
            raise LLMError("server not reachable (%s): %s" % (self.base, e.reason))

    def server_ctx(self):
        """Context size the llama.cpp server runs with, or None."""
        u = urllib.parse.urlparse(self.base)
        try:
            with self._open("%s://%s/props" % (u.scheme, u.netloc)) as r:
                props = json.loads(r.read().decode("utf-8"))
            n = props["default_generation_settings"]["n_ctx"]
            return int(n) if n else None
        except (LLMError, ValueError, KeyError, TypeError, OSError):
            return None

    def chat(self, messages, tools, max_tokens, temperature, on_text=None, on_think=None):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = tools
        content, calls, finish, usage = [], {}, None, {}
        with self._open(self.base + "/chat/completions", body) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("error"):
                    err = ev["error"]
                    raise LLMError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
                usage = ev.get("usage") or usage
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("reasoning_content") and on_think:
                        on_think(d["reasoning_content"])
                    if d.get("content"):
                        content.append(d["content"])
                        if on_text:
                            on_text(d["content"])
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        c["id"] = tc.get("id") or c["id"]
                        fn = tc.get("function") or {}
                        c["name"] = c["name"] or fn.get("name") or ""
                        a = fn.get("arguments", "")
                        c["args"] += a if isinstance(a, str) else json.dumps(a)
                    finish = ch.get("finish_reason") or finish
        return {"content": "".join(content), "calls": [calls[k] for k in sorted(calls)],
                "finish": finish, "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}


# ------------------------------------------------------------------- agent
class ContextError(Exception):
    pass


def split_turns(msgs):
    """msgs -> turns -> units. A unit (assistant msg + its tool results, or a user msg)
    is never split; a turn starts at a real user message."""
    turns = []
    for m in msgs:
        cont = m["role"] == "tool" or m.get("_cont")
        if m["role"] == "user" and not cont:
            turns.append([[m]])
        elif cont and turns and turns[-1]:
            turns[-1][-1].append(m)
        else:
            if not turns:
                turns.append([])
            turns[-1].append([m])
    return turns


def flatten(turns):
    return [m for t in turns for u in t for m in u]


def render(msgs):
    out = []
    for m in msgs:
        c = (m.get("content") or "").strip()
        for t in m.get("tool_calls") or []:
            c += " [call %s %s]" % (t["function"]["name"], t["function"]["arguments"][:200])
        role = "tool result" if (m["role"] == "tool" or m.get("_cont")) else m["role"]
        out.append("%s: %s" % (role, c[:400]))
    return "\n".join(out)


def parse_text_calls(text):
    """Text protocol: `TOOL: {"name": ..., "arguments": {...}}` anywhere in the reply."""
    calls, dec, pos = [], json.JSONDecoder(), 0
    while True:
        i = text.find("TOOL:", pos)
        if i < 0:
            return calls
        j = text.find("{", i)
        try:
            if j < 0:
                raise ValueError
            obj, pos = dec.raw_decode(text, j)
        except ValueError:
            calls.append({"id": "", "name": "?", "args": None,
                          "error": "invalid or cut-off TOOL JSON. Use one line: "
                                   'TOOL: {"name": "...", "arguments": {...}}'})
            return calls
        if isinstance(obj, dict) and obj.get("name"):
            a = obj.get("arguments", obj.get("args", {}))
            calls.append({"id": "", "name": obj["name"], "args": a if isinstance(a, dict) else {}})


class Agent:
    def __init__(self, args, client, tools, n_ctx):
        self.a, self.client, self.tools = args, client, tools
        self.n_ctx = n_ctx
        self.native = args.protocol == "native"
        self.msgs, self.summary = [], ""
        self.ratio = 3.0            # chars per token, calibrated from server usage
        self.used = 0               # tokens of the last request (prompt + reply)
        self.tool_json = json.dumps(TOOL_SPECS, separators=(",", ":")) if self.native else ""

    # -- sizes
    @property
    def reserve(self):
        return self.a.max_tokens or max(256, min(2048, self.n_ctx // 4))

    def budget(self):
        return self.n_ctx - self.reserve - 64

    def tool_cap(self):
        return max(400, int(self.n_ctx * self.ratio * 0.2))

    def system(self):
        s = ("You are a careful assistant that reads and edits text files in the user's project folder "
             "using tools. The context window is small (%d tokens), so:\n"
             "- Don't read more than needed: use list_files, search, and read_file with start_line.\n"
             "- Prefer edit_file with a short unique snippet over rewriting whole files.\n"
             "- Write long files in parts: write_file, then write_file with append=true.\n"
             "- Don't repeat file contents in your answer; reply briefly. "
             "Answer in the user's language.\n"
             "Paths are relative to the project folder." % self.n_ctx)
        if not self.native:
            s += ("\n\nTo use a tool, write one line: TOOL: {\"name\": \"<tool>\", \"arguments\": {...}}\n"
                  "Then stop and wait for the result. Tools:\n")
            for t in TOOL_SPECS:
                f = t["function"]
                req = f["parameters"]["required"]
                args = ", ".join(k + ("" if k in req else "?") for k in f["parameters"]["properties"])
                s += "- %s(%s): %s\n" % (f["name"], args, f["description"])
        if self.summary:
            s += "\n\nNotes on the earlier conversation (older messages were dropped):\n" + self.summary
        return s

    def chars(self, msgs):
        n = len(self.system()) + len(self.tool_json)
        for m in msgs:
            n += len(m.get("content") or "") + 16
            if m.get("tool_calls"):
                n += len(json.dumps(m["tool_calls"]))
        return n

    def est(self):
        return int(self.chars(self.msgs) / self.ratio)

    # -- context fitting
    def elide_old(self):
        units = [u for t in split_turns(self.msgs) for u in t]
        for u in units[:-2]:
            for m in u:
                if (m["role"] == "tool" or m.get("_cont")) and len(m["content"]) > 160:
                    m["content"] = m["content"][:120] + " ...[old output elided]"
                for tc in m.get("tool_calls") or []:
                    try:
                        a = json.loads(tc["function"]["arguments"])
                        a = {k: (v[:100] + " ...[elided]" if isinstance(v, str) and len(v) > 200 else v)
                             for k, v in a.items()}
                        tc["function"]["arguments"] = json.dumps(a)
                    except (ValueError, AttributeError):
                        pass

    def fit(self):
        budget = self.budget()
        dropped = []
        if self.est() > budget:
            self.elide_old()
        while self.est() > budget:
            turns = split_turns(self.msgs)
            if len(turns) > 1:
                dropped += turns[0]
                self.msgs = flatten(turns[1:])
            elif len(turns[0]) > 2:        # user msg + >=2 units: drop the oldest unit after the user msg
                dropped.append(turns[0][1])
                self.msgs = flatten([[turns[0][0]]] + [turns[0][2:]])
            else:
                break
        if dropped:
            self.absorb(flatten(dropped))
        over = self.est() - budget
        if over > 0:                        # a single message is too big: cut it
            last = self.msgs[-1]
            cut = int(over * self.ratio) + 200
            if len(last.get("content") or "") > cut + 100:
                last["content"] = last["content"][:-cut] + "\n...[cut to fit the context window]"
            else:
                raise ContextError("input does not fit into the context window (%d tokens); "
                                   "raise it with /ctx or use smaller pieces" % self.n_ctx)

    def absorb(self, dropped):
        """Fold dropped messages into the rolling summary."""
        say(dim("[context full: %d old messages dropped%s]" %
                (len(dropped), " and summarized" if self.a.summarize else "")))
        if not self.a.summarize:
            return
        text = render([m for m in dropped if m["role"] != "system"])
        text = text[-int(self.n_ctx * self.ratio * 0.5):]
        prompt = ("Notes so far:\n%s\n\nNew events:\n%s\n\nRewrite the notes so they include the new events. "
                  "Max 150 words: user goal, files read/changed and how, open tasks, key facts. "
                  "Output only the notes." % (self.summary or "(none)", text))
        try:
            res = self.client.chat(
                [{"role": "system", "content": "You compress conversation history into short notes."},
                 {"role": "user", "content": prompt}],
                None, min(400, max(128, self.n_ctx // 6)), 0.2)
            if res["content"].strip():
                self.summary = res["content"].strip()[:int(self.n_ctx * self.ratio * 0.15)]
        except LLMError as e:
            say(dim("[summary failed: %s]" % e))

    def clear(self):
        self.msgs, self.summary, self.used = [], "", 0

    def compact(self):
        old, self.msgs = self.msgs, []
        if old:
            self.absorb(old)
        else:
            say("nothing to compact")

    # -- model call
    def call(self):
        for attempt in range(3):
            self.fit()
            payload = [{"role": "system", "content": self.system()}] + \
                [{k: v for k, v in m.items() if not k.startswith("_")} for m in self.msgs]
            sent = self.chars(self.msgs)
            started = [False]

            def on_text(t):
                started[0] = True
                say(t, end="")

            try:
                res = self.client.chat(payload, TOOL_SPECS if self.native else None, self.reserve,
                                       self.a.temperature, on_text, lambda t: say(dim(t), end=""))
            except LLMError as e:
                if e.is_context and attempt < 2:
                    self.n_ctx = int(self.n_ctx * 0.75)
                    say(dim("[server rejected the size, assuming context %d]" % self.n_ctx))
                    continue
                raise
            if started[0]:
                say()
            pt = res["prompt_tokens"]
            if pt:
                self.ratio = min(6.0, max(1.5, 0.5 * self.ratio + 0.5 * sent / pt))
                self.used = pt + (res["completion_tokens"] or 0)
            else:
                self.used = self.est()
            return res

    def execute(self, c):
        if c.get("error"):
            return "ERROR: " + c["error"]
        fn = self.tools.table.get(c["name"])
        if not fn:
            return "ERROR: unknown tool '%s'. Tools: %s" % (c["name"], ", ".join(self.tools.table))
        try:
            out = fn(**c["args"])
        except TypeError as e:
            return "ERROR: bad arguments: %s" % e
        except ToolError as e:
            return "ERROR: %s" % e
        except OSError as e:
            return "ERROR: %s" % (e.strerror or e)
        cap = self.tool_cap()
        if len(out) > cap:
            out = out[:cap] + "\n...[output cut, %d chars]" % (len(out) - cap)
        return out

    def run_turn(self, text):
        self.msgs.append({"role": "user", "content": text})
        try:
            for _ in range(self.a.max_steps):
                res = self.call()
                calls = self.collect_calls(res)
                msg = {"role": "assistant", "content": res["content"]}
                if self.native and calls:
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": c["raw"]}}
                                         for c in calls]
                self.msgs.append(msg)
                if not calls:
                    if res["finish"] == "length":
                        say(paint("33", "[reply cut off: max-tokens reached]"))
                    return
                for c in calls:
                    say(dim("→ " + _fmt_call(c)))
                    out = self.execute(c)
                    say(dim("  " + out.split("\n")[0][:100] + (" ..." if "\n" in out else "")))
                    if self.native:
                        self.msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
                    else:
                        self.msgs.append({"role": "user", "_cont": True,
                                          "content": "TOOL RESULT %s:\n%s" % (c["name"], out)})
            say(paint("33", "[step limit reached; continue with a new message]"))
        except KeyboardInterrupt:
            say(paint("33", "\n[interrupted]"))
            if self.msgs[-1]["role"] != "assistant":
                self.msgs.append({"role": "assistant", "content": "(interrupted)"})
        except (LLMError, ContextError) as e:
            say(paint("31", "error: %s" % e))
            if self.msgs and self.msgs[-1]["role"] == "user" and not self.msgs[-1].get("_cont"):
                self.msgs.pop()  # let the user retry without a dangling message

    def collect_calls(self, res):
        if not self.native:
            return [dict(c, raw="") for c in parse_text_calls(res["content"])]
        calls = []
        for i, c in enumerate(res["calls"]):
            c = dict(c, id=c["id"] or "call_%d" % len(self.msgs) + "_%d" % i, raw=c["args"] or "{}")
            try:
                c["args"] = json.loads(c["raw"])
                if not isinstance(c["args"], dict):
                    raise ValueError
            except ValueError:
                c["args"] = None
                c["error"] = ("arguments are not valid JSON%s. Use smaller pieces "
                              "(edit_file with a short snippet, or write_file with append=true)." %
                              (" (your reply hit the token limit)" if res["finish"] == "length" else ""))
                c["raw"] = "{}"
            calls.append(c)
        return calls


def _fmt_call(c):
    a = c.get("args") or {}
    return "%s(%s)" % (c["name"], ", ".join(
        "%s=%s" % (k, json.dumps(v, ensure_ascii=False)[:40]) for k, v in a.items()))


# --------------------------------------------------------------------- cli
HELP = """\
/ctx [n]    show / set context window (tokens)
/compact    summarize the conversation to free context
/clear      forget the conversation
/auto       toggle auto-approve of file changes
/help       this help
/quit       exit  (Ctrl+C aborts a running reply)
End a line with \\ for multi-line input."""


def read_input():
    lines = []
    while True:
        line = input("› " if not lines else "… ")
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def parse_args(argv=None):
    env = os.environ.get
    ap = argparse.ArgumentParser(
        prog="miniharness",
        description="Small agent harness for local LLMs (llama.cpp / OpenAI-compatible API).")
    ap.add_argument("-d", "--dir", default=".", help="project folder the model may read/edit (default: .)")
    ap.add_argument("-u", "--url", default=env("MINIHARNESS_URL", DEFAULT_URL),
                    help="OpenAI-compatible base URL (default: %s)" % DEFAULT_URL)
    ap.add_argument("-m", "--model", default=env("MINIHARNESS_MODEL", ""), help="model name (optional for llama.cpp)")
    ap.add_argument("--api-key", default=env("MINIHARNESS_API_KEY", env("OPENAI_API_KEY", "")))
    ap.add_argument("-c", "--ctx", type=int, default=_int(env("MINIHARNESS_CTX"), 0),
                    help="context window in tokens; default: read from server (/props), else %d" % DEFAULT_CTX)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="tokens reserved for one reply (default: ctx/4, 256..2048)")
    ap.add_argument("--protocol", choices=["native", "text"], default="native",
                    help="native: OpenAI tool calls (llama-server --jinja); text: plain-text tool protocol")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=20, help="max tool rounds per message")
    ap.add_argument("--no-summary", dest="summarize", action="store_false",
                    help="drop old messages without summarizing them")
    ap.add_argument("-y", "--yes", action="store_true", help="apply file changes without asking")
    ap.add_argument("-p", "--prompt", help="run a single prompt and exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    Term.setup()
    root = Path(args.dir)
    if not root.is_dir():
        sys.exit("not a folder: %s" % args.dir)
    client = Client(args.url, args.api_key, args.model)

    n_ctx, src = args.ctx, "--ctx / MINIHARNESS_CTX"
    if not n_ctx:
        n_ctx, src = client.server_ctx(), "server /props"
    if not n_ctx:
        n_ctx, src = DEFAULT_CTX, "default (server did not tell; use --ctx)"

    state = {"auto": args.yes}

    def confirm(title, preview):
        say(paint("36", title))
        say(preview)
        if state["auto"]:
            return True
        try:
            ans = input("apply? [y/N/a=always] ").strip().lower()
        except EOFError:
            return False
        if ans in ("a", "always"):
            state["auto"] = True
        return ans in ("y", "yes", "j", "ja", "a", "always")

    agent = Agent(args, client, None, n_ctx)
    agent.tools = Tools(root, confirm, agent.tool_cap)
    say(paint("1", "miniharness") + dim(" · %s · folder %s · context %d tokens (%s) · reply reserve %d" %
                                        (args.url, agent.tools.root, n_ctx, src, agent.reserve)))
    fixed = agent.chars([]) / agent.ratio
    if fixed > 0.4 * agent.budget():
        say(paint("33", "warning: system prompt + tools use ~%d of %d tokens; "
                        "try --protocol text or a larger context" % (fixed, n_ctx)))

    if args.prompt:
        agent.run_turn(args.prompt)
        return
    say(dim("/help for commands, Ctrl+D to quit"))
    while True:
        try:
            line = read_input()
        except EOFError:
            say()
            break
        except KeyboardInterrupt:
            say(dim("\n(use /quit or Ctrl+D to exit)"))
            continue
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                say(HELP)
            elif cmd == "/clear":
                agent.clear()
                say(dim("cleared"))
            elif cmd == "/compact":
                agent.compact()
            elif cmd == "/auto":
                state["auto"] = not state["auto"]
                say("auto-approve " + ("on" if state["auto"] else "off"))
            elif cmd == "/ctx":
                if arg.strip().isdigit():
                    agent.n_ctx = int(arg)
                say("context %d tokens, last request used ~%d (%d%%)" %
                    (agent.n_ctx, agent.used, 100 * agent.used // max(1, agent.n_ctx)))
            else:
                say("unknown command, see /help")
            continue
        agent.run_turn(line)
        say(dim("[context %d/%d · %d%%]" % (agent.used, agent.n_ctx, 100 * agent.used // max(1, agent.n_ctx))))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

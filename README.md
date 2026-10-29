# miniharness

Schlanker Terminal-Harness für ein **lokales LLM** (llama.cpp, Ollama oder jede OpenAI-kompatible API).
Das Modell kann Textdateien in **einem** Ordner auflisten, lesen, durchsuchen und bearbeiten –
ausgelegt auf **kleine Kontextfenster**. Eine Datei, nur Python-Standardbibliothek
(Python ≥ 3.8), läuft unter Linux, macOS und Windows.

## Start

```bash
# 1. Server starten (llama.cpp). --jinja wird für native Tool-Calls benötigt.
llama-server -m modell.gguf -c 8192 --jinja

# 2. Harness im Projektordner starten
python miniharness.py --dir ./mein-ordner
```

Windows: `py miniharness.py --dir C:\pfad\ordner` (Port, URL usw. wie unten).

## Config-Datei: Server automatisch starten

```bash
python miniharness.py --init      # legt eine kommentierte miniharness.ini an
# ini anpassen (mindestens [server] model = Pfad/zur/datei.gguf), dann nur noch:
python miniharness.py
```

Die Datei wird in dieser Reihenfolge gesucht: `--config PFAD`, `MINIHARNESS_CONFIG`, aktueller Ordner,
Ordner des Skripts. Relative Pfade in der Datei gelten relativ zur Datei. Vorrang: Kommandozeile >
Umgebungsvariable (`MINIHARNESS_*`) > Config.

```ini
[harness]
dir = .
ctx = 8192            ; gilt für Harness UND den gestarteten Server
protocol = native

[server]
start = true
type = llama.cpp      ; llama.cpp | ollama | custom
model = C:\modelle\qwen.gguf     ; oder hf_repo = Org/Repo-GGUF
port = 8080
args = --n-gpu-layers 99
```

- **llama.cpp:** gestartet wird `llama-server -m … -c <ctx> --host … --port … --jinja -np 1 <args>`
  (`--jinja` nur bei `protocol = native`; `-np 1`, damit das ganze Fenster dem Harness gehört).
  Liegt `llama-server` nicht im PATH, `command = /voller/pfad/llama-server` setzen.
- **Ollama:** `ollama serve` mit `OLLAMA_CONTEXT_LENGTH=<ctx>` und `OLLAMA_HOST`; `model` ist der Modellname
  (vorher `ollama pull`).
- **custom:** beliebiger Befehl in `command`, `{host}`, `{port}`, `{ctx}` werden ersetzt.
- Lauscht auf dem Port schon ein Server, wird dieser benutzt und nicht neu gestartet (dann gelten
  dessen Einstellungen). Der selbst gestartete Server wird beim Beenden wieder gestoppt
  (Strg+C bricht nur die laufende Antwort ab, nicht den Server). Wer den Server dauerhaft laufen
  lassen will, startet ihn einfach selbst.
- Serverausgabe steht in `miniharness-server.log`; startet der Server nicht, werden die letzten Zeilen gezeigt.
- `--no-server` ignoriert den `[server]`-Abschnitt. Für Zugriff aus dem Netz `host = 0.0.0.0` setzen.

## Server im lokalen Netzwerk

Das Modell kann auf einem anderen Rechner laufen:

```bash
# Server-Rechner: auf allen Interfaces lauschen (Standard ist nur 127.0.0.1)
llama-server -m modell.gguf -c 8192 --jinja --host 0.0.0.0 --port 8080 --api-key geheim

# Client-Rechner
python miniharness.py --url http://192.168.1.50:8080/v1 --api-key geheim
```

Port 8080 muss in der Firewall des Servers freigegeben sein. Ohne `--api-key` kann jeder im Netz
den Server nutzen. Für lokale/private Adressen werden System-Proxys (`HTTP_PROXY`) umgangen;
`https://`-URLs funktionieren ebenfalls.

## Ollama

Ollama bietet eine OpenAI-kompatible API unter `/v1`:

```bash
ollama pull qwen2.5-coder:7b          # Modell mit Tool-Support wählen
python miniharness.py --url http://127.0.0.1:11434/v1 -m qwen2.5-coder:7b --ctx 8192
```

- Ohne `-m` wird das erste Modell des Servers genommen (steht in der Startzeile).
- **Wichtig – Kontext:** Ollama kürzt zu lange Prompts *stillschweigend* (vorne, also inklusive
  System-Prompt). Das Fenster, mit dem Ollama läuft, wird über die OpenAI-API **nicht** pro Anfrage
  gesetzt. Stellen Sie es serverseitig ein, z. B. `OLLAMA_CONTEXT_LENGTH=8192 ollama serve` oder in einer
  Modelfile `PARAMETER num_ctx 8192`, und geben Sie dasselbe mit `--ctx` an.
  Ohne `--ctx` liest das Harness den Wert aus `/api/ps` bzw. dem `num_ctx` der Modelfile; sonst rät es 4096
  und korrigiert sich nach der ersten Antwort, sobald Ollama das geladene Modell meldet.
  Läuft der Server mit weniger als `--ctx`, erscheint eine Warnung.
- Modelle ohne Tool-Support melden einen Fehler – dann `--protocol text` verwenden.
- Im Netzwerk: `OLLAMA_HOST=0.0.0.0 ollama serve`, dann wie oben mit der Server-IP und Port 11434.

## Kontextfenster

Reihenfolge, in der die Größe bestimmt wird:

1. `--ctx N` bzw. Umgebungsvariable `MINIHARNESS_CTX`
2. automatisch vom Server (llama.cpp `/props`, Ollama `/api/ps` bzw. `num_ctx`)
3. Fallback 4096 (mit Hinweis)

Zur Laufzeit mit `/ctx 8192` änderbar. Das Harness hält jede Anfrage unter
`ctx - Antwortreserve` (`--max-tokens`, Standard ctx/4, 256–2048):

- Tokens werden geschätzt und anhand der echten `usage`-Werte des Servers laufend kalibriert.
- Passt es nicht mehr: alte Tool-Ausgaben und große `write_file`-Argumente werden gekürzt,
  dann die ältesten Nachrichten verworfen und per Modell zu kurzen **Notizen** zusammengefasst
  (`--no-summary` schaltet das ab). `/compact` macht das manuell.
- Tool-Ausgaben sind auf ~20 % des Fensters begrenzt; `read_file` liefert Dateien stückweise
  (`start_line`), `search` findet Stellen ohne ganze Dateien zu lesen.
- Lehnt der Server eine Anfrage wegen Größe ab, wird das Fenster automatisch verkleinert und neu versucht.

## Tools des Modells

`list_files`, `read_file`, `search`, `edit_file` (eindeutiger Textausschnitt ersetzen),
`write_file` (überschreiben oder `append`). Zugriff nur innerhalb von `--dir`, nur UTF-8-Textdateien
(CRLF/BOM bleiben erhalten). Änderungen werden als Diff angezeigt und müssen bestätigt werden
(`a` = immer, `-y` / `/auto` = automatisch).

## Optionen

| Option | Bedeutung |
|---|---|
| `-d, --dir` | Ordner, den das Modell sehen darf (Standard `.`) |
| `-u, --url` | API-Basis-URL (Standard `http://127.0.0.1:8080/v1`, Env `MINIHARNESS_URL`) |
| `-m, --model` / `--api-key` | optional (Env `MINIHARNESS_MODEL`, `MINIHARNESS_API_KEY`) |
| `-c, --ctx` | Kontextfenster in Tokens |
| `--max-tokens` | Reserve für eine Antwort |
| `--protocol text` | Tool-Aufrufe als Text (`TOOL: {...}`) – für Server ohne `--jinja`/Tool-Support und spart Platz bei sehr kleinen Fenstern |
| `-p, --prompt` | einen Auftrag ausführen und beenden |
| `-y, --yes` | Änderungen ohne Rückfrage übernehmen |

Befehle im Programm: `/ctx [n]`, `/compact`, `/clear`, `/auto`, `/help`, `/quit`.
Strg+C bricht eine laufende Antwort ab; Zeile mit `\` am Ende = mehrzeilige Eingabe.

## Tests

```bash
python -m unittest discover -s tests
```

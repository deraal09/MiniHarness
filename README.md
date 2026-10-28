# miniharness

Schlanker Terminal-Harness für ein **lokales LLM** (llama.cpp oder jede OpenAI-kompatible API).
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

## Kontextfenster

Reihenfolge, in der die Größe bestimmt wird:

1. `--ctx N` bzw. Umgebungsvariable `MINIHARNESS_CTX`
2. automatisch vom llama.cpp-Server (`/props`)
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

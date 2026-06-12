# baish — AI Shell

A Unix shell where bash commands, AI agents, and AI functions compose naturally. Plain text goes to bash; `?` asks the agent; `/fn` calls an AI function; `cmd | /fn` pipes bash output into AI.

```
~/project $ git diff | ?review this for bugs
~/project $ cat error.log | ?why is this failing
~/project $ cat schema.sql | /extract --type table-names
~/project $ cat ARCHITECTURE.md | /ctx add arch
~/project $ ?given the arch, what needs to change for multi-tenancy
```

---

## Requirements

- Go 1.24+
- [Ollama](https://ollama.ai) running locally with at least one model pulled
- Linux or macOS (WSL2 supported)

---

## Build and Install

```bash
cd POC/ai-shell
make install          # builds and copies binary to ~/.local/bin/baish
```

Or build without installing:
```bash
make build            # outputs ./bin/baish
```

Ensure `~/.local/bin` is in your `PATH`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then run:
```bash
baish
```

---

## Configuration

On first run, baish writes defaults to `~/.config/baish/config.toml`:

```toml
max_history_messages = 20
tool_output_max_chars = 4000
tool_output_overflow = "truncate"
ctx_max_inject_chars = 8000
```

Edit the file directly or use the `/config` command:

```
/config                              # show all settings
/config set max_history_messages 30
/config set tool_output_overflow summarize
/config set ctx_max_inject_chars 16000
/config reset                        # restore defaults
```

| Key | Default | Description |
|---|---|---|
| `max_history_messages` | 20 | Agent conversation turns kept (aligned to user-message boundaries) |
| `tool_output_max_chars` | 4000 | Characters before tool output is truncated/summarized |
| `tool_output_overflow` | `truncate` | What to do when output exceeds cap: `truncate` or `summarize` (extra LLM call) |
| `ctx_max_inject_chars` | 8000 | Max chars per context slot injected into agent system prompt |

---

## Input Routing

| Input | Routes to | Example |
|---|---|---|
| Anything else | bash | `ls -la`, `git status` |
| `?message` | AI agent | `?explain this error` |
| `/fn [args]` | AI function | `/summarize --length short` |
| `cmd \| /fn` | bash → AI function | `cat file.txt \| /summarize` |
| `cmd \| ?question` | bash output → agent | `cat error.log \| ?why is this failing` |
| `any input &` | background job | `?analyze codebase & ` |

The parser is quote-aware and won't mistake absolute paths like `/usr/bin/grep` for AI functions.

---

## AI Agent (`?`)

```
?<message>
```

The agent has access to a `bash` tool it can use to run shell commands and iterate toward an answer. It sees the current directory and your active context slots.

```
~/project $ ?list all Go files with more than 200 lines
~/project $ ?what does the pipeline package do
~/project $ git diff | ?are there any issues with this change
```

Agent history is kept for `max_history_messages` turns. Clear it with `/history clear`.

---

## AI Functions (`/fn`)

### `/summarize`

Summarize text passed via pipe.

```bash
cat README.md | /summarize
cat README.md | /summarize --length short    # short | medium | long
cat README.md | /summarize --style bullets   # prose | bullets | outline
```

### `/extract`

Extract structured data from text.

**LLM extraction** (`--type`) — describe what you want in plain English:
```bash
cat schema.sql | /extract --type "table names"
cat config.yaml | /extract --type "database connection settings" --format json
cat Makefile | /extract --type "build targets" --format list
cat access.log | /extract --type "unique IP addresses"
```

**Regex extraction** (`--data-types`) — fast pattern matching:
```bash
cat log.txt | /extract --data-types emails,urls
cat config | /extract --data-types ip-addresses,ports
```

Supported data types: `emails`, `urls`, `ip-addresses`, `phone-numbers`, `dates`, `credit-cards`, `ssns`, `aws-keys`, `private-keys`

Output formats: `list` (default), `json`, `csv`

---

## Context (`/ctx`)

Named slots that are injected into the agent's system prompt automatically — so you can load files once and ask multiple questions.

```bash
cat ARCHITECTURE.md | /ctx add arch
cat schema.sql      | /ctx add schema

?given the arch and schema, what needs to change for multi-tenancy
?which tables in the schema relate to authentication

/ctx list               # arch (4.2KB)  schema (1.1KB)
/ctx show arch          # print slot content
/ctx clear arch         # remove one slot
/ctx clear              # remove all slots
```

Named slots (`/ctx add name`) persist to `~/.config/baish/contexts/<name>.md` and are restored on startup. The unnamed `default` slot is session-only.

If a slot exceeds `ctx_max_inject_chars` (default 8KB), the agent sees the first N chars with a truncation notice. The full content is always available via `/ctx show`.

---

## Background Jobs (`&`)

Add `&` to any input to run it in the background:

```bash
cat bigfile.pdf | /summarize &
?analyze the codebase for security issues &
sleep 30 &
```

The prompt shows running/done counts:
```
~/project [2⋯ 1✓] $     # 2 running, 1 completed (unread)
~/project [3✓] $          # 3 done, unread (shown in yellow)
```

Completion notifications are printed before the next prompt. View and manage jobs:

```
/jobs           # list all jobs; clears unread indicator
/job 3          # print full output of job 3
```

Agent queries (`?`) serialize — a background `?query &` waits for any in-progress agent turn to finish before starting, so conversation history stays consistent.

---

## Permissions

The agent's bash tool is classified into three tiers:

| Tier | Behavior |
|---|---|
| `auto` | Runs silently (`ls`, `cat`, `grep`, `git log`, `echo`, etc.) |
| `confirm` | Prompts `allow? [y/N]` before running (`rm`, `git commit`, `curl`, file writes, etc.) |
| `deny` | Never runs (`rm -rf /`, `sudo`, `dd if=`, fork bombs) |

When prompted:
```
  [bash] about to run: rm -rf ./dist
  allow? [y/N]
```

`y` runs the command; `n` returns "user declined" to the agent so it can try another approach.

Check the classification for a command:
```
/permissions rm -rf ./dist
```

---

## Other Commands

```
/tools          # list available AI functions with descriptions
/history        # print agent conversation history
/history clear  # clear agent history
/model <name>   # switch Ollama model for this session
/help           # show command reference
exit            # quit
```

---

## Troubleshooting

**`connection refused` or agent not responding**

Ollama isn't running or the endpoint doesn't match. Check:
```bash
curl http://localhost:11434/api/tags
```
If Ollama is on a different port, start baish with:
```bash
OLLAMA_ENDPOINT=http://localhost:11435 baish
```

**Agent says it has no tools / doesn't use bash**

The model may be too small. Try a larger model:
```bash
/model llama3.2
```
baish works best with models that support tool calling (function calling). If the model doesn't support native tool calling, the `SimplePlanner` falls back to `USE_TOOL:` text parsing — this works but is less reliable.

**Agent gives a blank or truncated response**

Tool output may be hitting the cap. Check `tool_output_max_chars`:
```
/config
/config set tool_output_max_chars 8000
```

**Context not appearing in agent responses**

Verify the slot exists and has content:
```
/ctx list
/ctx show myslot
```
If the slot is very large it will be truncated. Raise the cap:
```
/config set ctx_max_inject_chars 16000
```

**`/extract --type` returns nothing useful**

The model needs to be capable of structured extraction. Try a larger model, or use `--data-types` for regex-based extraction instead.

**Permissions blocking commands you want the agent to run automatically**

`confirm`-tier includes broad patterns like `curl` and `make`. When prompted, answer `y`. User-configurable permission overrides are on the roadmap.

**Debug logging**

```bash
BAISH_DEBUG=1 baish
```

---

## Planned Features

- **Sandboxing** — OverlayFS + Docker container isolation; `/diff`, `/undo`, `/checkpoint`
- **Interactive Dockerfile builder** — track environment changes (apt, pip, npm installs) and synthesize a Dockerfile on exit
- **Skills** — record a workflow and save it as a slash command; make skills callable by the agent
- **Multi-provider LLM** — Anthropic (Claude), OpenAI, NVIDIA NIMs, and other OpenAI-compatible endpoints
- **Multi-pane TUI** — optional bubbletea side panel with live agent reasoning, context browser, and task list
- **MCP server integration** — expose MCP-compatible tool servers to the baish agent
- **Context compaction** — lazy LLM summary when a context slot grows very large; agent can search the full file if needed

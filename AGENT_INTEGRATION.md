# Agent integration guide

This file is written for an AI coding agent. Paste the link into the agent and
ask it to follow the instructions, or copy the prompt below.

> Read https://raw.githubusercontent.com/MapleProjects/kaede-recall-mcp/master/AGENT_INTEGRATION.md
> and follow it to install and configure Kaede Recall MCP on this machine.
> Do not skip the verification step.

## What Kaede Recall is

A typed long-term memory server for coding agents. It exposes four MCP tools
over a vector collection:

- `memory-recall` searches memory with a hard character budget.
- `memory-write` stores one durable memory, dedupes by hash and links entities.
- `memory-update` edits a memory in place.
- `memory-delete` removes a memory by id or by semantic match.

There is no second, untyped document API. Every write goes through the same
store.

## Requirements

- Python 3.10 or newer.
- A reachable vector database, for example `http://localhost:6333`.
- Git.

## 1. Install

```shell
git clone https://github.com/MapleProjects/kaede-recall-mcp
cd kaede-recall-mcp
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

The binary is `.venv/bin/kaede-recall-mcp`. The package is not published on
PyPI, so install it from the repository.

## 2. Run the embedding daemon (recommended)

The daemon loads the embedding model once and serves vectors over loopback.
MCP clients then stay small instead of loading their own copy of the model.

```shell
.venv/bin/python -m kaede_recall_mcp.memory_os.service
```

It listens on `127.0.0.1:7451` and exposes `GET /health`, `POST /recall`,
`POST /embed` and `POST /invalidate`.

A user systemd unit:

```ini
[Unit]
Description=Kaede Recall embedding and recall daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/kaede-recall-mcp
Environment=KAEDE_BACKEND_URL=http://localhost:6333
Environment=MEMORY_OS_COLLECTION=kaede-memory
Environment=EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-es
ExecStart=/path/to/kaede-recall-mcp/.venv/bin/python -m kaede_recall_mcp.memory_os.service
Restart=always

[Install]
WantedBy=default.target
```

## 3. Configure the MCP client

Use the same server key everywhere: `kaede-recall`.

### Claude Code, Claude Desktop, Gemini, Cursor and other JSON clients

```json
{
  "mcpServers": {
    "kaede-recall": {
      "command": "/path/to/kaede-recall-mcp/.venv/bin/kaede-recall-mcp",
      "env": {
        "KAEDE_BACKEND_URL": "http://localhost:6333",
        "COLLECTION_NAME": "kaede-memory",
        "EMBEDDING_MODEL": "jinaai/jina-embeddings-v2-base-es"
      }
    }
  }
}
```

### OpenCode and MiMo Code

These clients use `environment` instead of `env`, and the command as an array:

```json
{
  "mcpServers": {
    "kaede-recall": {
      "type": "local",
      "command": ["/path/to/kaede-recall-mcp/.venv/bin/kaede-recall-mcp"],
      "environment": {
        "KAEDE_BACKEND_URL": "http://localhost:6333",
        "COLLECTION_NAME": "kaede-memory",
        "EMBEDDING_MODEL": "jinaai/jina-embeddings-v2-base-es"
      }
    }
  }
}
```

### Hermes Agent

```shell
hermes config set mcp_servers.kaede-recall.command /path/to/kaede-recall-mcp/.venv/bin/kaede-recall-mcp
hermes config set mcp_servers.kaede-recall.env.KAEDE_BACKEND_URL http://localhost:6333
hermes config set mcp_servers.kaede-recall.env.COLLECTION_NAME kaede-memory
hermes config set mcp_servers.kaede-recall.env.EMBEDDING_MODEL jinaai/jina-embeddings-v2-base-es
```

## 4. Recommended system prompt

Append this block to the agent's system prompt file. The file name depends on
the client: `SOUL.md` for Hermes, `CLAUDE.md` for Claude Code, `GEMINI.md` for
Gemini, `AGENTS.md` for OpenCode and MiMo Code.

```markdown
## Hard Gate 1: Codebase Memory MCP (MANDATORY FOR CODE)

For ANY task involving code (explain, debug, review, modify, locate usages,
analyze architecture), DO NOT call `read_file`, `search_files` or shell grep
before running the graph chain.

- Exceptions only: config files, README/docs, new files from scratch, logs,
  single-file scripts under 50 lines.
- Chain: `get_architecture(project)` then `search_graph` then `trace_path` or
  `query_graph`, then `get_code_snippet` or `search_code`.
- The `project` argument is the normalized id returned by `get_architecture`,
  never the filesystem path. If the project is not indexed, run
  `index_repository(repo_path)` first.
- If the MCP tools are deferred, call `tool_search("codebase-memory")` and
  `tool_describe` first. Deferred is not absent.
- LOCALIZAR USOS, before any edit, rename or delete: map every caller with
  `trace_path` or `query_graph` and cross-check file plus line. Never locate
  usages by guessing or by grep first.
- Invocation errors are not missing symbols. `trace_path` needs the exact
  `qualified_name` (`Class.method`). `query_graph` matches qualified names,
  use `m.name CONTAINS '...'`. `search_code` is literal unless `regex=true`.
- Fall back to file tools only after a correct graph call confirms the symbol
  has no indexed nodes.

## Hard Gate 2: Persistent Memory (Kaede Recall)

Kaede Recall is the only long-term memory. Do not store durable facts in the
agent's built-in memory file.

- The pre-turn hook, when available, already injects relevant nodes. Search
  again only when the context is missing or a full body is needed.
- `memory-recall(query | ids, walk_depth)` searches on demand or opens compact
  hits.
- `memory-write(kind, text, summary, scope, project, tags, entities, links)`
  stores immediately. It dedupes by hash and links entities.
- `memory-update(point_id, ...)` and `memory-delete(point_id | query, dry_run)`
  fix or remove entries. Confirm with the user before overwriting a
  contradiction.
- Personal data: check memory first. If it is missing, ask the user directly.
  Never web search or derive it. Save it immediately after.
- Never store temporary state, file dumps, session logs or daily-changing data.

## Operational Rules

- No side effects without explicit permission. Do not send emails, publish,
  deploy or call external APIs unless the user asked for it.
- Do not edit code unless the user asked for a change. Explaining, reviewing
  and auditing are read-only.
- Verify service status, paths and ports before assuming. Batch independent
  tool calls. After two failures, change approach.
```

## 5. Verify

```shell
hermes mcp test kaede-recall
```

Expected output: four tools, `memory-recall`, `memory-write`, `memory-update`
and `memory-delete`.

For a manual check:

```shell
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"1"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | /path/to/kaede-recall-mcp/.venv/bin/kaede-recall-mcp
```

## License

Apache-2.0. This project is derived from
[vector/mcp-server-vector](https://github.com/vector/mcp-server-vector); the
attribution is in `NOTICE`.

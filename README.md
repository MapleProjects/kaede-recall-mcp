# Kaede Recall MCP

> The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is an open protocol that enables
> seamless integration between LLM applications and external data sources and tools. Whether you're building an
> AI-powered IDE, enhancing a chat interface, or creating custom AI workflows, MCP provides a standardized way to
> connect LLMs with the context they need.

A Model Context Protocol server that turns a vector database into a typed long-term memory for coding agents.

## Overview

One collection, one write path, four tools. Every memory is a typed node, not an
opaque document, and every mutation goes through the same store.

### Architecture

- **Typed nodes.** Each point carries `kind` (fact, preference, profile, decision,
  problem, procedure, trace, doc), `text`, `summary`, `scope`, optional `project`,
  `status`, `hash`, `entities`, `edges` and `backlinks`.
- **Four tools.** `memory-recall`, `memory-write`, `memory-update` and
  `memory-delete`. There is no second, untyped document API.
- **Deterministic ids.** The point id is a uuid5 of the content hash, scope and
  project, so a retry or a replay lands on the same key.
- **Write path.** Dedupe by `hash`, auto-supersede a near-identical fact at 0.92
  only when scope, project and the entity set match, and link relations in the
  same call.
- **Read path.** One embedding per query. A global vector search, an entity slot,
  and an exact lane for paths, known file names and hex digests (real similarity
  plus a bounded +0.08). Slot 1 needs a raw score of 0.33; slot 2 can displace it
  at a 0.12 margin. An optional graph hop follows payload edges and drops
  superseded or deprecated nodes.
- **Embeddings.** A local daemon exposes `POST /embed` and `POST /invalidate` on
  `127.0.0.1:7451`. MCP clients ask it for vectors instead of loading their own
  model, and the daemon owns the entity registry and the recall cache.
- **Maintenance.** `bootstrap dedupe` and `bootstrap compact` consolidate
  historical duplicates offline from the stored vectors; a weekly timer takes a
  snapshot and compacts.

> Integrating with an agent? See [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md)
> for a copy-paste guide, the MCP config for each client and the recommended
> system prompt.

## Components

### Tools

Four tools, one collection, one write path. Every write goes through the same
typed memory store, so there is no second, untyped copy of the same memory.

1. `memory-write`
   - Store one durable memory. Dedupes by content hash, links relations and
     supersedes an older fact in the same call.
   - Input: `kind`, `text`, `summary`, `scope`, `project`, `tags`, `entities`,
     `links`, `supersedes_id`, `source`.
   - Returns: `Stored <kind>, id=<uuid>`, or `Already stored, id=<uuid>`.
2. `memory-recall`
   - Search long-term memory with a hard character budget. One dense search,
     an entity slot, exact-path/hash matching and an optional graph hop.
   - Input: `query`, `kinds`, `scope`, `project`, `limit`, `max_chars`,
     `walk_depth`, `include_superseded`, `ids`.
   - Returns: numbered hits with `[kind score]`, a trigger line, the `id`, and
     the body of the first one or two slots.
3. `memory-update`
   - Edit a memory in place. `text` recomputes the hash, re-extracts the
     entities and re-embeds `summary + first 1200 chars`. Tags and entities
     alone rewrite the payload without a model call.
   - Input: `point_id`, `text`, `summary`, `tags`, `entities`.
   - This tool is disabled when `KAEDE_READ_ONLY` is enabled.
4. `memory-delete`
   - Delete by exact `point_id` or by semantic `query`, with `dry_run` to
     preview the matches.
   - Input: `point_id`, `query`, `limit`, `score_threshold`, `dry_run`.
   - This tool is disabled when `KAEDE_READ_ONLY` is enabled.

To audit historical duplicates, `python -m kaede_recall_mcp.memory_os.bootstrap
dedupe --dry-run` lists active pairs that share an entity with similarity
`>= 0.90`; without `--dry-run` the older point of each pair is superseded by
the newer one.

### Payload fields written by the server

| Field          | Meaning                                                              |
|----------------|----------------------------------------------------------------------|
| `kind`         | `fact`, `preference`, `profile`, `decision`, `problem`, `procedure`, `trace`, `doc`. |
| `text`         | The memory body.                                                     |
| `summary`      | One-line trigger, also the head of the embedding input.              |
| `scope`        | `global` or `project:<id>`.                                          |
| `project`      | Optional project id, stored only when set.                           |
| `status`       | `active` or `superseded`.                                            |
| `hash`         | SHA-256 of the normalized `kind + text`. Dedupe key and id source.   |
| `entities`     | Structural aliases used by the entity slot.                          |
| `edges`        | Outgoing relations `{rel, to}`.                                      |
| `backlinks`    | Incoming relations `{rel, from}`.                                    |
| `created_at`   | ISO 8601 UTC timestamp of the first write.                           |
| `updated_at`   | ISO 8601 UTC timestamp of the last write.                            |

## Environment Variables

Configuration is done via environment variables. The only command-line argument is `--transport`, used to select the [transport protocol](#transport-protocols).

> [!NOTE]
> You cannot provide both `KAEDE_BACKEND_URL` and `KAEDE_BACKEND_PATH` at the same time.

| Name                     | Description                                                         | Default Value                                                     |
|--------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|
| `KAEDE_BACKEND_URL`             | URL of the vector backend                                            | None                                                              |
| `KAEDE_BACKEND_API_KEY`         | API key for the vector backend                                       | None                                                              |
| `COLLECTION_NAME`        | Name of the default collection to use.                              | None                                                              |
| `KAEDE_BACKEND_PATH`      | Path to the local vector database (alternative to `KAEDE_BACKEND_URL`)     | None                                                              |
| `EMBEDDING_PROVIDER`     | Embedding provider to use (currently only "fastembed" is supported) | `fastembed`                                                       |
| `EMBEDDING_MODEL`        | Name of the embedding model to use                                  | `sentence-transformers/all-MiniLM-L6-v2`                          |
| `MEMORY_OS_EMBED_URL`    | Memory OS daemon used for embeddings and invalidation               | `http://127.0.0.1:7451`                                           |
| `MEMORY_OS_EMBED_REMOTE` | Ask the daemon before loading a local model (`false` forces local)  | `true`                                                            |
| `KAEDE_SEARCH_LIMIT`    | Maximum number of results to return from search                     | `10`                                                              |
| `KAEDE_READ_ONLY`       | Enable read-only mode (disables mutation tools)                      | `false`                                                           |
| `KAEDE_DEDUPE`          | Skip a write when the same content and metadata already exist        | `true`                                                            |
| `KAEDE_RETRIES`         | Attempts for a call that fails on a transient connection error       | `3`                                                               |

### FastMCP Environment Variables

Since `kaede-recall-mcp` is based on FastMCP, it also supports all the FastMCP environment variables. The most
important ones are listed below:

| Environment Variable                       | Description                                                     | Default Value |
|--------------------------------------------|-----------------------------------------------------------------|---------------|
| `FASTMCP_LOG_LEVEL`                        | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)       | `INFO`        |
| `FASTMCP_SERVER_DEBUG`                     | Enable debug mode                                               | `false`       |
| `FASTMCP_SERVER_HOST`                      | Host address to bind the server to                              | `127.0.0.1`   |
| `FASTMCP_SERVER_PORT`                      | Port to run the server on                                       | `8000`        |
| `FASTMCP_SERVER_ON_DUPLICATE_RESOURCES`    | Behavior for duplicate resources (warn, error, replace, ignore) | `warn`        |
| `FASTMCP_SERVER_ON_DUPLICATE_TOOLS`        | Behavior for duplicate tools (warn, error, replace, ignore)     | `warn`        |
| `FASTMCP_SERVER_ON_DUPLICATE_PROMPTS`      | Behavior for duplicate prompts (warn, error, replace, ignore)   | `warn`        |
| `FASTMCP_SERVER_DEPENDENCIES`              | List of dependencies to install in the server environment       | `[]`          |

> [!NOTE]
> Server-specific settings use the `FASTMCP_SERVER_` prefix. This may change in future versions.

## Installation

### From source

The package is distributed from this repository, not from PyPI.

```shell
git clone https://github.com/MapleProjects/kaede-recall-mcp
cd kaede-recall-mcp
uv pip install -e .
```

```shell
KAEDE_BACKEND_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
kaede-recall-mcp
```

#### Transport Protocols

The server supports different transport protocols that can be specified using the `--transport` flag:

```shell
KAEDE_BACKEND_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
kaede-recall-mcp --transport sse
```

Supported transport protocols:

- `stdio` (default): Standard input/output transport, might only be used by local MCP clients
- `sse`: Server-Sent Events transport, perfect for remote clients
- `streamable-http`: Streamable HTTP transport, perfect for remote clients, more recent than SSE

The default transport is `stdio` if not specified.

When SSE transport is used, the server will listen on the specified port and wait for incoming connections. The default
port is 8000, however it can be changed using the `FASTMCP_SERVER_PORT` environment variable.

```shell
KAEDE_BACKEND_URL="http://localhost:6333" \
COLLECTION_NAME="my-collection" \
FASTMCP_SERVER_PORT=1234 \
kaede-recall-mcp --transport sse
```

### Using Podman (recommended)

Podman runs rootless and needs no daemon. The same `Dockerfile` works.

```bash
# Build the image
podman build -t kaede-recall-mcp .

# Run the container
podman run --rm -p 8000:8000 \
  -e FASTMCP_SERVER_HOST="0.0.0.0" \
  -e KAEDE_BACKEND_URL="http://your-vector-server:6333" \
  -e KAEDE_BACKEND_API_KEY="your-api-key" \
  -e COLLECTION_NAME="kaede-memory" \
  kaede-recall-mcp
```

> [!TIP]
> `FASTMCP_SERVER_HOST="0.0.0.0"` makes the server listen on all network
> interfaces, which is required inside a container.

Docker works too. Replace `podman` with `docker` in the commands above.

### Manual configuration of Claude Desktop

To use this server with the Claude Desktop app, add the following configuration to the "mcpServers" section of your
`claude_desktop_config.json`:

```json
{
  "kaede-recall": {
    "command": "/path/to/kaede-recall-mcp/.venv/bin/kaede-recall-mcp",
    "env": {
      "KAEDE_BACKEND_URL": "https://localhost:6333",
      "KAEDE_BACKEND_API_KEY": "your_api_key",
      "COLLECTION_NAME": "kaede-memory",
      "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

For local backend mode:

```json
{
  "kaede-recall": {
    "command": "/path/to/kaede-recall-mcp/.venv/bin/kaede-recall-mcp",
    "env": {
      "KAEDE_BACKEND_PATH": "/path/to/backend/database",
      "COLLECTION_NAME": "kaede-memory",
      "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

This MCP server will automatically create a collection with the specified name if it doesn't exist.

By default, the server will use the `sentence-transformers/all-MiniLM-L6-v2` embedding model to encode memories.
For the time being, only [FastEmbed](https://vector.github.io/fastembed/) models are supported.

## Support for other tools

This MCP server can be used with any MCP-compatible client. For example, you can use it with
[Cursor](https://docs.cursor.com/context/model-context-protocol) and [VS Code](https://code.visualstudio.com/docs), which provide built-in support for the Model Context
Protocol.

### Using with Cursor/Windsurf

You can configure this MCP server to work as a code search tool for Cursor or Windsurf by customizing the tool
descriptions:

```bash
KAEDE_BACKEND_URL="http://localhost:6333" \
COLLECTION_NAME="code-snippets" \
TOOL_STORE_DESCRIPTION="Store reusable code snippets for later retrieval. \
The 'information' parameter should contain a natural language description of what the code does, \
while the actual code should be included in the 'metadata' parameter as a 'code' property. \
The value of 'metadata' is a Python dictionary with strings as keys. \
Use this whenever you generate some code snippet." \
TOOL_FIND_DESCRIPTION="Search for relevant code snippets based on natural language descriptions. \
The 'query' parameter should describe what you're looking for, \
and the tool will return the most relevant code snippets. \
Use this when you need to find existing code snippets for reuse or reference." \
kaede-recall-mcp --transport sse # Enable SSE transport
```

In Cursor/Windsurf, you can then configure the MCP server in your settings by pointing to this running server using
SSE transport protocol. The description on how to add an MCP server to Cursor can be found in the [Cursor
documentation](https://docs.cursor.com/context/model-context-protocol#adding-an-mcp-server-to-cursor). If you are
running Cursor/Windsurf locally, you can use the following URL:

```
http://localhost:8000/sse
```

> [!TIP]
> We suggest SSE transport as a preferred way to connect Cursor/Windsurf to the MCP server, as it can support remote
> connections. That makes it easy to share the server with your team or use it in a cloud environment.

This configuration transforms the MCP server into a specialized code search tool that can:

1. Store code snippets, documentation, and implementation details
2. Retrieve relevant code examples based on semantic search
3. Help developers find specific implementations or usage patterns

You can populate the database by storing natural language descriptions of code snippets (in the `information` parameter)
along with the actual code (in the `metadata.code` property), and then search for them using natural language queries
that describe what you're looking for.

> [!NOTE]
> The tool descriptions provided above are examples and may need to be customized for your specific use case. Consider
> adjusting the descriptions to better match your team's workflow and the specific types of code snippets you want to
> store and retrieve.

**If you have successfully installed the `kaede-recall-mcp`, but still can't get it to work with Cursor, please
consider creating the [Cursor rules](https://docs.cursor.com/context/rules-for-ai) so the MCP tools are always used when
the agent produces a new code snippet.** You can restrict the rules to only work for certain file types, to avoid using
the MCP server for the documentation or other types of content.

### Using with Claude Code

You can enhance Claude Code's capabilities by connecting it to this MCP server, enabling semantic search over your
existing codebase.

#### Setting up kaede-recall-mcp

1. Add the MCP server to Claude Code:

    ```shell
    # Add kaede-recall for code search
    claude mcp add kaede-recall \
    -e KAEDE_BACKEND_URL="http://localhost:6333" \
    -e COLLECTION_NAME="kaede-memory" \
    -e EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
    -- /path/to/kaede-recall-mcp/.venv/bin/kaede-recall-mcp
    ```

2. Verify the server was added:

    ```shell
    claude mcp list
    ```

#### Using Semantic Code Search in Claude Code

Tool descriptions, specified in `TOOL_STORE_DESCRIPTION` and `TOOL_FIND_DESCRIPTION`, guide Claude Code on how to use
the MCP server. The ones provided above are examples and may need to be customized for your specific use case. However,
Claude Code should be already able to:

1. Use the `memory-write` tool to store code snippets with descriptions.
2. Use the `memory-recall` tool to search for relevant code snippets using natural language.

### Run MCP server in Development Mode

The MCP server can be run in development mode using the `mcp dev` command. This will start the server and open the MCP
inspector in your browser.

```shell
COLLECTION_NAME=mcp-dev fastmcp dev src/kaede_recall_mcp/server.py
```

### Using with VS Code

Add this to your User Settings (JSON) or to `.vscode/mcp.json`:

```json
{
  "servers": {
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

## Contributing

If you have suggestions for how kaede-recall-mcp could be improved, or want to report a bug, open an issue!
We'd love all and any contributions.

### Testing `kaede-recall-mcp` locally

The [MCP inspector](https://github.com/modelcontextprotocol/inspector) is a developer tool for testing and debugging MCP
servers. It runs both a client UI (default port 5173) and an MCP proxy server (default port 3000). Open the client UI in
your browser to use the inspector.

```shell
KAEDE_BACKEND_URL=":memory:" COLLECTION_NAME="test" \
fastmcp dev src/kaede_recall_mcp/server.py
```

Once started, open your browser to http://localhost:5173 to access the inspector interface.

## License

This MCP server is licensed under the Apache License 2.0. This means you are free to use, modify, and distribute the
software, subject to the terms and conditions of the Apache License 2.0. For more details, please see the LICENSE file
in the project repository.

## Privacy and support

- [Privacy Policy](https://example.com/vector-databaselegal/privacy-policy/)
- [Support](https://github.com/vector/kaede-recall-mcp/issues)

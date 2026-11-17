FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Build from the repository itself: the package is not published on PyPI.
COPY pyproject.toml README.md NOTICE LICENSE ./
COPY src ./src
RUN uv pip install --system --no-cache-dir .

# Expose the default port for SSE transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV KAEDE_BACKEND_URL=""
ENV KAEDE_BACKEND_API_KEY=""
ENV COLLECTION_NAME="kaede-memory"
ENV EMBEDDING_MODEL="jinaai/jina-embeddings-v2-base-es"

# Run the server with SSE transport
CMD ["kaede-recall-mcp", "--transport", "sse"]

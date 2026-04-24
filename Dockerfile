FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    rm -rf /var/lib/apt/lists/*

# Install google-adk from the local source
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir -e .

# Create default agents directory
RUN mkdir -p /app/agents

EXPOSE 8000

ENTRYPOINT ["adk", "web", "--host", "0.0.0.0", "--port", "8000", "/app/agents"]

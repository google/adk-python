FROM python:3.11-slim

WORKDIR /app

# Install the local ADK wheel (built from the repo under test).
# Install the .whl file directly so pip uses it instead of PyPI.
# Dependencies are resolved from PyPI automatically.
COPY wheels/ /tmp/wheels/
SHELL ["/bin/bash", "-c"]
RUN pip install --no-cache-dir /tmp/wheels/google_adk-*.whl \
    && rm -rf /tmp/wheels

# Copy the agent into the expected adk layout:
#   /app/agents/trigger_echo_agent/__init__.py
#   /app/agents/trigger_echo_agent/agent.py
COPY __init__.py agents/trigger_echo_agent/__init__.py
COPY agent.py    agents/trigger_echo_agent/agent.py

ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

CMD ["adk", "api_server", "--host=0.0.0.0", "--port=8080", "--trigger_sources=pubsub,eventarc", "/app/agents"]

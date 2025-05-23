# config/dockerfile_template.py

_DOCKERFILE_TEMPLATE = """
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Change ownership of /app to myuser
RUN chown -R myuser:myuser /app

# Switch to the non-root user
USER myuser

# Set up environment variables
ENV PATH="/home/myuser/.local/bin:$PATH"

# Install ADK
RUN pip install google-adk=={adk_version}

# Copy agent
COPY "agents/{app_name}/" "/app/agents/{app_name}/"
{install_agent_deps}

EXPOSE {port}

CMD adk {command} --port={port} {host_option} {session_db_option} {trace_to_cloud_option} "/app/agents"
"""

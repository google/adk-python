# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

_DOCKERFILE_TEMPLATE = """
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Switch to the non-root user
USER myuser

# Set up environment variables - Start
ENV PATH="/home/myuser/.local/bin:$PATH"
{extra_env_vars}
# Set up environment variables - End

# Install ADK - Start
RUN pip install "google-adk[a2a]=={adk_version}"
# Remove dev_server.py to ensure production-safe endpoints only (disabling dev endpoints in production)
RUN python -c "import os, glob, google.adk.cli as cli; d = os.path.dirname(cli.__file__); [os.remove(f) for f in glob.glob(os.path.join(d, 'dev_server*'))]; [os.remove(f) for f in glob.glob(os.path.join(d, '__pycache__', 'dev_server*'))]" || true
# Install ADK - End

# Copy agent - Start

# Set permission
COPY --chown=myuser:myuser "agents/{app_name}/" "/app/agents/{app_name}/"
{extra_packages_copy}
# Copy agent - End

# Install Agent Deps - Start
{install_agent_deps}
# Install Agent Deps - End

EXPOSE {port}

CMD adk {command} --port={port} {host_option} {service_option} {trace_to_cloud_option} {otel_to_cloud_option} {allow_origins_option} {a2a_option} {trigger_sources_option} {trigger_oidc_audience_option} {trigger_oidc_service_accounts_option} {gemini_enterprise_option}{express_mode_option} "/app/agents"
"""

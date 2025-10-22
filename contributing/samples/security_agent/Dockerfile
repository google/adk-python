# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install runtime prerequisites (bash for scripts, lsof for port checks, curl for healthchecks)
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash lsof curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application source
COPY . .

# Ensure scripts are executable
RUN chmod +x scripts/start_all.sh

# Expose agent, Flask UI, and Chainlit ports
EXPOSE 8000 5001 8001

# Default command launches all services
CMD ["/bin/bash", "./scripts/start_all.sh"]


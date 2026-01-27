# Agent Analytics Dashboard Docker Image
#
# Build:
#   docker build -t agent-analytics-dashboard .
#
# Run:
#   docker run -p 8080:8080 \
#     -e BQ_AGENT_ANALYTICS_PROJECT=your-project \
#     -e BQ_AGENT_ANALYTICS_DATASET=your-dataset \
#     -v ~/.config/gcloud:/root/.config/gcloud \
#     agent-analytics-dashboard

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY __init__.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

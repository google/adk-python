# Multi-stage build for ADK Security Agent
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x run_backend.py run_frontend.py

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV USE_LEGACY=true
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=info

# Create necessary directories
RUN mkdir -p logs config/secrets

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "run_backend.py"]
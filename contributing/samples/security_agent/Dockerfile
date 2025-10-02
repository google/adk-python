# Lightweight Python image optimized for Cloud Run
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install minimal dependencies for Google Cloud libraries
RUN apk add --no-cache gcc musl-dev linux-headers

# Copy and install minimal requirements
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt && \
    pip install --no-cache-dir gunicorn==21.2.0

# Copy only necessary files
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/

# Create non-root user
RUN adduser -D -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run with minimal workers
CMD exec gunicorn --bind :${PORT} --workers 1 --threads 2 --timeout 60 app:app
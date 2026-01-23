FROM python:3.12-slim

WORKDIR /app

# Install uv for faster package management (rarely changes)
RUN pip install uv

# Copy only dependency specification first for better caching
COPY pyproject.toml README.md ./

# Create minimal package structure for dependency resolution
RUN mkdir -p adk_rlm && echo '__version__ = "0.1.0"' > adk_rlm/__init__.py

# Install dependencies (this layer is cached unless pyproject.toml changes)
RUN uv pip install --system -e ".[all]" --index-url https://pypi.org/simple/

# Now copy the actual source code (changes frequently)
COPY adk_rlm/ adk_rlm/

# Create logs directory
RUN mkdir -p /app/logs

# Expose port (Cloud Run uses PORT env var)
ENV PORT=8080
EXPOSE 8080

# Run the web server
CMD ["sh", "-c", "python -m adk_rlm.web --host 0.0.0.0 --port $PORT"]

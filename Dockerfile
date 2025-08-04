# Use a Python base image for building
FROM python:3.10-slim-buster as builder

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements files first to leverage Docker cache
COPY contributing/samples/security_agent/requirements.txt ./requirements.txt
COPY contributing/samples/security_agent/backend/requirements.txt ./backend_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r backend_requirements.txt

# --- Final Stage ---
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /usr/local/bin/streamlit /usr/local/bin/streamlit
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# Copy the entire application code
COPY . .

# Ensure the run.sh script is executable
RUN chmod +x ./run.sh

# Expose the ports for the backend (FastAPI) and frontend (Streamlit)
EXPOSE 8000
EXPOSE 8501

# Command to run the application using the run.sh script
# The run.sh script handles starting both backend and frontend
CMD ["python", "./run.py"]

"""Centralized configuration for the frontend application.

This module provides a single source of truth for all configuration values,
ensuring consistency across the application and proper use of environment variables.
"""

import os
from typing import Optional
from pathlib import Path

# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    
    # Try to find .env file in current directory or parent directories
    current_dir = Path(__file__).parent
    for _ in range(5):  # Look up to 5 levels up
        env_file = current_dir / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            break
        current_dir = current_dir.parent
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# =====================================
# Backend Configuration
# =====================================

# Backend server URL (from environment or default)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# API version path
API_V1_BASE_PATH = "/api/v1"

# Full API base URL
API_BASE_URL = f"{BACKEND_URL}{API_V1_BASE_PATH}"

# =====================================
# Google Cloud Configuration
# =====================================

# GCP Project ID (from environment)
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", None)

# Default project ID for development (should be None in production)
DEFAULT_PROJECT_ID = GOOGLE_CLOUD_PROJECT

# Service account key file path
GOOGLE_SERVICE_ACCOUNT_KEY_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY_FILE", None)

# Standard Google Cloud credentials environment variable
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)

# =====================================
# Application Configuration
# =====================================

# Database configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", None)

# Default user email for analysis
DEFAULT_USER_EMAIL = os.getenv("DEFAULT_USER_EMAIL", None)

# Server ports
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
ADK_WEB_PORT = int(os.getenv("ADK_WEB_PORT", "8080"))

# =====================================
# ADK Configuration
# =====================================

# Enable ADK evaluation features
ADK_EVALUATION_ENABLED = os.getenv("ADK_EVALUATION_ENABLED", "false").lower() == "true"

# =====================================
# Vertex AI Configuration
# =====================================

# Vertex AI project (usually same as GOOGLE_CLOUD_PROJECT)
VERTEX_AI_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID", GOOGLE_CLOUD_PROJECT)

# Vertex AI location/region
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")

# =====================================
# Cloud Build Configuration
# =====================================

# Region for deployments
DEPLOYMENT_REGION = os.getenv("_REGION", "us-central1")

# Repository and service names
REPO_NAME = os.getenv("_REPO_NAME", "adk-security-agent")
IMAGE_NAME = os.getenv("_IMAGE_NAME", "security-agent")
SERVICE_NAME = os.getenv("_SERVICE_NAME", "security-agent")

# =====================================
# Validation Functions
# =====================================

def validate_config() -> dict:
    """Validate the configuration and return any warnings or errors.
    
    Returns:
        dict: A dictionary containing 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []
    
    # Check required configurations
    if not GOOGLE_CLOUD_PROJECT:
        warnings.append("GOOGLE_CLOUD_PROJECT is not set. Some features may not work properly.")
    
    if not GOOGLE_APPLICATION_CREDENTIALS and not GOOGLE_SERVICE_ACCOUNT_KEY_FILE:
        warnings.append("No Google Cloud credentials configured. Authentication may fail.")
    
    # Check backend connectivity
    if BACKEND_URL == "http://localhost:8000":
        warnings.append("Using default localhost backend URL. Ensure backend is running locally.")
    
    return {
        "errors": errors,
        "warnings": warnings
    }

def get_project_id() -> Optional[str]:
    """Get the configured GCP project ID.
    
    Returns:
        str or None: The project ID if configured, None otherwise
    """
    return GOOGLE_CLOUD_PROJECT

def get_api_url(endpoint: str) -> str:
    """Build a full API URL for the given endpoint.
    
    Args:
        endpoint: The API endpoint path (e.g., "/security/score")
        
    Returns:
        str: The full API URL
    """
    # Remove leading slash if present to avoid double slashes
    endpoint = endpoint.lstrip("/")
    return f"{API_BASE_URL}/{endpoint}"

def is_production() -> bool:
    """Check if the application is running in production mode.
    
    Returns:
        bool: True if in production, False otherwise
    """
    # Consider it production if not using localhost
    return not BACKEND_URL.startswith("http://localhost")
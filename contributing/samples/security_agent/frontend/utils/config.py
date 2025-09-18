"""
Frontend Configuration Management
=================================

Centralized configuration management for the Streamlit frontend.
Loads settings from environment variables and provides default values.
"""

import os
from typing import Dict, Any

class FrontendConfig:
    """Manages frontend configuration."""

    @staticmethod
    def get_backend_url() -> str:
        """
        Get the backend API URL from environment variables.

        Returns:
            Backend API URL
        """
        return os.environ.get("BACKEND_API_URL", "http://localhost:8000")

    @staticmethod
    def get_project_id() -> str:
        """
        Get the GCP project ID from environment variables.

        Returns:
            GCP project ID
        """
        return os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")

    @staticmethod
    def get_frontend_agent_config() -> Dict[str, Any]:
        """
        Get frontend agent configuration.

        Returns:
            Dictionary with agent configuration settings
        """
        return {
            # Frontend Router Agent settings
            'router_enabled': os.environ.get("FRONTEND_ROUTER_ENABLED", "true").lower() == "true",
            'gemini_api_key': os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
            'router_model': os.environ.get("FRONTEND_ROUTER_MODEL", "gemini-1.5-flash"),

            # Local Lookup Agent settings
            'local_cache_enabled': os.environ.get("FRONTEND_LOCAL_CACHE_ENABLED", "true").lower() == "true",
            'local_response_timeout': int(os.environ.get("LOCAL_RESPONSE_TIMEOUT", "5")),

            # Query Enhancement settings
            'enhancement_enabled': os.environ.get("QUERY_ENHANCEMENT_ENABLED", "true").lower() == "true",
            'max_context_messages': int(os.environ.get("MAX_CONTEXT_MESSAGES", "4")),
            'enhancement_timeout': int(os.environ.get("ENHANCEMENT_TIMEOUT", "10")),

            # Fallback and error handling
            'fallback_to_backend': os.environ.get("FALLBACK_TO_BACKEND", "true").lower() == "true",
            'retry_attempts': int(os.environ.get("AGENT_RETRY_ATTEMPTS", "2")),

            # Logging and debugging
            'debug_mode': os.environ.get("FRONTEND_AGENT_DEBUG", "false").lower() == "true",
            'log_queries': os.environ.get("LOG_FRONTEND_QUERIES", "false").lower() == "true",
            'log_enhancements': os.environ.get("LOG_QUERY_ENHANCEMENTS", "false").lower() == "true",
        }

    @staticmethod
    def is_frontend_agent_enabled() -> bool:
        """
        Check if frontend agents are enabled.

        Returns:
            True if frontend agents should be used
        """
        config = FrontendConfig.get_frontend_agent_config()
        return (config['router_enabled'] or config['local_cache_enabled']) and config.get('gemini_api_key')

    @staticmethod
    def get_agent_api_key() -> str:
        """
        Get the API key for frontend agents.

        Returns:
            API key for Gemini/Google AI
        """
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

    @staticmethod
    def should_log_agent_activity() -> bool:
        """
        Check if agent activity should be logged.

        Returns:
            True if logging is enabled
        """
        config = FrontendConfig.get_frontend_agent_config()
        return config.get('debug_mode', False) or config.get('log_queries', False)

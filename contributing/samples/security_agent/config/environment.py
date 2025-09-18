import os
from dotenv import load_dotenv

class EnvironmentConfig:
    """Manages environment variables for the application."""

    REQUIRED_VARS = [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS"
    ]

    @staticmethod
    def load_environment():
        """Load environment variables from .env file."""
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            return True
        return False

    @staticmethod
    def get_configuration_summary():
        """Get a summary of the current environment configuration."""
        EnvironmentConfig.load_environment()
        
        summary = {
            "is_valid": True,
            "valid_count": 0,
            "missing_vars": [],
            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT", "Not Set")
        }

        for var in EnvironmentConfig.REQUIRED_VARS:
            if os.getenv(var):
                summary["valid_count"] += 1
            else:
                summary["is_valid"] = False
                summary["missing_vars"].append(var)
        
        return summary

"""
Centralized environment variable configuration for the security agent.
Handles loading, validation, and defaults for all environment variables.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """Centralized environment variable management."""
    
    # Environment variable definitions with defaults and validation
    ENV_VARS = {
        # Core GCP Configuration
        'GOOGLE_CLOUD_PROJECT': {
            'required': True,
            'default': None,
            'description': 'Google Cloud Project ID',
            'validator': lambda x: x and x != 'your-project-id'
        },
        'GOOGLE_APPLICATION_CREDENTIALS': {
            'required': True,
            'default': None,
            'description': 'Path to service account JSON file',
            'validator': lambda x: x and Path(x).exists() if x else False
        },
        'GOOGLE_CLOUD_LOCATION': {
            'required': False,
            'default': 'us-central1',
            'description': 'Google Cloud region/location',
            'validator': None
        },
        'GOOGLE_GENAI_USE_VERTEXAI': {
            'required': False,
            'default': 'TRUE',
            'description': 'Use Vertex AI for GenAI',
            'validator': lambda x: x.upper() in ['TRUE', 'FALSE'] if x else False
        },
        
        # Database Configuration
        'DATABASE_PATH': {
            'required': False,
            'default': None,  # Will be set by DatabaseConfig
            'description': 'Path to SQLite database file',
            'validator': None
        },
        'DATA_REFRESH_INTERVAL': {
            'required': False,
            'default': '1800',  # 30 minutes
            'description': 'Data refresh interval in seconds',
            'validator': lambda x: x.isdigit() and int(x) > 0 if x else False
        },
        
        # Application URLs and Ports
        'BACKEND_URL': {
            'required': False,
            'default': 'http://localhost:8000',
            'description': 'Backend API URL',
            'validator': None
        },
        'FRONTEND_URL': {
            'required': False,
            'default': 'http://localhost:8501',
            'description': 'Frontend URL',
            'validator': None
        },
        'BACKEND_PORT': {
            'required': False,
            'default': '8000',
            'description': 'Backend server port',
            'validator': lambda x: x.isdigit() and 1000 <= int(x) <= 65535 if x else False
        },
        'FRONTEND_PORT': {
            'required': False,
            'default': '8501',
            'description': 'Frontend server port',
            'validator': lambda x: x.isdigit() and 1000 <= int(x) <= 65535 if x else False
        },
        'BACKEND_HOST': {
            'required': False,
            'default': '0.0.0.0',
            'description': 'Backend server host',
            'validator': None
        },
        
        # Security and Rate Limiting
        'RATE_LIMIT_CHAT': {
            'required': False,
            'default': '30',
            'description': 'Chat requests per minute limit',
            'validator': lambda x: x.isdigit() and int(x) > 0 if x else False
        },
        'ENABLE_RATE_LIMITING': {
            'required': False,
            'default': 'true',
            'description': 'Enable rate limiting',
            'validator': lambda x: x.lower() in ['true', 'false'] if x else False
        },
        
        # Agent Configuration
        'AGENT_MODE': {
            'required': False,
            'default': 'sqlite',
            'description': 'Agent operation mode',
            'validator': lambda x: x in ['sqlite', 'api', 'hybrid'] if x else False
        },
    }
    
    @classmethod
    def load_environment(cls, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load and validate all environment variables.
        
        Args:
            force_reload: Force reloading of environment files
            
        Returns:
            Dict containing loaded environment variables
        """
        if force_reload or not hasattr(cls, '_loaded_env'):
            cls._load_env_files()
            cls._loaded_env = cls._validate_and_set_defaults()
        
        return cls._loaded_env.copy()
    
    @classmethod
    def _load_env_files(cls) -> None:
        """Load environment files in order of precedence."""
        # Find project root
        current_file = Path(__file__).resolve()
        project_root = None
        
        for parent in current_file.parents:
            if parent.name == "security_agent":
                project_root = parent
                break
        
        if not project_root:
            project_root = current_file.parent.parent
        
        # Environment file locations in order of precedence (first found wins)
        env_locations = [
            project_root / "deploy" / ".env",
            project_root / ".env",
            project_root / "backend" / ".env",
            project_root / "config" / ".env",
        ]
        
        env_loaded = False
        for env_path in env_locations:
            if env_path.exists():
                load_dotenv(env_path, override=False)  # Don't override existing vars
                logger.info(f"✅ Loaded environment from: {env_path}")
                env_loaded = True
                break
        
        if not env_loaded:
            logger.warning("⚠️ No .env file found, using system environment variables only")
    
    @classmethod
    def _validate_and_set_defaults(cls) -> Dict[str, Any]:
        """Validate environment variables and set defaults."""
        env_status = {
            'valid': {},
            'invalid': {},
            'missing_required': [],
            'using_defaults': []
        }
        
        for var_name, config in cls.ENV_VARS.items():
            value = os.getenv(var_name)
            
            # Handle missing values
            if not value:
                if config['required']:
                    env_status['missing_required'].append({
                        'name': var_name,
                        'description': config['description']
                    })
                    continue
                elif config['default'] is not None:
                    value = config['default']
                    os.environ[var_name] = value
                    env_status['using_defaults'].append({
                        'name': var_name,
                        'value': value,
                        'description': config['description']
                    })
            
            # Validate value if validator is provided
            if config['validator'] and value:
                try:
                    if config['validator'](value):
                        env_status['valid'][var_name] = {
                            'value': value,
                            'description': config['description']
                        }
                    else:
                        env_status['invalid'][var_name] = {
                            'value': value,
                            'description': config['description'],
                            'error': 'Validation failed'
                        }
                except Exception as e:
                    env_status['invalid'][var_name] = {
                        'value': value,
                        'description': config['description'],
                        'error': str(e)
                    }
            else:
                env_status['valid'][var_name] = {
                    'value': value,
                    'description': config['description']
                }
        
        # Log results
        cls._log_env_status(env_status)
        
        return env_status
    
    @classmethod
    def _log_env_status(cls, env_status: Dict[str, Any]) -> None:
        """Log environment variable status."""
        # Log valid variables
        if env_status['valid']:
            logger.info(f"✅ {len(env_status['valid'])} environment variables loaded successfully")
            for var_name, info in env_status['valid'].items():
                # Mask sensitive values
                display_value = cls._mask_sensitive_value(var_name, info['value'])
                logger.debug(f"  {var_name}={display_value}")
        
        # Log defaults used
        if env_status['using_defaults']:
            logger.info(f"📝 Using defaults for {len(env_status['using_defaults'])} variables:")
            for item in env_status['using_defaults']:
                logger.info(f"  {item['name']}={item['value']} ({item['description']})")
        
        # Log validation errors
        if env_status['invalid']:
            logger.warning(f"⚠️ {len(env_status['invalid'])} environment variables failed validation:")
            for var_name, info in env_status['invalid'].items():
                logger.warning(f"  {var_name}: {info['error']}")
        
        # Log missing required variables
        if env_status['missing_required']:
            logger.error(f"❌ {len(env_status['missing_required'])} required environment variables are missing:")
            for item in env_status['missing_required']:
                logger.error(f"  {item['name']}: {item['description']}")
    
    @classmethod
    def _mask_sensitive_value(cls, var_name: str, value: str) -> str:
        """Mask sensitive environment variable values for logging."""
        sensitive_vars = ['GOOGLE_APPLICATION_CREDENTIALS', 'API_KEY', 'SECRET', 'TOKEN', 'PASSWORD']
        
        if any(sensitive in var_name.upper() for sensitive in sensitive_vars):
            if len(value) > 8:
                return f"{value[:4]}...{value[-4:]}"
            else:
                return "***"
        
        return value
    
    @classmethod
    def get_required_variables(cls) -> List[str]:
        """Get list of required environment variables."""
        return [name for name, config in cls.ENV_VARS.items() if config['required']]
    
    @classmethod
    def validate_configuration(cls) -> bool:
        """
        Validate that all required configuration is present and valid.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        env_status = cls.load_environment()
        
        # Check for missing required variables
        if env_status.get('missing_required'):
            return False
        
        # Check for invalid variables
        if env_status.get('invalid'):
            return False
        
        return True
    
    @classmethod
    def get_configuration_summary(cls) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        env_status = cls.load_environment()
        
        return {
            'valid_count': len(env_status.get('valid', {})),
            'invalid_count': len(env_status.get('invalid', {})),
            'missing_required_count': len(env_status.get('missing_required', [])),
            'using_defaults_count': len(env_status.get('using_defaults', [])),
            'is_valid': cls.validate_configuration(),
            'project_id': os.getenv('GOOGLE_CLOUD_PROJECT', 'not-configured'),
            'agent_mode': os.getenv('AGENT_MODE', 'sqlite'),
            'database_path': os.getenv('DATABASE_PATH', 'not-configured'),
        }

# Convenience functions for backward compatibility
def load_environment() -> Dict[str, Any]:
    """Load and validate all environment variables."""
    return EnvironmentConfig.load_environment()

def validate_configuration() -> bool:
    """Validate that all required configuration is present and valid."""
    return EnvironmentConfig.validate_configuration()

def get_configuration_summary() -> Dict[str, Any]:
    """Get a summary of the current configuration."""
    return EnvironmentConfig.get_configuration_summary()

# Auto-load environment on module import
try:
    load_environment()
    logger.info("🔧 Environment configuration loaded automatically")
except Exception as e:
    logger.error(f"❌ Failed to load environment configuration: {e}")
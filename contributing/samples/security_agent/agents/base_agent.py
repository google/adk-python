"""
Base Agent Configuration

Base agent class and common functionality for ADK agents.
This module provides the foundation for creating specialized agents.
"""

import os
from pathlib import Path
from typing import List, Any
import vertexai
import google.genai


def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key and value and value != 'your-api-key-here':
                        os.environ.setdefault(key.strip(), value.strip())


def initialize_vertex_ai():
    """Initialize Vertex AI with project and location settings"""
    # Load .env configuration
    load_env_file()

    # Set Vertex AI environment variables for auto-detection
    os.environ.setdefault('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
    os.environ.setdefault('GOOGLE_CLOUD_LOCATION', 'us-central1')

    # Set google-genai specific environment variables for Vertex AI configuration
    os.environ.setdefault('GOOGLE_GENAI_USE_VERTEXAI', 'true')
    os.environ.setdefault('GOOGLE_GENAI_PROJECT', 'mgm-digitalconcierge')
    os.environ.setdefault('GOOGLE_GENAI_LOCATION', 'us-central1')

    try:
        # Initialize Vertex AI
        vertexai.init(
            project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge'),
            location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
        )
        print(f"✅ Vertex AI initialized for project: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')}")
        return True
    except Exception as e:
        print(f"❌ Vertex AI initialization failed: {e}")
        return False


def collect_tools_from_modules(tool_modules: List[Any]) -> List[Any]:
    """Collect all tools from specified modules.
    
    Args:
        tool_modules: List of tool modules to collect tools from
        
    Returns:
        List of tool functions
    """
    tools = []
    for module in tool_modules:
        # Get all functions from the module
        for item_name in dir(module):
            item = getattr(module, item_name)
            if callable(item) and not item_name.startswith('_'):
                tools.append(item)
    return tools
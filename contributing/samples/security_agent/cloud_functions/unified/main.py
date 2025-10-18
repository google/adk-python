"""
Vellox wrapper for Cloud Functions deployment

This module wraps the FastAPI application with Vellox to make it
compatible with Google Cloud Functions HTTP triggers.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Vellox and wrap our ASGI app
try:
    from vellox import wrap_asgi
    from app.main import app

    # Create the Cloud Function handler
    # Vellox will handle the conversion between Cloud Functions format and ASGI
    unified_handler = wrap_asgi(app)

except ImportError as e:
    # Fallback for local testing without Vellox
    print(f"Vellox not installed, falling back to direct FastAPI: {e}")

    from app.main import app
    import functions_framework

    @functions_framework.http
    def unified_handler(request):
        """Fallback handler for testing without Vellox"""
        # Simple routing for testing
        path = request.path
        method = request.method

        if path == "/" and method == "GET":
            return {
                "service": "Unified Security Data Fetchers",
                "version": "2.0.0",
                "status": "healthy",
                "mode": "fallback",
                "timestamp": datetime.utcnow().isoformat()
            }

        return {"error": "Vellox not installed - limited functionality"}, 501


# Export the handler for Cloud Functions
__all__ = ['unified_handler']

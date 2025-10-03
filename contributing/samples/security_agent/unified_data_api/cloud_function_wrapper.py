#!/usr/bin/env python3
"""
Cloud Function wrapper using Vellox
Wraps the FastAPI app to deploy as a single Cloud Function

Installation:
    pip install vellox

Deployment:
    gcloud functions deploy unified-data-api \
        --gen2 \
        --runtime=python311 \
        --region=us-central1 \
        --source=. \
        --entry-point=main \
        --trigger-http \
        --allow-unauthenticated
"""

from vellox import Vellox
from .main import app

# Wrap FastAPI app for Cloud Functions
vellox = Vellox(app)


def main(request):
    """
    Cloud Function entry point

    Args:
        request: HTTP request object

    Returns:
        HTTP response
    """
    return vellox(request)


# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

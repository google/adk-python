"""
Google Cloud API Keys - Thin client for API Key management.

This module provides a thin client wrapper around the Google Cloud API Keys V2 API
for creating, managing, and securing API keys.

Docs: https://cloud.google.com/python/docs/reference/apikeys/latest
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import the Google Cloud API Keys client
try:
    from google.cloud import api_keys_v2
    from google.api_core import exceptions as gcp_exceptions
    API_KEYS_CLIENT_AVAILABLE = True
    logger.info(" Google Cloud API Keys client available")
except ImportError:
    API_KEYS_CLIENT_AVAILABLE = False

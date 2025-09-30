"""
Base configuration and shared utilities for BigQuery tools
"""

import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
DEFAULT_DATASET = os.getenv("DEFAULT_DATASET", "security_insights")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "security_findings")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "100"))
ENABLE_EXPLORATION = os.getenv("ENABLE_EXPLORATION", "true").lower() == "true"

# Initialize BigQuery client
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    logger.info(f"✅ BigQuery client initialized for project: {PROJECT_ID}")
except Exception as e:
    logger.error(f"❌ Failed to initialize BigQuery client: {e}")
    bq_client = None

def check_client():
    """Check if BigQuery client is initialized"""
    if not bq_client:
        raise Exception("BigQuery client not initialized")
    return bq_client

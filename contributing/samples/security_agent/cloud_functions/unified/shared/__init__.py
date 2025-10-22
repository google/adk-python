"""
Shared utilities for unified Cloud Functions
"""

from .auth import get_authenticated_client
from .bigquery_utils import (
    ensure_dataset_exists,
    ensure_table_exists,
    insert_rows_batch,
    get_bq_client
)
from .response import create_response, create_error_response
from .config import Config

__all__ = [
    'get_authenticated_client',
    'ensure_dataset_exists',
    'ensure_table_exists',
    'insert_rows_batch',
    'get_bq_client',
    'create_response',
    'create_error_response',
    'Config'
]
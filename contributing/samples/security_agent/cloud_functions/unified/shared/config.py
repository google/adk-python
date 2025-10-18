"""
Configuration management for unified Cloud Functions
"""

import os
from typing import Optional


class Config:
    """Centralized configuration for Cloud Functions"""

    # Project configuration
    PROJECT_ID: str = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    REGION: str = os.environ.get('REGION', 'us-central1')

    # BigQuery configuration
    BQ_DATASET_ID: str = os.environ.get('BQ_DATASET_ID', 'security_insights')
    BQ_LOCATION: str = os.environ.get('BQ_LOCATION', 'us-central1')  # Use us-central1 for consistency

    # Function configuration
    MAX_RETRIES: int = int(os.environ.get('MAX_RETRIES', '3'))
    TIMEOUT_SECONDS: int = int(os.environ.get('TIMEOUT_SECONDS', '300'))

    # Feature flags
    ENABLE_SAMPLE_DATA: bool = os.environ.get('ENABLE_SAMPLE_DATA', 'true').lower() == 'true'
    ENABLE_CACHING: bool = os.environ.get('ENABLE_CACHING', 'false').lower() == 'true'

    # MSA Analyzer specific
    MSA_DATASET_ID: str = os.environ.get('MSA_DATASET_ID', 'security_data')

    @classmethod
    def get_table_id(cls, table_name: str, dataset: Optional[str] = None) -> str:
        """Get fully qualified table ID"""
        dataset_id = dataset or cls.BQ_DATASET_ID
        return f"{cls.PROJECT_ID}.{dataset_id}.{table_name}"

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        required_vars = ['PROJECT_ID']
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
"""
Base fetcher class for all data fetchers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from google.cloud import bigquery
import logging
from datetime import datetime

from shared import (
    get_bq_client,
    ensure_dataset_exists,
    ensure_table_exists,
    insert_rows_batch,
    Config
)

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """Base class for all data fetchers"""

    def __init__(self, bq_client: Optional[bigquery.Client] = None):
        """Initialize the fetcher"""
        self.config = Config
        self._bq_client = bq_client
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def bq_client(self) -> bigquery.Client:
        """Lazily instantiate a BigQuery client."""
        if self._bq_client is None:
            self._bq_client = get_bq_client()
        return self._bq_client

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Name of the BigQuery table"""
        pass

    @property
    @abstractmethod
    def schema(self) -> List[bigquery.SchemaField]:
        """BigQuery table schema"""
        pass

    @property
    def dataset_id(self) -> str:
        """Dataset ID (can be overridden by subclasses)"""
        return self.config.BQ_DATASET_ID

    @property
    def table_id(self) -> str:
        """Full table ID"""
        return self.config.get_table_id(self.table_name, self.dataset_id)

    @abstractmethod
    def fetch_data(self) -> List[Dict[str, Any]]:
        """
        Fetch data from the source

        Returns:
            List of dictionaries representing the data
        """
        pass

    def prepare_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare data for BigQuery insertion

        Args:
            raw_data: Raw fetched data

        Returns:
            Prepared data for BigQuery
        """
        # Add common fields
        for record in raw_data:
            if 'ingestion_time' not in record:
                record['ingestion_time'] = datetime.utcnow().isoformat()
            if 'project_id' not in record:
                record['project_id'] = self.config.PROJECT_ID

        return raw_data

    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate data before insertion

        Args:
            data: Data to validate

        Returns:
            True if data is valid
        """
        if not data:
            self.logger.warning("No data to validate")
            return False

        # Check that all records have required fields
        required_fields = [field.name for field in self.schema if field.mode == "REQUIRED"]

        for idx, record in enumerate(data):
            missing = [field for field in required_fields if field not in record or record[field] is None]
            if missing:
                self.logger.error(f"Record {idx} missing required fields: {missing}")
                return False

        return True

    def setup_infrastructure(self) -> None:
        """Set up BigQuery dataset and table"""
        # Ensure dataset exists
        ensure_dataset_exists(
            self.bq_client,
            self.dataset_id,
            location=self.config.BQ_LOCATION
        )

        # Ensure table exists
        ensure_table_exists(
            self.bq_client,
            self.table_id,
            self.schema
        )

    def run(self) -> Dict[str, Any]:
        """
        Main execution method

        Returns:
            Execution result with status and metadata
        """
        try:
            self.logger.info(f"Starting {self.__class__.__name__} execution")

            # Setup infrastructure
            self.setup_infrastructure()

            # Fetch data
            self.logger.info("Fetching data from source")
            raw_data = self.fetch_data()

            if not raw_data:
                return {
                    'status': 'success',
                    'message': 'No data to process',
                    'records_processed': 0,
                    'table': self.table_id
                }

            # Prepare data
            self.logger.info(f"Preparing {len(raw_data)} records")
            prepared_data = self.prepare_data(raw_data)

            # Validate data
            if not self.validate_data(prepared_data):
                raise ValueError("Data validation failed")

            # Insert data
            self.logger.info(f"Inserting {len(prepared_data)} records into BigQuery")
            errors = insert_rows_batch(
                self.bq_client,
                self.table_id,
                prepared_data
            )

            if errors:
                self.logger.error(f"Insert errors: {errors}")
                return {
                    'status': 'partial_success',
                    'message': f"Inserted data with {len(errors)} errors",
                    'records_processed': len(prepared_data),
                    'errors': errors[:10],  # Limit error details
                    'table': self.table_id
                }

            return {
                'status': 'success',
                'message': f"Successfully processed {len(prepared_data)} records",
                'records_processed': len(prepared_data),
                'table': self.table_id
            }

        except Exception as e:
            self.logger.error(f"Execution failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'records_processed': 0,
                'table': self.table_id
            }

    def get_sample_data(self) -> List[Dict[str, Any]]:
        """
        Get sample data for testing/development

        Returns:
            List of sample records
        """
        return []

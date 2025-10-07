#!/usr/bin/env python3
"""
Modular BigQuery operations for the unified data API
Separates table creation, schema management, and data insertion
"""

from typing import List, Dict, Any, Type, Optional
from google.cloud import bigquery
from pydantic import BaseModel
import logging
from datetime import datetime

from .models import (
    IAMAccount, CustomRole, ServiceAccountRole,
    ComputeInstance, FirewallRule, Network,
    StorageBucket, SecurityFinding, SecurityFeed,
    ReleaseNote, ConfluencePage
)

logger = logging.getLogger(__name__)


class BigQueryOperations:
    """Handles all BigQuery operations with type safety"""

    def __init__(self, project_id: str, dataset_id: str = "security_insights"):
        """
        Initialize BigQuery operations

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset name
        """
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_ref = f"{project_id}.{dataset_id}"

        # Model to table mapping
        self.table_mappings = {
            "iam_accounts": IAMAccount,
            "custom_roles": CustomRole,
            "service_account_roles": ServiceAccountRole,
            "compute_instances": ComputeInstance,
            "firewall_rules": FirewallRule,
            "networks": Network,
            "storage_buckets": StorageBucket,
            "security_findings": SecurityFinding,
            "security_feeds": SecurityFeed,
            "release_notes": ReleaseNote,
            "confluence_pages": ConfluencePage,
        }

    # ========================================================================
    # Dataset Operations
    # ========================================================================

    def ensure_dataset_exists(self) -> bool:
        """Ensure the dataset exists, create if it doesn't"""
        try:
            dataset = bigquery.Dataset(self.dataset_ref)
            dataset.location = "US"
            dataset.description = "Security insights and GCP resource inventory"

            self.client.create_dataset(dataset, exists_ok=True)
            logger.info(f"✅ Dataset {self.dataset_id} ready")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create dataset: {e}")
            return False

    # ========================================================================
    # Schema Generation from Pydantic Models
    # ========================================================================

    def _pydantic_to_bq_field(self, field_name: str, field_info: Any) -> bigquery.SchemaField:
        """Convert a Pydantic field to BigQuery SchemaField"""
        from typing import get_origin, get_args
        import datetime as dt

        field_type = field_info.annotation
        origin = get_origin(field_type)

        # Handle Optional types
        mode = "NULLABLE"
        if origin is not None:
            if origin is list or origin is List:
                mode = "REPEATED"
                field_type = get_args(field_type)[0]
            elif str(origin) in ["typing.Union", "types.UnionType"]:
                # Optional[X] is Union[X, None]
                args = get_args(field_type)
                field_type = args[0] if args[1] is type(None) else args[0]

        # Map Python types to BigQuery types
        type_mapping = {
            str: "STRING",
            int: "INTEGER",
            float: "FLOAT",
            bool: "BOOLEAN",
            dt.datetime: "TIMESTAMP",
            dt.date: "DATE",
            dt.time: "TIME",
            dict: "JSON",
            Dict: "JSON",
        }

        # Get BQ type
        bq_type = "STRING"  # default
        for py_type, bq in type_mapping.items():
            if field_type == py_type or (hasattr(field_type, "__origin__") and field_type.__origin__ == py_type):
                bq_type = bq
                break

        # Handle Enums
        if hasattr(field_type, "__bases__") and any(base.__name__ == "Enum" for base in field_type.__bases__):
            bq_type = "STRING"

        description = field_info.description or ""

        return bigquery.SchemaField(
            name=field_name,
            field_type=bq_type,
            mode=mode,
            description=description
        )

    def generate_schema_from_model(self, model_class: Type[BaseModel]) -> List[bigquery.SchemaField]:
        """
        Generate BigQuery schema from Pydantic model

        Args:
            model_class: Pydantic model class

        Returns:
            List of BigQuery SchemaField objects
        """
        schema = []
        for field_name, field_info in model_class.model_fields.items():
            try:
                schema_field = self._pydantic_to_bq_field(field_name, field_info)
                schema.append(schema_field)
            except Exception as e:
                logger.warning(f"Could not convert field {field_name}: {e}")
                # Fallback to STRING
                schema.append(bigquery.SchemaField(
                    name=field_name,
                    field_type="STRING",
                    mode="NULLABLE"
                ))

        return schema

    # ========================================================================
    # Table Operations
    # ========================================================================

    def create_table(
        self,
        table_name: str,
        model_class: Optional[Type[BaseModel]] = None,
        schema: Optional[List[bigquery.SchemaField]] = None,
        overwrite: bool = False
    ) -> bool:
        """
        Create a BigQuery table from Pydantic model or schema

        Args:
            table_name: Name of the table
            model_class: Pydantic model class to generate schema from
            schema: Explicit schema (if not using model_class)
            overwrite: Whether to replace existing table

        Returns:
            True if successful
        """
        try:
            table_ref = f"{self.dataset_ref}.{table_name}"

            # Delete existing table if overwrite=True
            if overwrite:
                try:
                    self.client.delete_table(table_ref)
                    logger.info(f"Deleted existing table {table_name}")
                except Exception:
                    pass  # Table doesn't exist

            # Generate schema
            if model_class:
                schema = self.generate_schema_from_model(model_class)
            elif not schema:
                raise ValueError("Must provide either model_class or schema")

            # Create table
            table = bigquery.Table(table_ref, schema=schema)
            table.description = f"Auto-generated from {model_class.__name__}" if model_class else ""

            # Partitioning by created_at if field exists
            if any(field.name == "created_at" for field in schema):
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="created_at"
                )

            self.client.create_table(table, exists_ok=not overwrite)
            logger.info(f"✅ Table {table_name} created with {len(schema)} fields")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create table {table_name}: {e}")
            return False

    def create_all_tables(self, overwrite: bool = False) -> Dict[str, bool]:
        """
        Create all tables from the table_mappings

        Args:
            overwrite: Whether to replace existing tables

        Returns:
            Dict mapping table names to success status
        """
        results = {}
        self.ensure_dataset_exists()

        for table_name, model_class in self.table_mappings.items():
            success = self.create_table(
                table_name=table_name,
                model_class=model_class,
                overwrite=overwrite
            )
            results[table_name] = success

        return results

    # ========================================================================
    # Insert Operations
    # ========================================================================

    def insert_records(
        self,
        table_name: str,
        records: List[BaseModel],
        model_class: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        Insert Pydantic model instances into BigQuery table

        Args:
            table_name: Target table name
            records: List of Pydantic model instances
            model_class: Optional model class for validation

        Returns:
            Dict with success status and metrics
        """
        if not records:
            return {
                "success": True,
                "inserted": 0,
                "failed": 0,
                "errors": []
            }

        try:
            table_ref = f"{self.dataset_ref}.{table_name}"

            # Convert Pydantic models to dicts
            rows_to_insert = []
            for record in records:
                row_dict = record.model_dump(mode='json')
                rows_to_insert.append(row_dict)

            # Perform insert
            errors = self.client.insert_rows_json(table_ref, rows_to_insert)

            if errors:
                logger.error(f"Insert errors for {table_name}: {errors}")
                return {
                    "success": False,
                    "inserted": len(records) - len(errors),
                    "failed": len(errors),
                    "errors": errors
                }

            logger.info(f"✅ Inserted {len(records)} records into {table_name}")
            return {
                "success": True,
                "inserted": len(records),
                "failed": 0,
                "errors": []
            }

        except Exception as e:
            logger.error(f"❌ Insert failed for {table_name}: {e}")
            return {
                "success": False,
                "inserted": 0,
                "failed": len(records),
                "errors": [str(e)]
            }

    def upsert_records(
        self,
        table_name: str,
        records: List[BaseModel],
        key_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Upsert records using MERGE statement (insert or update based on key)

        Args:
            table_name: Target table name
            records: List of Pydantic model instances
            key_fields: Fields to use as unique key for matching

        Returns:
            Dict with success status and metrics
        """
        if not records:
            return {"success": True, "upserted": 0}

        try:
            table_ref = f"{self.dataset_ref}.{table_name}"

            # Create temporary table
            temp_table_name = f"{table_name}_temp_{int(datetime.utcnow().timestamp())}"
            temp_table_ref = f"{self.dataset_ref}.{temp_table_name}"

            # Get schema from main table
            main_table = self.client.get_table(table_ref)
            temp_table = bigquery.Table(temp_table_ref, schema=main_table.schema)
            self.client.create_table(temp_table)

            # Insert into temp table
            rows_to_insert = [record.model_dump(mode='json') for record in records]
            self.client.insert_rows_json(temp_table_ref, rows_to_insert)

            # Build MERGE query
            key_condition = " AND ".join([f"target.{k} = source.{k}" for k in key_fields])
            all_fields = [field.name for field in main_table.schema]
            update_set = ", ".join([f"{f} = source.{f}" for f in all_fields if f not in key_fields])
            insert_fields = ", ".join(all_fields)
            insert_values = ", ".join([f"source.{f}" for f in all_fields])

            merge_query = f"""
            MERGE `{table_ref}` AS target
            USING `{temp_table_ref}` AS source
            ON {key_condition}
            WHEN MATCHED THEN
                UPDATE SET {update_set}
            WHEN NOT MATCHED THEN
                INSERT ({insert_fields})
                VALUES ({insert_values})
            """

            # Execute MERGE
            query_job = self.client.query(merge_query)
            query_job.result()

            # Clean up temp table
            self.client.delete_table(temp_table_ref)

            logger.info(f"✅ Upserted {len(records)} records into {table_name}")
            return {
                "success": True,
                "upserted": len(records),
                "errors": []
            }

        except Exception as e:
            logger.error(f"❌ Upsert failed for {table_name}: {e}")
            # Try to clean up temp table
            try:
                self.client.delete_table(temp_table_ref)
            except:
                pass

            return {
                "success": False,
                "upserted": 0,
                "errors": [str(e)]
            }

    # ========================================================================
    # Query Operations
    # ========================================================================

    def query_to_models(
        self,
        query: str,
        model_class: Type[BaseModel]
    ) -> List[BaseModel]:
        """
        Execute query and convert results to Pydantic models

        Args:
            query: SQL query string
            model_class: Pydantic model class to convert rows to

        Returns:
            List of model instances
        """
        try:
            query_job = self.client.query(query)
            results = query_job.result()

            models = []
            for row in results:
                row_dict = dict(row.items())
                model = model_class(**row_dict)
                models.append(model)

            return models

        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return []

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get metadata about a table"""
        try:
            table_ref = f"{self.dataset_ref}.{table_name}"
            table = self.client.get_table(table_ref)

            return {
                "table_name": table_name,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None,
                "schema_fields": len(table.schema),
                "partitioning": str(table.time_partitioning) if table.time_partitioning else None
            }
        except Exception as e:
            logger.error(f"Failed to get info for {table_name}: {e}")
            return {"error": str(e)}

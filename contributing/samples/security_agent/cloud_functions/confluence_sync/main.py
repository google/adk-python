"""
Cloud Function for Syncing Confluence Documentation to BigQuery

This function:
1. Fetches documentation from Confluence API
2. Processes and enriches the content
3. Stores in BigQuery for analysis
4. Can be triggered via HTTP or Cloud Scheduler
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
import time
from urllib.parse import parse_qs, urlparse

import requests
from google.cloud import bigquery
from google.cloud import secretmanager
from google.cloud.exceptions import NotFound
import functions_framework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment helpers


def _load_env_file_if_present() -> None:
    """Load environment variables from a .env style file if provided."""

    def _load_file(path: Optional[str]) -> bool:
        if not path:
            return False
        expanded = os.path.expanduser(path)
        if not os.path.isfile(expanded):
            return False

        logger.info("Loading environment variables from %s", expanded)
        with open(expanded, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Do not override existing environment variables
                os.environ.setdefault(key, value)
        return True

    # Explicit path takes precedence, fall back to local .env if present
    explicit_path = os.environ.get("CONFLUENCE_ENV_FILE")
    if not _load_file(explicit_path):
        default_path = os.path.join(os.path.dirname(__file__), ".env")
        _load_file(default_path)


_load_env_file_if_present()


# Configuration
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
DATASET_ID = os.environ.get('BQ_DATASET_ID', 'security_data')
TABLE_ID = os.environ.get('BQ_TABLE_ID', 'confluence_documents')
CONFLUENCE_SPACES = os.environ.get('CONFLUENCE_SPACES', 'SEC,POLICY,GCP').split(',')
SYNC_BATCH_SIZE = int(os.environ.get('SYNC_BATCH_SIZE', '50'))

# Secret Manager for sensitive credentials
SECRET_CONFLUENCE_URL = os.environ.get('SECRET_CONFLUENCE_URL', 'confluence-url')
SECRET_CONFLUENCE_USER = os.environ.get('SECRET_CONFLUENCE_USER', 'confluence-username')
SECRET_CONFLUENCE_TOKEN = os.environ.get('SECRET_CONFLUENCE_TOKEN', 'confluence-api-token')


class ConfluenceAPIClient:
    """Minimal Confluence Cloud REST v2 client focused on page retrieval."""

    def __init__(self, base_url: str, username: str, api_token: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_root = f"{self.base_url}/wiki/api/v2"
        self.site_root = f"{self.base_url}/wiki"
        self.timeout = timeout

        self.session = requests.Session()
        self.session.auth = (username, api_token)
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.api_root}{path}"
        for attempt in range(3):
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
            )

            if response.status_code == 429 or response.status_code >= 500:
                sleep_time = min(2 ** attempt, 8)
                logger.warning(
                    "Confluence API %s %s failed with %s. Retrying in %s seconds",
                    method,
                    path,
                    response.status_code,
                    sleep_time,
                )
                time.sleep(sleep_time)
                continue

            if response.status_code >= 400:
                try:
                    details = response.json()
                except ValueError:
                    details = response.text
                raise RuntimeError(
                    f"Confluence API error {response.status_code} for {method} {path}: {details}"
                )

            if response.content:
                return response.json()
            return {}

        raise RuntimeError(f"Confluence API request {method} {path} failed after retries")

    def get_space_id(self, space_key: str) -> str:
        data = self._request("get", "/spaces", params={"keys": space_key})
        results = data.get("results", [])
        if not results:
            raise ValueError(f"Space with key {space_key} not found")
        # API returns id as int or string
        return str(results[0].get("id"))

    def list_space_pages(
        self,
        space_id: str,
        limit: int = SYNC_BATCH_SIZE,
        cursor: Optional[str] = None,
        modified_since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "space-id": space_id,
            "limit": limit,
            "status": "current",
            "sort": "updated-date DESC",
        }
        if cursor:
            params["cursor"] = cursor
        if modified_since:
            iso_value = modified_since.replace(microsecond=0).isoformat()
            if iso_value.endswith("+00:00"):
                iso_value = iso_value[:-6] + "Z"
            elif not iso_value.endswith("Z"):
                iso_value = f"{iso_value}Z"
            params["modified-since"] = iso_value
        return self._request("get", "/pages", params=params)

    def get_page(self, page_id: str) -> Dict[str, Any]:
        params = {
            "body-format": "storage",
            "include": "ancestors,body.storage,metadata.labels,version",
        }
        # Uses https://developer.atlassian.com/cloud/confluence/rest/v2/api-group-page/#api-pages-id-get
        return self._request("get", f"/pages/{page_id}", params=params)

    def get_page_labels(self, page_id: str) -> List[Dict[str, Any]]:
        data = self._request("get", f"/pages/{page_id}/labels")
        return data.get("results", [])

    @staticmethod
    def extract_cursor(next_link: Optional[str]) -> Optional[str]:
        if not next_link:
            return None
        parsed = urlparse(next_link)
        if not parsed.query:
            return next_link
        query = parse_qs(parsed.query)
        cursor_values = query.get("cursor")
        if cursor_values:
            return cursor_values[0]
        return next_link

    def build_page_url(self, page_data: Dict[str, Any]) -> Optional[str]:
        links = page_data.get("_links", {})
        webui = links.get("webui")
        if webui:
            return f"{self.site_root}{webui}"
        page_id = page_data.get("id")
        if page_id:
            return f"{self.site_root}/pages/{page_id}"
        return None


class ConfluenceBigQuerySync:
    """Handles syncing Confluence documents to BigQuery."""

    def __init__(self):
        """Initialize the sync service."""
        self.bq_client = bigquery.Client(project=PROJECT_ID)
        self.secrets_client = secretmanager.SecretManagerServiceClient()
        self.confluence = None
        self._init_bigquery_dataset()

    def _get_secret(self, secret_id: str) -> str:
        """Retrieve secret from Secret Manager."""
        try:
            name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
            response = self.secrets_client.access_secret_version(request={"name": name})
            return response.payload.data.decode('UTF-8')
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {str(e)}")
            # Fall back to environment variable
            return os.environ.get(secret_id.upper().replace('-', '_'), '')

    def _init_confluence_client(self):
        """Initialize Confluence client with credentials from Secret Manager."""
        if not self.confluence:
            url = self._get_secret(SECRET_CONFLUENCE_URL)
            username = self._get_secret(SECRET_CONFLUENCE_USER)
            api_token = self._get_secret(SECRET_CONFLUENCE_TOKEN)

            if not all([url, username, api_token]):
                raise ValueError("Missing Confluence credentials")

            self.confluence = ConfluenceAPIClient(
                base_url=url,
                username=username,
                api_token=api_token,
            )
            logger.info(f"Initialized Confluence API client for {url}")

    def _init_bigquery_dataset(self):
        """Create BigQuery dataset and table if they don't exist."""
        # Create dataset if not exists
        dataset_id = f"{PROJECT_ID}.{DATASET_ID}"
        try:
            dataset = self.bq_client.get_dataset(dataset_id)
            logger.info(f"Dataset {dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            dataset.description = "Security and compliance documentation from Confluence"
            dataset = self.bq_client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset_id}")

        # Create table if not exists
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        try:
            table = self.bq_client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except NotFound:
            schema = [
                bigquery.SchemaField("document_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("space_key", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("content", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("content_text", "STRING", mode="NULLABLE"),  # Plain text version
                bigquery.SchemaField("url", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("created_date", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("modified_date", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("created_by", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("modified_by", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("parent_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("parent_title", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("labels", "STRING", mode="REPEATED"),
                bigquery.SchemaField("content_hash", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("word_count", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("has_attachments", "BOOLEAN", mode="NULLABLE"),
                bigquery.SchemaField("attachment_count", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("version_number", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("sync_timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("sync_status", "STRING", mode="NULLABLE"),
                # Additional metadata fields
                bigquery.SchemaField("security_classification", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("compliance_tags", "STRING", mode="REPEATED"),
                bigquery.SchemaField("document_type", "STRING", mode="NULLABLE"),
            ]

            table = bigquery.Table(table_id, schema=schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="sync_timestamp"
            )
            table = self.bq_client.create_table(table)
            logger.info(f"Created table {table_id}")

        # Create audit log table
        audit_table_id = f"{PROJECT_ID}.{DATASET_ID}.confluence_sync_audit"
        try:
            audit_table = self.bq_client.get_table(audit_table_id)
        except NotFound:
            audit_schema = [
                bigquery.SchemaField("sync_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("sync_timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("sync_type", "STRING", mode="NULLABLE"),  # full, incremental
                bigquery.SchemaField("spaces_synced", "STRING", mode="REPEATED"),
                bigquery.SchemaField("documents_processed", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("documents_added", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("documents_updated", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("documents_deleted", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("errors_count", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("error_details", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("duration_seconds", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("status", "STRING", mode="NULLABLE"),  # success, partial, failed
            ]

            audit_table = bigquery.Table(audit_table_id, schema=audit_schema)
            audit_table = self.bq_client.create_table(audit_table)
            logger.info(f"Created audit table {audit_table_id}")

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract plain text from HTML content."""
        if not html_content:
            return ""

        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', html_content)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _classify_document(self, content: str, title: str, labels: List[str]) -> Dict[str, Any]:
        """Classify document based on content and metadata."""
        classification = {
            "document_type": "general",
            "security_classification": "internal",
            "compliance_tags": []
        }

        # Determine document type based on title and content
        title_lower = title.lower()
        content_lower = content.lower() if content else ""

        if any(term in title_lower for term in ['policy', 'procedure', 'standard']):
            classification["document_type"] = "policy"
        elif any(term in title_lower for term in ['guide', 'how-to', 'tutorial']):
            classification["document_type"] = "guide"
        elif any(term in title_lower for term in ['architecture', 'design', 'diagram']):
            classification["document_type"] = "architecture"
        elif any(term in title_lower for term in ['runbook', 'playbook', 'incident']):
            classification["document_type"] = "runbook"

        # Determine security classification
        if any(term in content_lower for term in ['confidential', 'restricted', 'sensitive']):
            classification["security_classification"] = "confidential"
        elif any(term in content_lower for term in ['public', 'external']):
            classification["security_classification"] = "public"

        # Identify compliance tags
        compliance_keywords = {
            "pci": ["pci", "pci-dss", "payment card"],
            "hipaa": ["hipaa", "phi", "protected health"],
            "gdpr": ["gdpr", "data protection", "privacy"],
            "sox": ["sox", "sarbanes", "financial"],
            "iso27001": ["iso 27001", "iso27001", "information security"],
            "cis": ["cis benchmark", "cis controls"]
        }

        for tag, keywords in compliance_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                classification["compliance_tags"].append(tag)

        # Add labels as compliance tags if they match
        for label in labels:
            if label.lower() in compliance_keywords:
                if label.lower() not in classification["compliance_tags"]:
                    classification["compliance_tags"].append(label.lower())

        return classification

    def fetch_confluence_documents(self, spaces: List[str],
                                  modified_since: Optional[datetime] = None) -> List[Dict]:
        """Fetch documents from Confluence spaces using the REST v2 API."""
        self._init_confluence_client()
        all_documents: List[Dict] = []

        for space in spaces:
            try:
                logger.info(f"Fetching documents from space: {space}")
                space_id = self.confluence.get_space_id(space)
            except Exception as space_error:
                logger.error(f"Unable to resolve Confluence space {space}: {space_error}")
                continue

            cursor: Optional[str] = None
            while True:
                try:
                    page_batch = self.confluence.list_space_pages(
                        space_id=space_id,
                        limit=SYNC_BATCH_SIZE,
                        cursor=cursor,
                        modified_since=modified_since,
                    )
                except Exception as request_error:
                    logger.error(
                        f"Failed to list pages for space {space}: {request_error}"
                    )
                    break

                pages = page_batch.get("results", [])
                if not pages:
                    break

                for page_summary in pages:
                    page_id = str(page_summary.get("id") or page_summary.get("contentId") or "")
                    if not page_id:
                        logger.warning("Skipping page without id in space %s", space)
                        continue

                    try:
                        page_details = self.confluence.get_page(page_id)
                        if not page_details:
                            logger.warning("Empty response for page %s", page_id)
                            continue

                        # Enrich with labels if not included
                        if "labels" not in page_details or not page_details.get("labels"):
                            page_details["labels"] = self.confluence.get_page_labels(page_id)

                        doc = self._process_confluence_page(
                            page_details,
                            space_key=space,
                            page_summary=page_summary,
                        )
                        all_documents.append(doc)
                    except Exception as page_error:
                        logger.error(
                            "Error processing page %s in space %s: %s",
                            page_id,
                            space,
                            page_error,
                        )

                cursor = None
                # Prefer cursor from API, fall back to links.next URL
                cursor = page_batch.get("cursor", {}).get("next") if isinstance(page_batch.get("cursor"), dict) else page_batch.get("cursor")
                if not cursor:
                    cursor = page_batch.get("_links", {}).get("next") or page_batch.get("links", {}).get("next")
                cursor = ConfluenceAPIClient.extract_cursor(cursor)

                if not cursor:
                    break

            logger.info(f"Fetched {len(all_documents)} total documents after space {space}")

        return all_documents

    def _process_confluence_page(
        self,
        page: Dict[str, Any],
        space_key: str,
        page_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a Confluence page into BigQuery format."""

        body = page.get("body", {})
        content_html = (
            body.get("storage", {}).get("value")
            or body.get("value", "")
            or page_summary.get("body", {}).get("storage", {}).get("value", "")
            if page_summary
            else ""
        )
        content_html = content_html or ""
        content_text = self._extract_text_from_html(content_html)

        labels: List[str] = []
        if isinstance(page.get("labels"), list):
            labels = [label.get("name", "") for label in page.get("labels", []) if isinstance(label, dict)]
        elif isinstance(page.get("labels"), dict):
            labels = [label.get("name", "") for label in page.get("labels", {}).get("results", [])]
        else:
            metadata_labels = page.get("metadata", {}).get("labels", {}).get("results", [])
            labels = [label.get("name", "") for label in metadata_labels]

        ancestors = page.get("ancestors")
        parent_id = None
        parent_title = None
        if isinstance(ancestors, list) and ancestors:
            ancestor = ancestors[-1]
            if isinstance(ancestor, dict):
                parent_id = ancestor.get("id") or ancestor.get("contentId")
                parent_title = ancestor.get("title")
        if not parent_id:
            parent_id = page.get("parentId")

        has_attachments = "attachment" in content_html.lower()

        title = page.get("title") or (page_summary or {}).get("title", "")
        classification = self._classify_document(content_text, title, labels)

        created_date = (
            page.get("createdAt")
            or page.get("createdDate")
            or page.get("history", {}).get("createdDate")
        )
        modified_date = (
            page.get("updatedAt")
            or page.get("modifiedAt")
            or page.get("version", {}).get("when")
            or page.get("history", {}).get("lastUpdated", {}).get("when")
        )

        created_by = (
            page.get("createdBy", {}).get("displayName")
            if isinstance(page.get("createdBy"), dict)
            else (page.get("history", {}).get("createdBy", {}).get("displayName", ""))
        )
        modified_by = (
            page.get("updatedBy", {}).get("displayName")
            if isinstance(page.get("updatedBy"), dict)
            else (page.get("history", {}).get("lastUpdated", {}).get("by", {}).get("displayName", ""))
        )

        page_id = str(page.get("id") or page.get("contentId") or page_summary.get("id") if page_summary else "")
        page_url = self.confluence.build_page_url(page) if hasattr(self.confluence, "build_page_url") else None
        if not page_url and page_summary:
            links = page_summary.get("_links", {})
            webui = links.get("webui")
            if webui and hasattr(self.confluence, "site_root"):
                page_url = f"{self.confluence.site_root}{webui}"

        document = {
            'document_id': page_id,
            'space_key': space_key,
            'title': title,
            'content': content_html[:1000000],  # Limit content size
            'content_text': content_text[:500000],  # Plain text version
            'url': page_url,
            'created_date': self._parse_confluence_date(created_date),
            'modified_date': self._parse_confluence_date(modified_date),
            'created_by': created_by or '',
            'modified_by': modified_by or '',
            'parent_id': parent_id,
            'parent_title': parent_title,
            'labels': [label for label in labels if label],
            'content_hash': hashlib.md5(content_html.encode()).hexdigest(),
            'word_count': len(content_text.split()) if content_text else 0,
            'has_attachments': has_attachments,
            'attachment_count': 0,  # Would need additional API call
            'version_number': page.get('version', {}).get('number', 1) if isinstance(page.get('version'), dict) else 1,
            'sync_timestamp': datetime.utcnow().isoformat(),
            'sync_status': 'success',
            'security_classification': classification['security_classification'],
            'compliance_tags': classification['compliance_tags'],
            'document_type': classification['document_type']
        }

        return document

    def _parse_confluence_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse Confluence date string to ISO format."""
        if not date_str:
            return None

        try:
            if isinstance(date_str, datetime):
                return date_str.replace(microsecond=0).isoformat()

            # Confluence dates are in ISO format but may have microseconds
            if isinstance(date_str, str) and '.' in date_str:
                date_str = date_str.split('.')[0] + 'Z'

            dt = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
            return dt.isoformat()
        except Exception as e:
            logger.warning(f"Failed to parse date {date_str}: {str(e)}")
            return None

    def sync_to_bigquery(self, documents: List[Dict]) -> Dict[str, int]:
        """Sync documents to BigQuery."""
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

        stats = {
            'added': 0,
            'updated': 0,
            'errors': 0
        }

        if not documents:
            logger.info("No documents to sync")
            return stats

        # Batch insert documents
        try:
            # First, get existing document IDs to determine updates vs inserts
            query = f"""
                SELECT document_id, content_hash
                FROM `{table_id}`
                WHERE document_id IN ({','.join([f'"{d["document_id"]}"' for d in documents])})
            """

            existing_docs = {}
            try:
                query_job = self.bq_client.query(query)
                for row in query_job:
                    existing_docs[row['document_id']] = row['content_hash']
            except Exception as e:
                logger.warning(f"Could not fetch existing documents: {str(e)}")

            # Prepare rows for insertion
            rows_to_insert = []
            for doc in documents:
                doc_id = doc['document_id']

                # Check if document needs update
                if doc_id in existing_docs:
                    if existing_docs[doc_id] != doc.get('content_hash'):
                        rows_to_insert.append(doc)
                        stats['updated'] += 1
                    # Skip if content hasn't changed
                else:
                    rows_to_insert.append(doc)
                    stats['added'] += 1

            # Insert/update documents
            if rows_to_insert:
                # Use streaming insert for real-time updates
                errors = self.bq_client.insert_rows_json(
                    table_id,
                    rows_to_insert,
                    ignore_unknown_values=True,
                    skip_invalid_rows=False
                )

                if errors:
                    logger.error(f"Failed to insert rows: {errors}")
                    stats['errors'] = len(errors)
                else:
                    logger.info(f"Successfully synced {len(rows_to_insert)} documents")

        except Exception as e:
            logger.error(f"Error syncing to BigQuery: {str(e)}")
            stats['errors'] = len(documents)

        return stats

    def create_sync_audit_record(self, sync_id: str, sync_type: str,
                                spaces: List[str], stats: Dict,
                                duration: float, status: str,
                                error_details: Optional[str] = None):
        """Create audit record for the sync operation."""
        audit_table_id = f"{PROJECT_ID}.{DATASET_ID}.confluence_sync_audit"

        audit_record = {
            'sync_id': sync_id,
            'sync_timestamp': datetime.utcnow().isoformat(),
            'sync_type': sync_type,
            'spaces_synced': spaces,
            'documents_processed': stats.get('added', 0) + stats.get('updated', 0),
            'documents_added': stats.get('added', 0),
            'documents_updated': stats.get('updated', 0),
            'documents_deleted': stats.get('deleted', 0),
            'errors_count': stats.get('errors', 0),
            'error_details': error_details,
            'duration_seconds': duration,
            'status': status
        }

        try:
            errors = self.bq_client.insert_rows_json(
                audit_table_id,
                [audit_record],
                ignore_unknown_values=True
            )

            if errors:
                logger.error(f"Failed to create audit record: {errors}")
            else:
                logger.info(f"Created audit record for sync {sync_id}")

        except Exception as e:
            logger.error(f"Error creating audit record: {str(e)}")


@functions_framework.http
def sync_confluence_to_bigquery(request):
    """
    HTTP Cloud Function entry point.

    Args:
        request: Flask request object

    Returns:
        JSON response with sync results
    """
    start_time = datetime.utcnow()
    sync_id = hashlib.md5(f"{start_time.isoformat()}".encode()).hexdigest()[:12]

    try:
        # Parse request
        request_json = request.get_json(silent=True)

        # Get parameters
        spaces = CONFLUENCE_SPACES
        if request_json and 'spaces' in request_json:
            spaces = request_json['spaces']

        sync_type = 'full'
        modified_since = None
        if request_json and 'sync_type' in request_json:
            sync_type = request_json['sync_type']

        # For incremental sync, get documents modified in last 7 days
        if sync_type == 'incremental':
            modified_since = datetime.utcnow() - timedelta(days=7)

        logger.info(f"Starting {sync_type} sync for spaces: {spaces}")

        # Initialize sync service
        sync_service = ConfluenceBigQuerySync()

        # Fetch documents
        documents = sync_service.fetch_confluence_documents(spaces, modified_since)
        logger.info(f"Fetched {len(documents)} documents from Confluence")

        # Sync to BigQuery
        stats = sync_service.sync_to_bigquery(documents)

        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Determine status
        status = 'success'
        if stats.get('errors', 0) > 0:
            status = 'partial' if stats.get('added', 0) + stats.get('updated', 0) > 0 else 'failed'

        # Create audit record
        sync_service.create_sync_audit_record(
            sync_id=sync_id,
            sync_type=sync_type,
            spaces=spaces,
            stats=stats,
            duration=duration,
            status=status
        )

        # Return response
        response = {
            'success': status != 'failed',
            'sync_id': sync_id,
            'sync_type': sync_type,
            'spaces': spaces,
            'documents_processed': len(documents),
            'documents_added': stats.get('added', 0),
            'documents_updated': stats.get('updated', 0),
            'errors': stats.get('errors', 0),
            'duration_seconds': round(duration, 2),
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        }

        return json.dumps(response), 200

    except Exception as e:
        logger.error(f"Sync failed: {str(e)}")

        # Try to create audit record for failure
        try:
            sync_service = ConfluenceBigQuerySync()
            duration = (datetime.utcnow() - start_time).total_seconds()
            sync_service.create_sync_audit_record(
                sync_id=sync_id,
                sync_type='unknown',
                spaces=[],
                stats={'errors': 1},
                duration=duration,
                status='failed',
                error_details=str(e)
            )
        except:
            pass

        return json.dumps({
            'success': False,
            'error': str(e),
            'sync_id': sync_id,
            'timestamp': datetime.utcnow().isoformat()
        }), 500


# Entry point for Cloud Scheduler (Pub/Sub trigger)
@functions_framework.cloud_event
def sync_confluence_scheduled(cloud_event):
    """
    Cloud Scheduler entry point (via Pub/Sub).

    Args:
        cloud_event: Cloud Event with Pub/Sub message
    """
    # Decode Pub/Sub message
    import base64

    message_data = {}
    if cloud_event.data and "message" in cloud_event.data:
        message = cloud_event.data["message"]
        if "data" in message:
            message_data = json.loads(
                base64.b64decode(message["data"]).decode()
            )

    # Default to incremental sync for scheduled runs
    sync_type = message_data.get('sync_type', 'incremental')
    spaces = message_data.get('spaces', CONFLUENCE_SPACES)

    # Create a mock request object
    class MockRequest:
        def get_json(self, silent=False):
            return {
                'sync_type': sync_type,
                'spaces': spaces
            }

    # Call the main sync function
    result = sync_confluence_to_bigquery(MockRequest())
    logger.info(f"Scheduled sync completed: {result}")
    return result
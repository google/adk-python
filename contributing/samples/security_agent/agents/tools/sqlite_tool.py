"""
SQLite Tool for GCP Security Agent
Provides database query capabilities for security analysis.
"""

import sqlite3
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import database utilities
try:
    from backend.utils.database import (
        get_database_path,
        validate_database,
        get_db_connection,
        create_database_if_missing
    )
    DATABASE_UTILS_AVAILABLE = True
except ImportError:
    DATABASE_UTILS_AVAILABLE = False

# Import ADK tools for compatibility (optional)
try:
    from google.adk.tools import FunctionTool
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    # Create a dummy FunctionTool for compatibility
    class FunctionTool:
        def __init__(self, *args, **kwargs):
            pass

# Import GCP Live Tool for real-time data
try:
    from .gcp_live_tool import gcp_live_tool
    GCP_LIVE_AVAILABLE = True
except ImportError:
    try:
        from gcp_live_tool import gcp_live_tool
        GCP_LIVE_AVAILABLE = True
    except ImportError:
        GCP_LIVE_AVAILABLE = False
        gcp_live_tool = None

# Import Google Search Tool for fallback capability
try:
    from .search_tool import google_search_tool
    SEARCH_AVAILABLE = True
except ImportError:
    try:
        from search_tool import google_search_tool
        SEARCH_AVAILABLE = True
    except ImportError:
        SEARCH_AVAILABLE = False
        google_search_tool = None

logger = logging.getLogger(__name__)

class SQLiteTool:
    """SQLite database tool for security queries."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite tool with database path."""
        if DATABASE_UTILS_AVAILABLE:
            # Use centralized database utilities
            self.db_path = get_database_path()
            logger.info(f"SQLite tool using centralized database path: {self.db_path}")

            # Validate database and create if missing
            is_valid, message = validate_database()
            if not is_valid:
                logger.warning(f"Database validation failed: {message}")
                logger.info("Attempting to create database...")
                created = create_database_if_missing()
                if created:
                    logger.info("Database created successfully")
                else:
                    logger.error("Failed to create database")
        else:
            # Fallback to original logic if database utils not available
            if db_path is None:
                db_path = os.getenv(
                    "DATABASE_PATH",
                    "backend/cache/gcp_data.db"
                )

            # Remove any quotes that might be in the environment variable
            if db_path:
                db_path = db_path.strip('"').strip("'")

            # Convert to Path and ensure absolute
            self.db_path = Path(db_path)

            # If relative path, make it absolute from project root
            if not self.db_path.is_absolute():
                # Get project root (agents/tools/../.. = project root)
                project_root = Path(__file__).parent.parent.parent
                self.db_path = (project_root / db_path).resolve()
            else:
                self.db_path = self.db_path.resolve()

            logger.info(f"SQLite tool using fallback database path: {self.db_path}")

            if not self.db_path.exists():
                logger.warning(f"Database file not found: {self.db_path}")
                # Create empty database if it doesn't exist
                self._create_empty_database()

    def _create_empty_database(self):
        """Create an empty database with basic tables."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create basic security_findings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_findings (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    severity TEXT,
                    state TEXT,
                    resource_name TEXT,
                    description TEXT,
                    recommendation TEXT,
                    event_time TEXT,
                    data TEXT
                )
            """)

            conn.commit()
            conn.close()
            logger.info(f"Created empty database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to create database: {e}")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Execute a SQL query on the database.

        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries

        Returns:
            Dictionary containing query results or error information
        """
        try:
            if DATABASE_UTILS_AVAILABLE:
                # Use centralized database connection
                with get_db_connection() as conn:
                    cursor = conn.cursor()

                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    # Check if it's a SELECT query
                    if query.strip().upper().startswith('SELECT'):
                        rows = cursor.fetchall()
                        # Convert Row objects to dictionaries
                        results = []
                        for row in rows:
                            results.append(dict(row))

                        return {
                            "success": True,
                            "data": results,
                            "row_count": len(results)
                        }
                    else:
                        # For INSERT, UPDATE, DELETE queries
                        conn.commit()
                        affected_rows = cursor.rowcount
                        return {
                            "success": True,
                            "affected_rows": affected_rows,
                            "message": f"Query executed successfully. {affected_rows} rows affected."
                        }
            else:
                # Fallback to original connection logic
                conn = None
                try:
                    conn = sqlite3.connect(str(self.db_path))
                    conn.row_factory = sqlite3.Row  # Enable column access by name
                    cursor = conn.cursor()

                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    # Check if it's a SELECT query
                    if query.strip().upper().startswith('SELECT'):
                        rows = cursor.fetchall()
                        # Convert Row objects to dictionaries
                        results = []
                        for row in rows:
                            results.append(dict(row))

                        conn.close()
                        return {
                            "success": True,
                            "data": results,
                            "row_count": len(results)
                        }
                    else:
                        # For INSERT, UPDATE, DELETE queries
                        conn.commit()
                        affected_rows = cursor.rowcount
                        conn.close()
                        return {
                            "success": True,
                            "affected_rows": affected_rows,
                            "message": f"Query executed successfully. {affected_rows} rows affected."
                        }
                finally:
                    if conn:
                        conn.close()

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def get_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        result = self.execute_query(query)

        if result["success"]:
            return [row["name"] for row in result["data"]]
        return []

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a specific table."""
        query = f"PRAGMA table_info({table_name})"
        result = self.execute_query(query)

        if result["success"]:
            return {
                "table": table_name,
                "columns": result["data"]
            }
        return {"error": result.get("error", "Unknown error")}

    def get_security_findings(self, severity: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """
        Get security findings from the database.

        Args:
            severity: Optional severity filter (CRITICAL, HIGH, MEDIUM, LOW)
            limit: Maximum number of results to return

        Returns:
            Dictionary containing security findings
        """
        if severity:
            query = "SELECT * FROM security_findings WHERE severity = ? LIMIT ?"
            params = (severity, limit)
        else:
            query = "SELECT * FROM security_findings LIMIT ?"
            params = (limit,)

        return self.execute_query(query, params)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from the database."""
        queries = {
            "total_findings": "SELECT COUNT(*) as count FROM security_findings",
            "severity_breakdown": """
                SELECT severity, COUNT(*) as count
                FROM security_findings
                GROUP BY severity
            """,
            "category_breakdown": """
                SELECT category, COUNT(*) as count
                FROM security_findings
                GROUP BY category
            """,
            "total_assets": "SELECT COUNT(*) as count FROM assets",
            "total_buckets": "SELECT COUNT(*) as count FROM storage_buckets",
            "total_firewall_rules": "SELECT COUNT(*) as count FROM firewall_rules"
        }

        stats = {}
        for key, query in queries.items():
            try:
                result = self.execute_query(query)
                if result["success"]:
                    stats[key] = result["data"]
            except:
                stats[key] = None

        # Return in the expected format with success key
        return {
            "success": True,
            "data": stats
        }

# Create a singleton instance
sqlite_tool_instance = SQLiteTool()

# Create ADK-compatible tool if available
if ADK_AVAILABLE:
    # Wrap the execute_query method as an ADK FunctionTool
    sqlite_tool = FunctionTool(func=sqlite_tool_instance.execute_query)
    get_tables_tool = FunctionTool(func=sqlite_tool_instance.get_tables)
    get_summary_tool = FunctionTool(func=sqlite_tool_instance.get_summary_stats)
else:
    # Fallback to the instance itself
    sqlite_tool = sqlite_tool_instance
    get_tables_tool = None
    get_summary_tool = None



def _determine_public_access(bucket: Dict[str, Any]) -> str:
    """
    Determine public access status from GCP bucket data.
    Convert GCP live data format to SQLite-compatible format.
    """
    # Check security findings for public access indicators
    security_findings = bucket.get("security_findings", [])
    for finding in security_findings:
        if finding.get("type") == "PUBLIC_BUCKET":
            return "Public to internet"

    # Check IAM configuration
    public_access_prevention = bucket.get("public_access_prevention")
    if public_access_prevention == "enforced":
        return "Not public"

    # Default to private if no clear indication
    return "Not public"


def query_security_data(query_type: str, **kwargs) -> Dict[str, Any]:
    """
    Queries the security data based on the specified query_type and parameters.

    Args:
        query_type: The type of security data to query (e.g., "security_summary", "assets").
        **kwargs: Additional parameters for the query (severity, limit, category, etc.).

    Returns:
        A dictionary containing the query results.
    """
    logger.info(f"Received query_type: {query_type} with params: {kwargs}")
    params = kwargs  # Use kwargs directly as params
    result = {"success": False, "error": "Invalid query_type or not yet implemented."}

    try:
        if query_type == "security_summary":
            # This would ideally be a complex query joining multiple tables
            # For now, let's return a summary of security findings
            result = sqlite_tool_instance.get_summary_stats()
            if result["success"]:
                result["message"] = "Prioritized security summary (placeholder - needs full implementation)"
            else:
                result["error"] = "Failed to get security summary: " + result.get("error", "Unknown error")

        elif query_type == "assets":
            asset_type = params.get("asset_type")
            service = params.get("service")
            name = params.get("name")
            
            sql_query = "SELECT * FROM assets"
            conditions = []
            sql_params = []

            if asset_type:
                conditions.append("asset_type LIKE ?")
                sql_params.append(f"%{asset_type}%")
            if service:
                conditions.append("service = ?")
                sql_params.append(service)
            if name:
                conditions.append("name LIKE ?")
                sql_params.append(f"%{name}%")
            
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query assets: " + result.get("error", "Unknown error")

        elif query_type == "security_findings":
            severity = params.get("severity")
            category = params.get("category")
            
            sql_query = "SELECT * FROM security_findings"
            conditions = []
            sql_params = []

            if severity:
                conditions.append("severity = ?")
                sql_params.append(severity)
            if category:
                conditions.append("category = ?")
                sql_params.append(category)
            
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query security findings: " + result.get("error", "Unknown error")

        elif query_type == "iam_analysis":
            principal = params.get("principal")
            sql_query = "SELECT * FROM iam_policies" # Assuming an iam_policies table
            sql_params = []
            if principal:
                sql_query += " WHERE principal = ?"
                sql_params.append(principal)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query IAM analysis: " + result.get("error", "Unknown error")

        elif query_type == "storage_buckets":
            # Cache-first strategy: Try SQLite cache first, then live GCP if needed
            bucket_name = params.get("bucket_name")

            # Step 1: Try cache first (fast)
            logger.info("⚡ Checking SQLite cache for storage buckets")
            sql_query = "SELECT * FROM storage_buckets"
            sql_params = []
            if bucket_name:
                sql_query += " WHERE name = ?"
                sql_params.append(bucket_name)

            cache_result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))

            # Step 2: If cache has data, use it (performance optimization)
            if cache_result["success"] and cache_result.get("data"):
                logger.info("📁 Using SQLite cached data for storage buckets (cache hit)")
                result = cache_result
                result["source"] = "sqlite_cache"

            # Step 3: If cache is empty, try live GCP data and update cache
            elif GCP_LIVE_AVAILABLE and gcp_live_tool and gcp_live_tool.storage_client:
                logger.info("🔴 Cache miss - fetching LIVE GCP data for storage buckets")
                try:
                    live_result = gcp_live_tool.execute("buckets", bucket_name=bucket_name, security_check=True)

                    if "error" not in live_result:
                        # Convert live data format to match expected structure
                        buckets = live_result.get("buckets", [])
                        converted_data = []

                        for bucket in buckets:
                            # Convert GCP live format to SQLite-like format
                            converted_bucket = {
                                "id": bucket.get("name", ""),  # Use name as ID
                                "name": bucket.get("name", ""),
                                "location": bucket.get("location", ""),
                                "storage_class": bucket.get("storage_class", ""),
                                "created": bucket.get("created", ""),
                                "public_access": _determine_public_access(bucket),
                                "encryption": "Google-managed" if not bucket.get("encryption") else "Customer-managed",
                                "versioning": 1 if bucket.get("versioning_enabled") else 0,
                                "lifecycle_rules": None,
                                "project_id": gcp_live_tool.project_id,
                                "created_at": bucket.get("created", "")
                            }
                            converted_data.append(converted_bucket)

                        result = {
                            "success": True,
                            "data": converted_data,
                            "source": "live_gcp_api",
                            "project_id": gcp_live_tool.project_id,
                            "security_findings": live_result.get("security_issues", [])
                        }

                        # TODO: Optionally update cache with fresh data for next time
                        logger.info("💾 Fresh data retrieved from GCP (cache can be updated)")

                    else:
                        # Live data failed, return empty cache result
                        logger.warning("Live GCP data failed, returning empty cache result")
                        result = cache_result
                        result["source"] = "sqlite_cache_empty"

                except Exception as e:
                    logger.warning(f"Live GCP query failed: {e}, returning cache result")
                    result = cache_result
                    result["source"] = "sqlite_cache_fallback"
                    if not result["success"]:
                        result["error"] = f"Both live GCP and cache failed: {e}"

            # Step 4: If both cache and live GCP failed, try Google Search fallback
            elif SEARCH_AVAILABLE and google_search_tool and (not cache_result["success"] or not cache_result.get("data")):
                logger.info("🔍 Cache and live GCP unavailable - trying Google Search fallback")
                search_query = f"Google Cloud Storage security best practices bucket configuration"
                if bucket_name:
                    search_query = f"Google Cloud Storage bucket security {bucket_name} configuration best practices"

                try:
                    search_result = google_search_tool.execute(search_query, search_type="gcp_docs", num_results=3)

                    if "error" not in search_result:
                        # Format search results as informational response
                        search_data = {
                            "search_query": search_query,
                            "documentation_results": search_result.get("results", []),
                            "message": "No cached or live bucket data available. Here are Google Cloud Storage security resources:",
                            "recommendations": [
                                "Enable uniform bucket-level access",
                                "Set up public access prevention",
                                "Use customer-managed encryption keys (CMEK)",
                                "Enable versioning for data protection",
                                "Configure lifecycle policies"
                            ]
                        }

                        result = {
                            "success": True,
                            "data": [search_data],  # Wrap in list for consistency
                            "source": "google_search_fallback",
                            "search_query": search_query
                        }
                        logger.info("🔍 Google Search fallback provided documentation results")
                    else:
                        # All fallbacks failed
                        result = cache_result
                        result["source"] = "all_sources_failed"
                        result["error"] = "Cache, live GCP, and search all failed"

                except Exception as e:
                    logger.warning(f"Google Search fallback failed: {e}")
                    result = cache_result
                    result["source"] = "search_fallback_failed"
                    result["error"] = f"All data sources failed: {e}"

            # Step 5: Truly no data sources available
            else:
                logger.info("📁 Using empty SQLite cache (no fallback options available)")
                result = cache_result
                result["source"] = "sqlite_cache_only"
                if not result["success"]:
                    result["error"] = "Failed to query storage buckets: " + result.get("error", "No data sources available")

        elif query_type == "api_keys":
            result = sqlite_tool_instance.execute_query("SELECT * FROM api_keys") # Assuming an api_keys table
            if not result["success"]:
                result["error"] = "Failed to query API keys: " + result.get("error", "Unknown error")

        elif query_type == "recommendations":
            result = sqlite_tool_instance.execute_query("SELECT * FROM recommendations") # Assuming a recommendations table
            if not result["success"]:
                result["error"] = "Failed to query recommendations: " + result.get("error", "Unknown error")

        elif query_type == "org_policies":
            result = sqlite_tool_instance.execute_query("SELECT * FROM org_policies") # Assuming an org_policies table
            if not result["success"]:
                result["error"] = "Failed to query org policies: " + result.get("error", "Unknown error")

        elif query_type == "service_usage":
            result = sqlite_tool_instance.execute_query("SELECT * FROM service_usage") # Assuming a service_usage table
            if not result["success"]:
                result["error"] = "Failed to query service usage: " + result.get("error", "Unknown error")

        elif query_type == "monitoring":
            result = sqlite_tool_instance.execute_query("SELECT * FROM monitoring_config") # Assuming a monitoring_config table
            if not result["success"]:
                result["error"] = "Failed to query monitoring config: " + result.get("error", "Unknown error")

        elif query_type == "logs":
            result = sqlite_tool_instance.execute_query("SELECT * FROM audit_logs_summary") # Assuming an audit_logs_summary table
            if not result["success"]:
                result["error"] = "Failed to query logs summary: " + result.get("error", "Unknown error")

        elif query_type == "firewall_rules":
            rule_name = params.get("rule_name")
            sql_query = "SELECT * FROM firewall_rules" # Assuming a firewall_rules table
            sql_params = []
            if rule_name:
                sql_query += " WHERE rule_name = ?"
                sql_params.append(rule_name)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query firewall rules: " + result.get("error", "Unknown error")

        elif query_type == "networks":
            result = sqlite_tool_instance.execute_query("SELECT * FROM vpc_networks") # Assuming a vpc_networks table
            if not result["success"]:
                result["error"] = "Failed to query networks: " + result.get("error", "Unknown error")

        elif query_type == "compute_instances":
            instance_name = params.get("instance_name")
            sql_query = "SELECT * FROM compute_instances" # Assuming a compute_instances table
            sql_params = []
            if instance_name:
                sql_query += " WHERE instance_name = ?"
                sql_params.append(instance_name)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query compute instances: " + result.get("error", "Unknown error")

        elif query_type == "gke_clusters":
            cluster_name = params.get("cluster_name")
            location = params.get("location")
            status = params.get("status")
            
            sql_query = "SELECT * FROM gke_clusters" # Assuming a gke_clusters table
            conditions = []
            sql_params = []

            if cluster_name:
                conditions.append("cluster_name = ?")
                sql_params.append(cluster_name)
            if location:
                conditions.append("location = ?")
                sql_params.append(location)
            if status:
                conditions.append("status = ?")
                sql_params.append(status)
            
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query GKE clusters: " + result.get("error", "Unknown error")

        elif query_type == "databases":
            result = sqlite_tool_instance.execute_query("SELECT * FROM databases") # Assuming a databases table
            if not result["success"]:
                result["error"] = "Failed to query databases: " + result.get("error", "Unknown error")

        elif query_type == "iam_accounts":
            email = params.get("email")
            sql_query = "SELECT * FROM iam_accounts" # Assuming an iam_accounts table
            sql_params = []
            if email:
                sql_query += " WHERE email = ?"
                sql_params.append(email)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query IAM accounts: " + result.get("error", "Unknown error")

        elif query_type == "secrets":
            secret_name = params.get("secret_name")
            sql_query = "SELECT * FROM secrets" # Assuming a secrets table
            sql_params = []
            if secret_name:
                sql_query += " WHERE secret_name = ?"
                sql_params.append(secret_name)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query secrets: " + result.get("error", "Unknown error")

        elif query_type == "cache_status":
            # This would require querying a metadata table about cache updates
            result = sqlite_tool_instance.execute_query("SELECT * FROM cache_metadata") # Assuming a cache_metadata table
            if not result["success"]:
                result["error"] = "Failed to get cache status: " + result.get("error", "Unknown error")

        elif query_type == "statistics": # Added handler for 'statistics'
            result = sqlite_tool_instance.get_summary_stats()
            if not result["success"]:
                result["error"] = "Failed to get statistics: " + result.get("error", "Unknown error")

        elif query_type == "msa_analysis":
            result = sqlite_tool_instance.execute_query("SELECT * FROM msa_analysis_history") # Assuming msa_analysis_history table
            if not result["success"]:
                result["error"] = "Failed to query MSA analysis history: " + result.get("error", "Unknown error")

        elif query_type == "msa_changes":
            service = params.get("service")
            sql_query = "SELECT * FROM msa_changes" # Assuming msa_changes table
            sql_params = []
            if service:
                sql_query += " WHERE service = ?"
                sql_params.append(service)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query MSA changes: " + result.get("error", "Unknown error")

        elif query_type == "org_policy_test":
            constraint = params.get("constraint")
            test_mode = params.get("test_mode", False)
            sql_query = "SELECT * FROM org_policy_tests" # Assuming org_policy_tests table
            sql_params = []
            conditions = []
            if constraint:
                conditions.append("constraint_name = ?")
                sql_params.append(constraint)
            if test_mode: # Assuming test_mode is stored as a boolean or integer
                conditions.append("test_mode = ?")
                sql_params.append(1 if test_mode else 0)
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query org policy tests: " + result.get("error", "Unknown error")

        elif query_type == "vpc_error_analysis":
            severity = params.get("severity")
            pattern = params.get("pattern")
            sql_query = "SELECT * FROM vpc_flow_log_errors" # Assuming vpc_flow_log_errors table
            sql_params = []
            conditions = []
            if severity:
                conditions.append("severity = ?")
                sql_params.append(severity)
            if pattern:
                conditions.append("error_pattern LIKE ?")
                sql_params.append(f"%{pattern}%")
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query VPC error analysis: " + result.get("error", "Unknown error")

        elif query_type == "support_tickets":
            priority = params.get("priority")
            status = params.get("status")
            sql_query = "SELECT * FROM support_tickets" # Assuming support_tickets table
            sql_params = []
            conditions = []
            if priority:
                conditions.append("priority = ?")
                sql_params.append(priority)
            if status:
                conditions.append("status = ?")
                sql_params.append(status)
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query support tickets: " + result.get("error", "Unknown error")

        elif query_type == "vpcsc_dry_run":
            perimeter = params.get("perimeter")
            severity = params.get("severity")
            sql_query = "SELECT * FROM vpcsc_dry_run_violations" # Assuming vpcsc_dry_run_violations table
            sql_params = []
            conditions = []
            if perimeter:
                conditions.append("perimeter_name = ?")
                sql_params.append(perimeter)
            if severity:
                conditions.append("severity = ?")
                sql_params.append(severity)
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query VPCSC dry run violations: " + result.get("error", "Unknown error")

        elif query_type == "vpcsc_readiness":
            result = sqlite_tool_instance.execute_query("SELECT * FROM vpcsc_readiness_report") # Assuming vpcsc_readiness_report table
            if not result["success"]:
                result["error"] = "Failed to query VPCSC readiness report: " + result.get("error", "Unknown error")

        elif query_type == "asset_inventory":
            category = params.get("category")
            importance = params.get("importance")
            environment = params.get("environment")
            public_only = params.get("public_only", False)

            sql_query = "SELECT * FROM asset_inventory" # Assuming asset_inventory table
            conditions = []
            sql_params = []

            if category:
                conditions.append("category = ?")
                sql_params.append(category)
            if importance:
                conditions.append("importance = ?")
                sql_params.append(importance)
            if environment:
                conditions.append("environment = ?")
                sql_params.append(environment)
            if public_only:
                conditions.append("is_public = ?")
                sql_params.append(1) # Assuming boolean is stored as 1/0
            
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query asset inventory: " + result.get("error", "Unknown error")

        elif query_type == "configuration_drift":
            result = sqlite_tool_instance.execute_query("SELECT * FROM configuration_drift") # Assuming configuration_drift table
            if not result["success"]:
                result["error"] = "Failed to query configuration drift: " + result.get("error", "Unknown error")

        elif query_type == "asset_report":
            report_type = params.get("report_type")
            export_format = params.get("export_format")
            sql_query = "SELECT * FROM asset_reports" # Assuming asset_reports table
            sql_params = []
            conditions = []
            if report_type:
                conditions.append("report_type = ?")
                sql_params.append(report_type)
            if export_format:
                conditions.append("export_format = ?")
                sql_params.append(export_format)
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query asset reports: " + result.get("error", "Unknown error")

        elif query_type == "msa_impact":
            project_id = params.get("project_id")
            sql_query = "SELECT * FROM msa_impacts" # Assuming msa_impacts table
            sql_params = []
            if project_id:
                sql_query += " WHERE project_id = ?"
                sql_params.append(project_id)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query MSA impacts: " + result.get("error", "Unknown error")

        elif query_type == "msa_permissions":
            permission = params.get("permission")
            sql_query = "SELECT * FROM msa_permission_changes" # Assuming msa_permission_changes table
            sql_params = []
            if permission:
                sql_query += " WHERE permission = ?"
                sql_params.append(permission)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query MSA permission changes: " + result.get("error", "Unknown error")

        elif query_type == "context_aware_analysis":
            focus = params.get("focus")
            timeframe = params.get("timeframe")
            sql_query = "SELECT * FROM context_aware_analysis" # Assuming context_aware_analysis table
            sql_params = []
            conditions = []
            if focus:
                conditions.append("focus_area = ?")
                sql_params.append(focus)
            if timeframe:
                conditions.append("timeframe = ?")
                sql_params.append(timeframe)
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query context aware analysis: " + result.get("error", "Unknown error")

        elif query_type == "cross_impact_analysis":
            domain = params.get("domain")
            depth = params.get("depth")
            sql_query = "SELECT * FROM cross_impact_analysis" # Assuming cross_impact_analysis table
            sql_params = []
            conditions = []
            if domain:
                conditions.append("impact_domain = ?")
                sql_params.append(domain)
            if depth:
                conditions.append("analysis_depth = ?")
                sql_params.append(depth)
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query cross impact analysis: " + result.get("error", "Unknown error")

        elif query_type == "custom":
            sql = params.get("sql")
            if sql:
                result = sqlite_tool_instance.execute_query(sql)
            else:
                result = {"success": False, "error": "Custom query requires 'sql' parameter."}
            if not result["success"]:
                result["error"] = "Failed to execute custom query: " + result.get("error", "Unknown error")

        # Knowledge Base Queries
        elif query_type == "coding_standards":
            search = params.get("search")
            language = params.get("language")
            sql_query = "SELECT * FROM knowledge_base WHERE type = 'coding_standards'"
            sql_params = []
            conditions = []
            if search:
                conditions.append("content LIKE ?")
                sql_params.append(f"%{search}%")
            if language:
                conditions.append("language = ?")
                sql_params.append(language)
            if conditions:
                sql_query += " AND " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query coding standards: " + result.get("error", "Unknown error")

        elif query_type == "enterprise_policies":
            severity = params.get("severity")
            sql_query = "SELECT * FROM knowledge_base WHERE type = 'enterprise_policies'"
            sql_params = []
            conditions = []
            if severity:
                conditions.append("severity = ?")
                sql_params.append(severity)
            if conditions:
                sql_query += " AND " + " AND ".join(conditions)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query enterprise policies: " + result.get("error", "Unknown error")

        elif query_type == "best_practices":
            result = sqlite_tool_instance.execute_query("SELECT * FROM knowledge_base WHERE type = 'best_practices'")
            if not result["success"]:
                result["error"] = "Failed to query best practices: " + result.get("error", "Unknown error")

        elif query_type == "compliance":
            result = sqlite_tool_instance.execute_query("SELECT * FROM knowledge_base WHERE type = 'compliance'")
            if not result["success"]:
                result["error"] = "Failed to query compliance: " + result.get("error", "Unknown error")

        elif query_type == "search_docs":
            # Direct Google Search for security documentation and best practices
            if not SEARCH_AVAILABLE or not google_search_tool:
                result = {"success": False, "error": "Google Search functionality not available"}
            else:
                query = params.get("query", "")
                search_type = params.get("search_type", "security")  # security, gcp_docs, vulnerability, general
                num_results = params.get("num_results", 5)

                if not query:
                    result = {"success": False, "error": "Search query parameter required"}
                else:
                    try:
                        logger.info(f"🔍 Executing direct Google Search: {query}")
                        search_result = google_search_tool.execute(query, search_type=search_type, num_results=num_results)

                        if "error" not in search_result:
                            # Format as security data result
                            result = {
                                "success": True,
                                "data": search_result.get("results", []),
                                "source": "google_search_direct",
                                "search_query": search_result.get("query", query),
                                "search_type": search_type,
                                "count": search_result.get("count", 0),
                                "message": f"Found {search_result.get('count', 0)} documentation results for: {query}"
                            }
                            logger.info(f"🔍 Google Search returned {search_result.get('count', 0)} results")
                        else:
                            result = {"success": False, "error": f"Search failed: {search_result['error']}"}

                    except Exception as e:
                        logger.error(f"Google Search error: {e}")
                        result = {"success": False, "error": f"Search execution failed: {str(e)}"}

        else:
            result = {"success": False, "error": f"Unknown query_type: {query_type}"}

    except Exception as e:
        logger.error(f"Error in query_security_data for query_type {query_type}: {e}")
        result = {"success": False, "error": f"An unexpected error occurred: {str(e)}"}
    
    return result

# Export for use
__all__ = ['sqlite_tool', 'SQLiteTool', 'sqlite_tool_instance', 'get_tables_tool', 'get_summary_tool', 'query_security_data']
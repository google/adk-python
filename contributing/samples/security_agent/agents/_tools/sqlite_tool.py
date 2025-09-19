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



def _evaluate_service_security_risks(service_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate security risks for adopting a new GCP service.
    This function provides security assessment rather than news reporting.
    """
    logger.info(f"🔐 Evaluating security risks for service: {service_name}")

    # Normalize service name for matching
    service_key = service_name.lower().replace(" ", "_").replace("-", "_")

    # Service security requirements database
    service_security_database = {
        "cloud_functions": {
            "service_display_name": "Cloud Functions",
            "category": "Serverless Compute",
            "required_roles": [
                "roles/cloudfunctions.admin",
                "roles/cloudfunctions.developer",
                "roles/storage.objectViewer",
                "roles/logging.logWriter"
            ],
            "required_apis": [
                "cloudfunctions.googleapis.com",
                "cloudbuild.googleapis.com",
                "storage.googleapis.com"
            ],
            "security_risks": [
                {
                    "risk": "Code Injection",
                    "severity": "HIGH",
                    "description": "Functions execute user-provided code with potential for injection attacks",
                    "mitigation": "Implement input validation, use runtime sandboxing, audit function code"
                },
                {
                    "risk": "Overprivileged Functions",
                    "severity": "HIGH",
                    "description": "Functions may inherit excessive IAM permissions",
                    "mitigation": "Apply principle of least privilege, use service-specific roles"
                },
                {
                    "risk": "Event Source Vulnerabilities",
                    "severity": "MEDIUM",
                    "description": "Malicious events could trigger unintended function execution",
                    "mitigation": "Validate event sources, implement event filtering"
                }
            ],
            "compliance_impact": [
                "Functions process data - ensure data residency compliance",
                "Audit logging required for function invocations",
                "Consider data encryption for sensitive workloads"
            ],
            "network_requirements": [
                "VPC connectivity for internal resource access",
                "Firewall rules for function triggers",
                "Consider Private Google Access for security"
            ]
        },
        "cloud_run": {
            "service_display_name": "Cloud Run",
            "category": "Serverless Containers",
            "required_roles": [
                "roles/run.admin",
                "roles/run.developer",
                "roles/storage.objectViewer",
                "roles/artifactregistry.reader"
            ],
            "required_apis": [
                "run.googleapis.com",
                "cloudbuild.googleapis.com",
                "artifactregistry.googleapis.com"
            ],
            "security_risks": [
                {
                    "risk": "Container Vulnerabilities",
                    "severity": "HIGH",
                    "description": "Base images and dependencies may contain security vulnerabilities",
                    "mitigation": "Regular image scanning, minimal base images, dependency updates"
                },
                {
                    "risk": "Public Internet Exposure",
                    "severity": "MEDIUM",
                    "description": "Services exposed to internet by default",
                    "mitigation": "Use IAM for authentication, implement VPC ingress controls"
                },
                {
                    "risk": "Secrets Management",
                    "severity": "MEDIUM",
                    "description": "Application secrets may be hardcoded or insecurely stored",
                    "mitigation": "Use Secret Manager, environment variables with proper access controls"
                }
            ],
            "compliance_impact": [
                "Container images must be scanned for vulnerabilities",
                "Runtime monitoring required for compliance",
                "Data processing location controls needed"
            ],
            "network_requirements": [
                "VPC connector for internal access",
                "Ingress control configuration",
                "Load balancer security policies"
            ]
        },
        "bigquery": {
            "service_display_name": "BigQuery",
            "category": "Data Analytics",
            "required_roles": [
                "roles/bigquery.admin",
                "roles/bigquery.dataEditor",
                "roles/bigquery.user",
                "roles/storage.objectViewer"
            ],
            "required_apis": [
                "bigquery.googleapis.com",
                "storage.googleapis.com"
            ],
            "security_risks": [
                {
                    "risk": "Data Exposure",
                    "severity": "CRITICAL",
                    "description": "Datasets may be accidentally made public or over-shared",
                    "mitigation": "Regular access audits, dataset-level permissions, data classification"
                },
                {
                    "risk": "Query Injection",
                    "severity": "HIGH",
                    "description": "Dynamic SQL queries vulnerable to injection attacks",
                    "mitigation": "Parameterized queries, input validation, query sanitization"
                },
                {
                    "risk": "Cost-Based DoS",
                    "severity": "MEDIUM",
                    "description": "Expensive queries could exhaust budget/quotas",
                    "mitigation": "Query quotas, slot reservations, query monitoring"
                }
            ],
            "compliance_impact": [
                "Data residency requirements for sensitive data",
                "Field-level encryption for PII/PHI data",
                "Audit logging for all data access operations"
            ],
            "network_requirements": [
                "Private Google Access for VPC-based access",
                "Authorized networks for external access",
                "VPC-native dataset access controls"
            ]
        },
        "gke": {
            "service_display_name": "Google Kubernetes Engine (GKE)",
            "category": "Container Orchestration",
            "required_roles": [
                "roles/container.admin",
                "roles/compute.instanceAdmin",
                "roles/servicemanagement.admin",
                "roles/storage.objectViewer"
            ],
            "required_apis": [
                "container.googleapis.com",
                "compute.googleapis.com",
                "monitoring.googleapis.com"
            ],
            "security_risks": [
                {
                    "risk": "Cluster Compromise",
                    "severity": "CRITICAL",
                    "description": "Kubernetes API server vulnerabilities could compromise entire cluster",
                    "mitigation": "Regular cluster updates, private clusters, authorized networks"
                },
                {
                    "risk": "Pod Security Issues",
                    "severity": "HIGH",
                    "description": "Privileged pods or containers with excessive permissions",
                    "mitigation": "Pod Security Standards, security contexts, admission controllers"
                },
                {
                    "risk": "Network Policies",
                    "severity": "MEDIUM",
                    "description": "Default allow-all pod-to-pod communication",
                    "mitigation": "Implement network policies, service mesh security"
                }
            ],
            "compliance_impact": [
                "Node OS hardening requirements",
                "Workload identity for pod authentication",
                "Binary authorization for container images"
            ],
            "network_requirements": [
                "Private cluster configuration",
                "Authorized networks for API access",
                "Firewall rules for node communication"
            ]
        }
    }

    # Get service configuration or create generic assessment
    service_config = service_security_database.get(service_key, {
        "service_display_name": service_name.title(),
        "category": "Unknown Service",
        "required_roles": ["Service-specific roles needed - requires analysis"],
        "required_apis": ["Service API requirements unknown"],
        "security_risks": [{
            "risk": "Unknown Security Profile",
            "severity": "MEDIUM",
            "description": f"Security assessment for {service_name} requires detailed analysis",
            "mitigation": "Conduct thorough security review before adoption"
        }],
        "compliance_impact": ["Compliance requirements need assessment"],
        "network_requirements": ["Network security requirements need evaluation"]
    })

    # Calculate overall risk score
    risk_score = _calculate_service_risk_score(service_config)

    # Get current environment context (this would ideally query current IAM state)
    current_environment = {
        "existing_roles": ["Basic compute and storage roles"],
        "security_gaps": ["Detailed current state analysis needed"],
        "readiness_score": 60  # Placeholder - would be calculated from actual current state
    }

    # Generate recommendations
    recommendations = _generate_service_adoption_recommendations(service_config, current_environment)

    return {
        "service_name": service_config["service_display_name"],
        "category": service_config["category"],
        "overall_risk_score": risk_score,
        "risk_level": _get_risk_level(risk_score),
        "required_permissions": service_config["required_roles"],
        "required_apis": service_config["required_apis"],
        "security_risks": service_config["security_risks"],
        "compliance_impact": service_config["compliance_impact"],
        "network_requirements": service_config["network_requirements"],
        "current_environment": current_environment,
        "recommendations": recommendations,
        "adoption_readiness": _assess_adoption_readiness(service_config, current_environment)
    }

def _calculate_service_risk_score(service_config: Dict[str, Any]) -> int:
    """Calculate overall risk score for service adoption (0-100, higher = more risk)"""
    base_score = 30  # Base risk for any new service

    # Add risk based on security risks
    risks = service_config.get("security_risks", [])
    for risk in risks:
        severity = risk.get("severity", "LOW")
        if severity == "CRITICAL":
            base_score += 25
        elif severity == "HIGH":
            base_score += 15
        elif severity == "MEDIUM":
            base_score += 8
        elif severity == "LOW":
            base_score += 3

    # Reduce risk based on category maturity
    category = service_config.get("category", "")
    if "Compute" in category:
        base_score -= 5  # Mature compute services
    elif "Analytics" in category:
        base_score += 5  # Data services have higher compliance risk

    return min(100, max(0, base_score))

def _get_risk_level(score: int) -> str:
    """Convert risk score to risk level"""
    if score >= 80:
        return "CRITICAL"
    elif score >= 60:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"

def _generate_service_adoption_recommendations(service_config: Dict[str, Any], current_env: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations for service adoption"""
    recommendations = []

    # IAM recommendations
    required_roles = service_config.get("required_roles", [])
    recommendations.append(f"📋 IAM Setup: Grant the following roles to appropriate principals: {', '.join(required_roles[:3])}")

    # API enablement
    apis = service_config.get("required_apis", [])
    recommendations.append(f"🔌 Enable APIs: {', '.join(apis[:2])}")

    # Security recommendations based on risks
    risks = service_config.get("security_risks", [])
    high_priority_risks = [r for r in risks if r.get("severity") in ["CRITICAL", "HIGH"]]
    for risk in high_priority_risks[:2]:  # Top 2 critical risks
        recommendations.append(f"⚠️ Address {risk['risk']}: {risk['mitigation']}")

    # Compliance recommendations
    compliance = service_config.get("compliance_impact", [])
    if compliance:
        recommendations.append(f"📊 Compliance: {compliance[0]}")

    # Network security
    network = service_config.get("network_requirements", [])
    if network:
        recommendations.append(f"🔒 Network: {network[0]}")

    return recommendations

def _assess_adoption_readiness(service_config: Dict[str, Any], current_env: Dict[str, Any]) -> Dict[str, Any]:
    """Assess readiness for service adoption"""
    readiness_score = current_env.get("readiness_score", 50)

    # Determine readiness status
    if readiness_score >= 80:
        status = "READY"
        message = "Environment is ready for service adoption"
    elif readiness_score >= 60:
        status = "MOSTLY_READY"
        message = "Minor configuration changes needed"
    elif readiness_score >= 40:
        status = "PARTIALLY_READY"
        message = "Significant security configuration required"
    else:
        status = "NOT_READY"
        message = "Major security improvements needed before adoption"

    return {
        "status": status,
        "score": readiness_score,
        "message": message,
        "blocking_issues": [] if readiness_score >= 60 else ["Security assessment required", "IAM configuration needed"]
    }

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

    Data Protection Strategy:
    - **Cache-first approach**: Preserves synthetic/demo data for proof-of-concept
    - **Explicit updates only**: Live GCP data fetched only when force_live_update=True
    - **Graceful fallback**: Falls back to cached data if live sources fail
    - **No automatic overwrites**: Synthetic data stays intact unless explicitly replaced

    Args:
        query_type: The type of security data to query (e.g., "security_summary", "storage_buckets").
        **kwargs: Additional parameters for the query:
            - severity, limit, category: Standard query filters
            - force_live_update: Boolean to force fetching live GCP data (default: False)
            - bucket_name, instance_name, etc.: Resource-specific filters

    Returns:
        A dictionary containing the query results with 'source' indicating data origin:
        - 'sqlite_cache_hit': Using cached/synthetic data (preserves demo data)
        - 'live_gcp_api': Fresh data from GCP APIs (when force_live_update=True)
        - 'sqlite_cache_fallback': Cached data used due to live source failure
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
            # Use the existing iam_accounts table instead of non-existent iam_policies
            principal = params.get("principal")

            # First, try to get comprehensive IAM data by joining multiple tables
            sql_query = """
            SELECT
                ia.email as account_email,
                ia.account_type,
                ia.created_date,
                ia.last_activity,
                ia.status,
                sf.finding_type as security_issue,
                sf.severity as issue_severity,
                sf.description as issue_description
            FROM iam_accounts ia
            LEFT JOIN security_findings sf ON sf.resource_name LIKE '%' || ia.email || '%'
            """
            sql_params = []

            if principal:
                sql_query += " WHERE ia.email = ? OR ia.email LIKE ?"
                sql_params.extend([principal, f"%{principal}%"])

            # Order by potential security issues first
            sql_query += " ORDER BY sf.severity DESC, ia.account_type, ia.email"

            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))

            # If the join query fails, fallback to simple iam_accounts query
            if not result["success"]:
                logger.warning("IAM analysis join query failed, falling back to basic iam_accounts query")
                sql_query = "SELECT * FROM iam_accounts"
                sql_params = []
                if principal:
                    sql_query += " WHERE email = ? OR email LIKE ?"
                    sql_params.extend([principal, f"%{principal}%"])

                result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))

            if not result["success"]:
                result["error"] = "Failed to query IAM analysis: " + result.get("error", "Unknown error")

        elif query_type == "storage_buckets":
            # Cache-first strategy: Preserve synthetic/demo data unless explicitly updating
            bucket_name = params.get("bucket_name")
            force_live_update = params.get("force_live_update", False)  # New parameter for explicit updates

            # Step 1: Check SQLite cache first (preserves synthetic data)
            sql_query = "SELECT * FROM storage_buckets"
            sql_params = []
            if bucket_name:
                sql_query += " WHERE name = ?"
                sql_params.append(bucket_name)

            cache_result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))

            # Step 2: If cache has data and not forcing live update, use cached data (preserves synthetic data)
            if cache_result["success"] and cache_result.get("data") and not force_live_update:
                logger.info("⚡ Checking SQLite cache for storage buckets")
                logger.info("📁 Using SQLite cached data for storage buckets (cache hit)")
                result = cache_result
                result["source"] = "sqlite_cache_hit"
                result["message"] = f"Using cached data ({len(result['data'])} buckets). Use force_live_update=True to fetch fresh data."

            # Step 3: Only try live GCP data if cache is empty OR force_live_update is True
            elif (not cache_result["success"] or not cache_result.get("data") or force_live_update) and GCP_LIVE_AVAILABLE and gcp_live_tool and gcp_live_tool.storage_client:
                logger.info("⚡ Checking SQLite cache for storage buckets")
                logger.info("🔴 Cache miss - fetching LIVE GCP data for storage buckets" if not force_live_update else "🔄 Force update - fetching LIVE GCP data for storage buckets")
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
                        logger.info("🔴 Using LIVE GCP data for storage buckets")

                        # Update cache with fresh data only when explicitly requested or cache was empty
                        if force_live_update or not cache_result.get("data"):
                            logger.info("💾 Updating SQLite cache with fresh GCP data")
                            # TODO: Implement cache update logic here if needed
                        else:
                            logger.info("💾 Fresh data retrieved from GCP (cache preserved)")

                    else:
                        # Live data failed, return cached result or empty
                        logger.warning("Live GCP data failed, returning cached result")
                        result = cache_result if cache_result["success"] else {"success": False, "error": "Both live GCP and cache failed"}
                        result["source"] = "sqlite_cache_fallback"

                except Exception as e:
                    logger.warning(f"Live GCP query failed: {e}, returning cached result")
                    result = cache_result if cache_result["success"] else {"success": False, "error": f"Both live GCP and cache failed: {e}"}
                    result["source"] = "sqlite_cache_fallback"

            # Step 4: Cache had no data and no live source available
            else:
                logger.info("📁 Using SQLite cached data for storage buckets")
                result = cache_result
                result["source"] = "sqlite_cache_only"
                if not result["success"]:
                    result["error"] = "Failed to query storage buckets: " + result.get("error", "No data sources available")


        elif query_type == "api_keys":
            # No api_keys table exists - fallback to security findings related to API keys
            result = sqlite_tool_instance.execute_query(
                "SELECT * FROM security_findings WHERE description LIKE '%API%' OR description LIKE '%key%' OR name LIKE '%api%'"
            )
            if not result["success"]:
                result["error"] = "Failed to query API-related security findings: " + result.get("error", "Unknown error")
            else:
                result["message"] = "No dedicated API keys table found. Showing API-related security findings instead."

        elif query_type == "recommendations":
            # No recommendations table exists - derive from security findings
            result = sqlite_tool_instance.execute_query(
                "SELECT name, description, recommendation, severity FROM security_findings WHERE recommendation IS NOT NULL AND recommendation != ''"
            )
            if not result["success"]:
                result["error"] = "Failed to query security recommendations: " + result.get("error", "Unknown error")
            else:
                result["message"] = "Security recommendations derived from security findings table."

        elif query_type == "org_policies":
            # No org_policies table exists - use security findings for policy-related issues
            result = sqlite_tool_instance.execute_query(
                "SELECT * FROM security_findings WHERE category LIKE '%policy%' OR description LIKE '%policy%' OR name LIKE '%policy%'"
            )
            if not result["success"]:
                result["error"] = "Failed to query policy-related security findings: " + result.get("error", "Unknown error")
            else:
                result["message"] = "No org_policies table found. Showing policy-related security findings instead."

        elif query_type == "service_usage":
            # No service_usage table exists - aggregate from available tables
            result = sqlite_tool_instance.execute_query(
                "SELECT 'storage' as service, COUNT(*) as count FROM storage_buckets UNION ALL " +
                "SELECT 'compute' as service, COUNT(*) as count FROM compute_instances UNION ALL " +
                "SELECT 'database' as service, COUNT(*) as count FROM databases UNION ALL " +
                "SELECT 'network' as service, COUNT(*) as count FROM networks"
            )
            if not result["success"]:
                result["error"] = "Failed to query aggregated service usage: " + result.get("error", "Unknown error")
            else:
                result["message"] = "Service usage aggregated from available resource tables."

        elif query_type == "monitoring":
            # No monitoring_config table exists - check for monitoring-related security findings
            result = sqlite_tool_instance.execute_query(
                "SELECT * FROM security_findings WHERE category LIKE '%monitor%' OR description LIKE '%monitor%' OR description LIKE '%log%'"
            )
            if not result["success"]:
                result["error"] = "Failed to query monitoring-related security findings: " + result.get("error", "Unknown error")
            else:
                result["message"] = "No monitoring_config table found. Showing monitoring-related security findings instead."

        elif query_type == "logs":
            # No audit_logs_summary table exists - show logging-related security findings
            result = sqlite_tool_instance.execute_query(
                "SELECT * FROM security_findings WHERE description LIKE '%log%' OR description LIKE '%audit%' OR category LIKE '%logging%'"
            )
            if not result["success"]:
                result["error"] = "Failed to query logging-related security findings: " + result.get("error", "Unknown error")
            else:
                result["message"] = "No audit_logs_summary table found. Showing logging-related security findings instead."

        elif query_type == "firewall_rules":
            # firewall_rules table exists - use correct column names
            rule_name = params.get("rule_name")
            sql_query = "SELECT * FROM firewall_rules"
            sql_params = []
            if rule_name:
                sql_query += " WHERE name = ?"  # Use 'name' instead of 'rule_name'
                sql_params.append(rule_name)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query firewall rules: " + result.get("error", "Unknown error")

        elif query_type == "networks":
            # Use the actual 'networks' table instead of 'vpc_networks'
            result = sqlite_tool_instance.execute_query("SELECT * FROM networks")
            if not result["success"]:
                result["error"] = "Failed to query networks: " + result.get("error", "Unknown error")

        elif query_type == "compute_instances":
            # compute_instances table exists - use correct column name
            instance_name = params.get("instance_name")
            sql_query = "SELECT * FROM compute_instances"
            sql_params = []
            if instance_name:
                sql_query += " WHERE name = ?"  # Use 'name' instead of 'instance_name'
                sql_params.append(instance_name)
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query compute instances: " + result.get("error", "Unknown error")

        elif query_type == "gke_clusters":
            # Fallback: Search for GKE/Kubernetes-related resources in compute_instances and security_findings since gke_clusters doesn't exist
            cluster_name = params.get("cluster_name")
            location = params.get("location")
            status = params.get("status")
            logger.info(f"☸️ Searching for GKE/Kubernetes resources (gke_clusters table not available)")

            # Search in compute_instances for GKE nodes and security_findings for GKE issues
            sql_query = """
            SELECT
                'compute_instance' as source_type,
                name as resource_name,
                machine_type,
                zone as location,
                status,
                creation_timestamp as created_date
            FROM compute_instances
            WHERE name LIKE '%gke%' OR name LIKE '%kubernetes%'
            UNION
            SELECT
                'security_finding' as source_type,
                resource_name,
                finding_type as machine_type,
                'N/A' as location,
                severity as status,
                created_date
            FROM security_findings
            WHERE description LIKE '%GKE%' OR description LIKE '%kubernetes%' OR resource_name LIKE '%gke%'
            """
            conditions = []
            sql_params = []

            if cluster_name:
                sql_query += " AND (resource_name LIKE ? OR name LIKE ?)"
                sql_params.extend([f"%{cluster_name}%", f"%{cluster_name}%"])
            if location:
                sql_query += " AND location LIKE ?"
                sql_params.append(f"%{location}%")
            if status:
                sql_query += " AND status LIKE ?"
                sql_params.append(f"%{status}%")

            sql_query += " ORDER BY created_at DESC"
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query GKE/Kubernetes resources: " + result.get("error", "Unknown error")

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
            # Fallback: Search for secret-related security findings since 'secrets' table doesn't exist
            secret_name = params.get("secret_name")
            logger.info(f"🔐 Searching for secret-related security findings (secrets table not available)")

            sql_query = """
            SELECT
                name,
                resource_name,
                severity,
                description,
                category,
                created_at
            FROM security_findings
            WHERE description LIKE '%secret%' OR description LIKE '%key%' OR description LIKE '%credential%'
            """
            sql_params = []

            if secret_name:
                sql_query += " AND (resource_name LIKE ? OR description LIKE ?)"
                sql_params.extend([f"%{secret_name}%", f"%{secret_name}%"])

            sql_query += " ORDER BY severity DESC, created_date DESC"
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query secret-related security findings: " + result.get("error", "Unknown error")

        elif query_type == "cache_status":
            # Fallback: Provide cache status from available table statistics since cache_metadata doesn't exist
            logger.info(f"📊 Getting cache status from available table counts")

            # Get summary statistics which includes table counts
            result = sqlite_tool_instance.get_summary_stats()
            if result["success"] and result.get("data"):
                # Transform the stats into cache status format
                stats = result["data"]
                cache_status = {
                    "cache_last_updated": "Available from table statistics",
                    "total_cached_records": sum(len(data) if isinstance(data, list) else 0 for data in stats.values() if data),
                    "available_tables": list(stats.keys()),
                    "table_counts": {k: len(v) if isinstance(v, list) else 0 for k, v in stats.items() if v}
                }
                result["data"] = cache_status
            else:
                result = {"success": False, "error": "Failed to get cache status from table statistics"}

        elif query_type == "statistics": # Added handler for 'statistics'
            result = sqlite_tool_instance.get_summary_stats()
            if not result["success"]:
                result["error"] = "Failed to get statistics: " + result.get("error", "Unknown error")

        elif query_type == "msa_analysis":
            # Fallback: Search for MSA-related security findings since msa_analysis_history doesn't exist
            logger.info(f"🔍 Searching for MSA-related security findings (msa_analysis_history table not available)")

            sql_query = """
            SELECT
                name,
                resource_name,
                severity,
                description,
                category,
                created_at
            FROM security_findings
            WHERE description LIKE '%MSA%' OR description LIKE '%service account%' OR name LIKE '%service%'
            ORDER BY severity DESC, created_at DESC
            """
            result = sqlite_tool_instance.execute_query(sql_query)
            if not result["success"]:
                result["error"] = "Failed to query MSA-related security findings: " + result.get("error", "Unknown error")

        elif query_type == "msa_changes":
            # Fallback: Search for service account changes in security findings since msa_changes doesn't exist
            service = params.get("service")
            logger.info(f"🔄 Searching for service account changes in security findings (msa_changes table not available)")

            sql_query = """
            SELECT
                name,
                resource_name,
                severity,
                description,
                category,
                created_at
            FROM security_findings
            WHERE (description LIKE '%change%' OR description LIKE '%modify%' OR description LIKE '%update%')
              AND (description LIKE '%service account%' OR description LIKE '%MSA%')
            """
            sql_params = []

            if service:
                sql_query += " AND (resource_name LIKE ? OR description LIKE ?)"
                sql_params.extend([f"%{service}%", f"%{service}%"])

            sql_query += " ORDER BY created_at DESC"
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query service account changes: " + result.get("error", "Unknown error")

        elif query_type == "org_policy_test":
            # Fallback: Search for organization policy related security findings since org_policy_tests doesn't exist
            constraint = params.get("constraint")
            test_mode = params.get("test_mode", False)
            logger.info(f"🏛️ Searching for organization policy findings (org_policy_tests table not available)")

            sql_query = """
            SELECT
                name,
                resource_name,
                severity,
                description,
                category,
                created_at
            FROM security_findings
            WHERE description LIKE '%policy%' OR description LIKE '%constraint%' OR name LIKE '%policy%'
            """
            sql_params = []

            if constraint:
                sql_query += " AND (resource_name LIKE ? OR description LIKE ?)"
                sql_params.extend([f"%{constraint}%", f"%{constraint}%"])

            sql_query += " ORDER BY severity DESC, created_date DESC"
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query organization policy findings: " + result.get("error", "Unknown error")

        elif query_type == "vpc_error_analysis":
            # Fallback: Search for VPC/network-related security findings since vpc_flow_log_errors doesn't exist
            severity = params.get("severity")
            pattern = params.get("pattern")
            logger.info(f"🌐 Searching for VPC/network error patterns in security findings (vpc_flow_log_errors table not available)")

            sql_query = """
            SELECT
                finding_type,
                resource_name,
                severity,
                description,
                source_type,
                created_date
            FROM security_findings
            WHERE (description LIKE '%VPC%' OR description LIKE '%network%' OR description LIKE '%error%')
               OR finding_type LIKE '%network%'
            """
            sql_params = []
            conditions = []

            if severity:
                conditions.append("severity = ?")
                sql_params.append(severity)
            if pattern:
                conditions.append("description LIKE ?")
                sql_params.append(f"%{pattern}%")

            if conditions:
                sql_query += " AND " + " AND ".join(conditions)

            sql_query += " ORDER BY severity DESC, created_date DESC"
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query VPC/network error analysis: " + result.get("error", "Unknown error")

        elif query_type == "support_tickets":
            # Fallback: Search for support-related security findings since support_tickets doesn't exist
            priority = params.get("priority")
            status = params.get("status")
            logger.info(f"🎫 Searching for support-related security findings (support_tickets table not available)")

            sql_query = """
            SELECT
                finding_type,
                resource_name,
                severity,
                description,
                source_type,
                created_date
            FROM security_findings
            WHERE description LIKE '%support%' OR description LIKE '%ticket%' OR description LIKE '%issue%'
            """
            sql_params = []
            conditions = []

            if priority:
                # Map priority to severity
                severity_map = {'high': 'HIGH', 'medium': 'MEDIUM', 'low': 'LOW'}
                mapped_severity = severity_map.get(priority.lower(), priority)
                conditions.append("severity = ?")
                sql_params.append(mapped_severity)
            if status:
                conditions.append("description LIKE ?")
                sql_params.append(f"%{status}%")

            if conditions:
                sql_query += " AND " + " AND ".join(conditions)

            sql_query += " ORDER BY severity DESC, created_date DESC"
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query support-related findings: " + result.get("error", "Unknown error")

        elif query_type == "vpcsc_dry_run":
            # Fallback: Search for VPC Service Controls related security findings since vpcsc_dry_run_violations doesn't exist
            perimeter = params.get("perimeter")
            severity = params.get("severity")
            logger.info(f"🛡️ Searching for VPC Service Controls findings (vpcsc_dry_run_violations table not available)")

            sql_query = """
            SELECT
                finding_type,
                resource_name,
                severity,
                description,
                source_type,
                created_date
            FROM security_findings
            WHERE description LIKE '%VPC Service Controls%' OR description LIKE '%perimeter%' OR description LIKE '%dry run%'
            """
            sql_params = []
            conditions = []

            if perimeter:
                conditions.append("(resource_name LIKE ? OR description LIKE ?)")
                sql_params.extend([f"%{perimeter}%", f"%{perimeter}%"])
            if severity:
                conditions.append("severity = ?")
                sql_params.append(severity)

            if conditions:
                sql_query += " AND " + " AND ".join(conditions)

            sql_query += " ORDER BY severity DESC, created_date DESC"
            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query VPC Service Controls findings: " + result.get("error", "Unknown error")

        elif query_type == "vpcsc_readiness":
            # Fallback: Search for VPC Service Controls readiness in security findings since vpcsc_readiness_report doesn't exist
            logger.info(f"📋 Searching for VPC Service Controls readiness findings (vpcsc_readiness_report table not available)")

            sql_query = """
            SELECT
                finding_type,
                resource_name,
                severity,
                description,
                source_type,
                created_date
            FROM security_findings
            WHERE description LIKE '%readiness%' OR description LIKE '%VPC Service Controls%' OR description LIKE '%compliance%'
            ORDER BY severity DESC, created_date DESC
            """
            result = sqlite_tool_instance.execute_query(sql_query)
            if not result["success"]:
                result["error"] = "Failed to query VPC Service Controls readiness findings: " + result.get("error", "Unknown error")

        elif query_type == "asset_inventory":
            # Fallback: Use existing 'assets' table which serves as the asset inventory
            category = params.get("category")
            importance = params.get("importance")
            environment = params.get("environment")
            public_only = params.get("public_only", False)
            logger.info(f"📦 Using assets table for asset inventory (asset_inventory table not available)")

            sql_query = "SELECT * FROM assets"
            conditions = []
            sql_params = []

            # Map parameters to available columns in assets table
            if category:
                conditions.append("(asset_type LIKE ? OR resource_type LIKE ?)")
                sql_params.extend([f"%{category}%", f"%{category}%"])
            if importance:
                # Map importance to existing fields
                conditions.append("(name LIKE ? OR resource_type LIKE ?)")
                sql_params.extend([f"%{importance}%", f"%{importance}%"])
            if environment:
                conditions.append("(location LIKE ? OR name LIKE ?)")
                sql_params.extend([f"%{environment}%", f"%{environment}%"])
            if public_only:
                conditions.append("(name LIKE '%public%' OR resource_type LIKE '%public%')")

            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)

            result = sqlite_tool_instance.execute_query(sql_query, tuple(sql_params))
            if not result["success"]:
                result["error"] = "Failed to query asset inventory from assets table: " + result.get("error", "Unknown error")

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

        elif query_type == "service_evaluation":
            # Service Adoption Security Risk Assessment
            service_name = params.get("service_name", "").lower().replace("_", " ").replace("-", " ")

            if not service_name:
                result = {"success": False, "error": "service_name parameter required for service evaluation"}
            else:
                # Get service security requirements and risk analysis
                evaluation_result = _evaluate_service_security_risks(service_name, params)
                result = {
                    "success": True,
                    "data": evaluation_result,
                    "message": f"Security risk assessment completed for {service_name}",
                    "source": "service_evaluation_engine"
                }

        else:
            result = {"success": False, "error": f"Unknown query_type: {query_type}"}

    except Exception as e:
        logger.error(f"Error in query_security_data for query_type {query_type}: {e}")
        result = {"success": False, "error": f"An unexpected error occurred: {str(e)}"}
    
    return result

# Export for use
__all__ = ['sqlite_tool', 'SQLiteTool', 'sqlite_tool_instance', 'get_tables_tool', 'get_summary_tool', 'query_security_data']
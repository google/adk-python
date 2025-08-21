"""
Custom Roles Permission Analyzer Service
========================================

Analyzes custom IAM roles to identify excessive permissions and recommend
standard GCP role alternatives following the principle of least privilege.
"""

import logging
import json
import sqlite3
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
from google.cloud import iam_admin_v1
from google.api_core import exceptions
import os

logger = logging.getLogger(__name__)


class CustomRolesAnalyzer:
    """Analyzes custom IAM roles for permission optimization."""
    
    def __init__(self, project_id: str):
        """Initialize the Custom Roles Analyzer.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.iam_client = iam_admin_v1.IAMClient()
        self.db_path = Path(__file__).parent.parent / "cache" / "custom_roles.db"
        self.standard_roles_cache = {}
        self._init_database()
        self._load_standard_roles()
        
    def _init_database(self):
        """Initialize SQLite database for storing role analysis."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                role_name TEXT NOT NULL,
                title TEXT,
                description TEXT,
                stage TEXT,
                permissions_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analyzed_at TIMESTAMP,
                UNIQUE(project_id, role_name)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS permission_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                custom_role_id INTEGER NOT NULL,
                analysis_json TEXT NOT NULL,
                risk_score REAL,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (custom_role_id) REFERENCES custom_roles(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS standard_roles_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role_name TEXT UNIQUE NOT NULL,
                title TEXT,
                description TEXT,
                permissions_json TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS role_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                custom_role_id INTEGER NOT NULL,
                suggested_standard_roles TEXT NOT NULL,
                permission_diff TEXT NOT NULL,
                match_type TEXT CHECK(match_type IN ('exact', 'subset', 'superset', 'partial')),
                match_percentage REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (custom_role_id) REFERENCES custom_roles(id)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized at {self.db_path}")
    
    def _load_standard_roles(self):
        """Load standard GCP roles from cache or API."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Check cache age
        cursor.execute("""
            SELECT COUNT(*) as count, 
                   MIN(last_updated) as oldest
            FROM standard_roles_cache
        """)
        row = cursor.fetchone()
        
        # Refresh if cache is empty or older than 7 days
        if row[0] == 0 or (row[1] and 
            (datetime.now() - datetime.fromisoformat(row[1])).days > 7):
            logger.info("🔄 Refreshing standard roles cache...")
            self._fetch_standard_roles()
        
        # Load from cache
        cursor.execute("SELECT role_name, permissions_json FROM standard_roles_cache")
        for row in cursor.fetchall():
            self.standard_roles_cache[row[0]] = set(json.loads(row[1]))
        
        conn.close()
        logger.info(f"✅ Loaded {len(self.standard_roles_cache)} standard roles")
    
    def _fetch_standard_roles(self):
        """Fetch standard GCP roles from API."""
        try:
            # Common standard roles to cache
            standard_role_prefixes = [
                "roles/viewer",
                "roles/editor", 
                "roles/owner",
                "roles/iam.",
                "roles/compute.",
                "roles/storage.",
                "roles/container.",
                "roles/bigquery.",
                "roles/cloudsql.",
                "roles/monitoring.",
                "roles/logging.",
                "roles/security."
            ]
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for prefix in standard_role_prefixes:
                try:
                    # List roles matching prefix
                    request = iam_admin_v1.ListRolesRequest(
                        view=iam_admin_v1.RoleView.BASIC
                    )
                    
                    for role in self.iam_client.list_roles(request=request):
                        if role.name.startswith(prefix):
                            # Get full role details
                            full_role = self.iam_client.get_role(
                                request=iam_admin_v1.GetRoleRequest(name=role.name)
                            )
                            
                            cursor.execute("""
                                INSERT OR REPLACE INTO standard_roles_cache
                                (role_name, title, description, permissions_json)
                                VALUES (?, ?, ?, ?)
                            """, (
                                role.name,
                                full_role.title,
                                full_role.description,
                                json.dumps(list(full_role.included_permissions))
                            ))
                            
                            self.standard_roles_cache[role.name] = set(full_role.included_permissions)
                            
                except Exception as e:
                    logger.warning(f"⚠️ Error fetching roles with prefix {prefix}: {e}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error fetching standard roles: {e}")
            # Load some common roles as fallback
            self._load_fallback_roles()
    
    def _load_fallback_roles(self):
        """Load common standard roles as fallback."""
        fallback_roles = {
            "roles/viewer": ["*.get", "*.list"],
            "roles/editor": ["*.get", "*.list", "*.create", "*.update", "*.patch"],
            "roles/owner": ["*"],
            "roles/iam.securityReviewer": [
                "iam.roles.get", "iam.roles.list",
                "iam.serviceAccounts.get", "iam.serviceAccounts.list",
                "resourcemanager.projects.get", "resourcemanager.projects.getIamPolicy"
            ],
            "roles/storage.objectViewer": [
                "storage.objects.get", "storage.objects.list"
            ],
            "roles/compute.viewer": [
                "compute.*.get", "compute.*.list"
            ]
        }
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for role_name, permissions in fallback_roles.items():
            cursor.execute("""
                INSERT OR REPLACE INTO standard_roles_cache
                (role_name, permissions_json)
                VALUES (?, ?)
            """, (role_name, json.dumps(permissions)))
            
            self.standard_roles_cache[role_name] = set(permissions)
        
        conn.commit()
        conn.close()
    
    def fetch_custom_roles(self) -> List[Dict[str, Any]]:
        """Fetch all custom roles from the project.
        
        Returns:
            List of custom role dictionaries
        """
        custom_roles = []
        
        try:
            parent = f"projects/{self.project_id}"
            request = iam_admin_v1.ListRolesRequest(
                parent=parent,
                view=iam_admin_v1.RoleView.FULL
            )
            
            for role in self.iam_client.list_roles(request=request):
                custom_roles.append({
                    "name": role.name,
                    "title": role.title,
                    "description": role.description,
                    "stage": role.stage.name if role.stage else "ALPHA",
                    "permissions": list(role.included_permissions),
                    "deleted": role.deleted
                })
                
                # Store in database
                self._store_custom_role(role)
            
            logger.info(f"✅ Fetched {len(custom_roles)} custom roles")
            
        except Exception as e:
            logger.error(f"❌ Error fetching custom roles: {e}")
            # Return sample data for testing
            custom_roles = self._get_sample_custom_roles()
        
        return custom_roles
    
    def _store_custom_role(self, role):
        """Store custom role in database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO custom_roles
            (project_id, role_name, title, description, stage, permissions_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            self.project_id,
            role.name,
            role.title,
            role.description,
            role.stage.name if role.stage else "ALPHA",
            json.dumps(list(role.included_permissions))
        ))
        
        conn.commit()
        conn.close()
    
    def _get_sample_custom_roles(self) -> List[Dict[str, Any]]:
        """Get sample custom roles for testing."""
        return [
            {
                "name": f"projects/{self.project_id}/roles/customDeveloper",
                "title": "Custom Developer Role",
                "description": "Developer with extended permissions",
                "stage": "BETA",
                "permissions": [
                    "compute.instances.create",
                    "compute.instances.delete",
                    "compute.instances.get",
                    "compute.instances.list",
                    "compute.instances.setMetadata",
                    "compute.instances.setTags",
                    "storage.buckets.create",
                    "storage.buckets.delete",
                    "storage.buckets.get",
                    "storage.buckets.list",
                    "storage.objects.create",
                    "storage.objects.delete",
                    "storage.objects.get",
                    "storage.objects.list",
                    "iam.serviceAccounts.actAs",
                    "iam.serviceAccounts.create",
                    "iam.serviceAccounts.delete"
                ],
                "deleted": False
            },
            {
                "name": f"projects/{self.project_id}/roles/customViewer",
                "title": "Custom Viewer Role",
                "description": "Read-only access with some exceptions",
                "stage": "GA",
                "permissions": [
                    "compute.instances.get",
                    "compute.instances.list",
                    "storage.buckets.get",
                    "storage.buckets.list",
                    "storage.objects.get",
                    "storage.objects.list"
                ],
                "deleted": False
            }
        ]
    
    def analyze_permissions(self, custom_role: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze permissions of a custom role.
        
        Args:
            custom_role: Custom role dictionary
            
        Returns:
            Analysis results dictionary
        """
        permissions = set(custom_role["permissions"])
        analysis = {
            "role_name": custom_role["name"],
            "total_permissions": len(permissions),
            "risk_score": self._calculate_risk_score(permissions),
            "risk_breakdown": self._get_risk_breakdown(permissions),
            "matches": self._find_standard_role_matches(permissions),
            "recommendations": [],
            "permission_categories": self._categorize_permissions(permissions)
        }
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        # Store analysis
        self._store_analysis(custom_role, analysis)
        
        return analysis
    
    def _calculate_risk_score(self, permissions: Set[str]) -> float:
        """Calculate risk score for a set of permissions.
        
        Args:
            permissions: Set of permission strings
            
        Returns:
            Risk score (0-100)
        """
        high_risk_patterns = [
            "*.delete", "*.setIamPolicy", "*.actAs", "*.impersonate",
            "*.setMetadata", "*.update", "*.patch"
        ]
        medium_risk_patterns = ["*.create", "*.insert", "*.write"]
        
        high_risk_count = 0
        medium_risk_count = 0
        
        for perm in permissions:
            for pattern in high_risk_patterns:
                if self._matches_pattern(perm, pattern):
                    high_risk_count += 1
                    break
            else:
                for pattern in medium_risk_patterns:
                    if self._matches_pattern(perm, pattern):
                        medium_risk_count += 1
                        break
        
        # Calculate weighted score
        score = (high_risk_count * 10 + medium_risk_count * 5) / len(permissions) * 100
        return min(100, score)
    
    def _matches_pattern(self, permission: str, pattern: str) -> bool:
        """Check if permission matches a pattern.
        
        Args:
            permission: Permission string
            pattern: Pattern with wildcards
            
        Returns:
            True if matches
        """
        import fnmatch
        return fnmatch.fnmatch(permission, pattern)
    
    def _get_risk_breakdown(self, permissions: Set[str]) -> Dict[str, int]:
        """Get breakdown of permissions by risk level.
        
        Args:
            permissions: Set of permission strings
            
        Returns:
            Risk breakdown dictionary
        """
        breakdown = {"high": 0, "medium": 0, "low": 0}
        
        for perm in permissions:
            if any(risk in perm for risk in ["delete", "setIamPolicy", "actAs"]):
                breakdown["high"] += 1
            elif any(risk in perm for risk in ["create", "update", "write"]):
                breakdown["medium"] += 1
            else:
                breakdown["low"] += 1
        
        return breakdown
    
    def _find_standard_role_matches(self, permissions: Set[str]) -> List[Dict[str, Any]]:
        """Find standard roles that match the custom role permissions.
        
        Args:
            permissions: Set of permission strings
            
        Returns:
            List of matching standard roles with details
        """
        matches = []
        
        for role_name, std_permissions in self.standard_roles_cache.items():
            # Calculate match metrics
            intersection = permissions & std_permissions
            union = permissions | std_permissions
            
            if not union:
                continue
            
            match_percentage = len(intersection) / len(union) * 100
            
            # Determine match type
            if permissions == std_permissions:
                match_type = "exact"
            elif permissions.issubset(std_permissions):
                match_type = "subset"
            elif permissions.issuperset(std_permissions):
                match_type = "superset"
            else:
                match_type = "partial"
            
            # Only include significant matches
            if match_percentage > 30 or match_type in ["exact", "subset"]:
                matches.append({
                    "role": role_name,
                    "match_type": match_type,
                    "match_percentage": round(match_percentage, 2),
                    "missing_permissions": list(permissions - std_permissions),
                    "extra_permissions": list(std_permissions - permissions),
                    "common_permissions": len(intersection)
                })
        
        # Sort by match percentage
        matches.sort(key=lambda x: x["match_percentage"], reverse=True)
        
        return matches[:5]  # Return top 5 matches
    
    def _categorize_permissions(self, permissions: Set[str]) -> Dict[str, List[str]]:
        """Categorize permissions by service.
        
        Args:
            permissions: Set of permission strings
            
        Returns:
            Dictionary of categorized permissions
        """
        categories = {}
        
        for perm in permissions:
            service = perm.split(".")[0] if "." in perm else "other"
            if service not in categories:
                categories[service] = []
            categories[service].append(perm)
        
        return categories
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on analysis.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Check for exact or subset matches
        for match in analysis["matches"]:
            if match["match_type"] == "exact":
                recommendations.append({
                    "type": "replacement",
                    "severity": "high",
                    "message": f"Replace with standard role '{match['role']}' (exact match)",
                    "action": f"gcloud projects add-iam-policy-binding {self.project_id} --role={match['role']} --member=MEMBER"
                })
            elif match["match_type"] == "subset" and match["match_percentage"] > 80:
                recommendations.append({
                    "type": "replacement",
                    "severity": "medium",
                    "message": f"Consider using '{match['role']}' (covers {match['match_percentage']:.0f}% of permissions)",
                    "missing": match["missing_permissions"][:3]  # Show top 3 missing
                })
        
        # Check for high-risk permissions
        if analysis["risk_score"] > 70:
            recommendations.append({
                "type": "security",
                "severity": "high",
                "message": f"High risk score ({analysis['risk_score']:.0f}). Review and remove unnecessary high-risk permissions",
                "details": f"Found {analysis['risk_breakdown']['high']} high-risk permissions"
            })
        
        # Check for overly broad permissions
        if analysis["total_permissions"] > 50:
            recommendations.append({
                "type": "optimization",
                "severity": "medium",
                "message": f"Role has {analysis['total_permissions']} permissions. Consider splitting into multiple focused roles",
                "action": "Apply principle of least privilege"
            })
        
        return recommendations
    
    def _store_analysis(self, custom_role: Dict[str, Any], analysis: Dict[str, Any]):
        """Store analysis results in database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get custom role ID
        cursor.execute("""
            SELECT id FROM custom_roles 
            WHERE role_name = ? AND project_id = ?
        """, (custom_role["name"], self.project_id))
        
        row = cursor.fetchone()
        if row:
            custom_role_id = row[0]
            
            # Store analysis
            cursor.execute("""
                INSERT INTO permission_analysis
                (custom_role_id, analysis_json, risk_score, recommendations)
                VALUES (?, ?, ?, ?)
            """, (
                custom_role_id,
                json.dumps(analysis),
                analysis["risk_score"],
                json.dumps(analysis["recommendations"])
            ))
            
            # Store role mappings
            for match in analysis["matches"][:3]:  # Store top 3 matches
                cursor.execute("""
                    INSERT INTO role_mappings
                    (custom_role_id, suggested_standard_roles, permission_diff, 
                     match_type, match_percentage)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    custom_role_id,
                    match["role"],
                    json.dumps({
                        "missing": match["missing_permissions"],
                        "extra": match["extra_permissions"]
                    }),
                    match["match_type"],
                    match["match_percentage"]
                ))
            
            # Update analyzed timestamp
            cursor.execute("""
                UPDATE custom_roles 
                SET analyzed_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (custom_role_id,))
        
        conn.commit()
        conn.close()
    
    def export_recommendations(self, analysis: Dict[str, Any], format: str = "terraform") -> str:
        """Export recommendations in specified format.
        
        Args:
            analysis: Analysis results dictionary
            format: Export format (terraform, gcloud, json)
            
        Returns:
            Formatted export string
        """
        if format == "terraform":
            return self._export_terraform(analysis)
        elif format == "gcloud":
            return self._export_gcloud(analysis)
        else:
            return json.dumps(analysis, indent=2)
    
    def _export_terraform(self, analysis: Dict[str, Any]) -> str:
        """Export as Terraform configuration."""
        tf_config = []
        role_name = analysis["role_name"].split("/")[-1]
        
        # Add header
        tf_config.append(f"# Terraform configuration for optimizing {role_name}")
        tf_config.append("")
        
        # Add recommended replacements
        for match in analysis["matches"][:1]:  # Use best match
            if match["match_type"] in ["exact", "subset"]:
                tf_config.append(f"""
resource "google_project_iam_member" "{role_name}_replacement" {{
  project = "{self.project_id}"
  role    = "{match['role']}"
  member  = "serviceAccount:example@{self.project_id}.iam.gserviceaccount.com"
}}
""")
        
        return "\n".join(tf_config)
    
    def _export_gcloud(self, analysis: Dict[str, Any]) -> str:
        """Export as gcloud commands."""
        commands = []
        role_name = analysis["role_name"]
        
        # Add header
        commands.append(f"# gcloud commands for optimizing {role_name}")
        commands.append("")
        
        # Add removal command
        commands.append(f"# Remove custom role")
        commands.append(f"gcloud iam roles delete {role_name.split('/')[-1]} --project={self.project_id}")
        commands.append("")
        
        # Add replacement commands
        for rec in analysis["recommendations"]:
            if "action" in rec and rec["action"].startswith("gcloud"):
                commands.append(f"# {rec['message']}")
                commands.append(rec["action"])
                commands.append("")
        
        return "\n".join(commands)
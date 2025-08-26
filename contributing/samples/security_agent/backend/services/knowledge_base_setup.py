"""
Enterprise Knowledge Base Database Setup
========================================

Creates and manages SQLite tables for organization-specific security standards,
coding guidelines, compliance frameworks, and best practices.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class KnowledgeBaseSetup:
    """Manages the enterprise knowledge base database schema and operations"""
    
    def __init__(self, db_path: str = "backend/cache/knowledge_base.db"):
        """
        Initialize knowledge base database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Enable full-text search
        self.cursor.execute("PRAGMA foreign_keys = ON")
        
        logger.info(f"Knowledge base database initialized at {self.db_path}")
    
    def create_schema(self):
        """Create all knowledge base tables with optimized indexes"""
        
        # Enterprise Policies Table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS enterprise_policies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                policy_name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                severity TEXT CHECK(severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')) NOT NULL,
                implementation_guide TEXT,
                exceptions TEXT,
                tags TEXT,  -- JSON array of tags
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                version INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Coding Standards Table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS coding_standards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language TEXT NOT NULL,
                standard_name TEXT NOT NULL,
                rule_description TEXT NOT NULL,
                example_good TEXT,
                example_bad TEXT,
                auto_fixable BOOLEAN DEFAULT 0,
                linter_rule TEXT,
                severity TEXT CHECK(severity IN ('ERROR', 'WARNING', 'INFO')) NOT NULL,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Compliance Frameworks Table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_frameworks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                framework_name TEXT NOT NULL,
                requirement_id TEXT NOT NULL,
                requirement_text TEXT NOT NULL,
                description TEXT,
                gcp_mapping TEXT,  -- JSON mapping to GCP services
                evidence_required TEXT,
                audit_frequency TEXT,
                last_audit_date DATE,
                compliance_status TEXT CHECK(compliance_status IN ('COMPLIANT', 'NON_COMPLIANT', 'PARTIAL', 'NOT_ASSESSED')),
                remediation_steps TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(framework_name, requirement_id)
            )
        """)
        
        # Best Practices Table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS best_practices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL,
                practice_name TEXT NOT NULL,
                category TEXT NOT NULL,
                rationale TEXT NOT NULL,
                implementation_guide TEXT NOT NULL,
                risk_if_not_followed TEXT,
                automation_possible BOOLEAN DEFAULT 0,
                terraform_snippet TEXT,
                gcloud_command TEXT,
                verification_steps TEXT,
                reference_links TEXT,  -- JSON array of reference URLs
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Policy History Table (for audit trail)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS policy_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                record_id INTEGER NOT NULL,
                action TEXT CHECK(action IN ('CREATE', 'UPDATE', 'DELETE')) NOT NULL,
                old_values TEXT,  -- JSON of previous values
                new_values TEXT,  -- JSON of new values
                changed_by TEXT NOT NULL,
                change_reason TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Policy Violations Table (track violations found)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS policy_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_type TEXT NOT NULL,
                policy_id INTEGER NOT NULL,
                resource_type TEXT,
                resource_name TEXT,
                violation_details TEXT,
                severity TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                resolution_notes TEXT,
                false_positive BOOLEAN DEFAULT 0
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_policies_category ON enterprise_policies(category)",
            "CREATE INDEX IF NOT EXISTS idx_policies_severity ON enterprise_policies(severity)",
            "CREATE INDEX IF NOT EXISTS idx_policies_active ON enterprise_policies(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_standards_language ON coding_standards(language)",
            "CREATE INDEX IF NOT EXISTS idx_standards_severity ON coding_standards(severity)",
            "CREATE INDEX IF NOT EXISTS idx_compliance_framework ON compliance_frameworks(framework_name)",
            "CREATE INDEX IF NOT EXISTS idx_compliance_status ON compliance_frameworks(compliance_status)",
            "CREATE INDEX IF NOT EXISTS idx_practices_service ON best_practices(service)",
            "CREATE INDEX IF NOT EXISTS idx_practices_category ON best_practices(category)",
            "CREATE INDEX IF NOT EXISTS idx_violations_policy ON policy_violations(policy_type, policy_id)",
            "CREATE INDEX IF NOT EXISTS idx_violations_resolved ON policy_violations(resolved_at)",
            "CREATE INDEX IF NOT EXISTS idx_history_table_record ON policy_history(table_name, record_id)"
        ]
        
        for index in indexes:
            self.cursor.execute(index)
        
        # Create full-text search virtual tables
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS policies_fts USING fts5(
                policy_name, description, implementation_guide,
                content=enterprise_policies
            )
        """)
        
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS standards_fts USING fts5(
                standard_name, rule_description,
                content=coding_standards
            )
        """)
        
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS practices_fts USING fts5(
                practice_name, rationale, implementation_guide,
                content=best_practices
            )
        """)
        
        # Create triggers for updating timestamps
        self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_policies_timestamp
            AFTER UPDATE ON enterprise_policies
            BEGIN
                UPDATE enterprise_policies SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END
        """)
        
        self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_standards_timestamp
            AFTER UPDATE ON coding_standards
            BEGIN
                UPDATE coding_standards SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END
        """)
        
        self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_compliance_timestamp
            AFTER UPDATE ON compliance_frameworks
            BEGIN
                UPDATE compliance_frameworks SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END
        """)
        
        self.cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_practices_timestamp
            AFTER UPDATE ON best_practices
            BEGIN
                UPDATE best_practices SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END
        """)
        
        self.conn.commit()
        logger.info("Knowledge base schema created successfully")
    
    def insert_sample_data(self):
        """Insert sample enterprise policies and best practices"""
        
        # Sample enterprise policies
        sample_policies = [
            {
                "category": "Access Control",
                "policy_name": "Least Privilege Access",
                "description": "All users and service accounts must have minimum required permissions",
                "severity": "CRITICAL",
                "implementation_guide": "1. Review all IAM bindings quarterly\n2. Use predefined roles where possible\n3. Document justification for custom roles",
                "tags": json.dumps(["iam", "security", "compliance"])
            },
            {
                "category": "Data Protection",
                "policy_name": "Encryption at Rest",
                "description": "All sensitive data must be encrypted at rest using CMEK",
                "severity": "HIGH",
                "implementation_guide": "1. Enable CMEK for all storage buckets\n2. Use Cloud KMS for key management\n3. Rotate keys annually",
                "tags": json.dumps(["encryption", "data", "compliance"])
            },
            {
                "category": "Network Security",
                "policy_name": "No Public IPs",
                "description": "Production resources must not have public IP addresses",
                "severity": "HIGH",
                "implementation_guide": "1. Use Cloud NAT for outbound traffic\n2. Use Cloud Load Balancer for inbound traffic\n3. Implement Private Google Access",
                "tags": json.dumps(["network", "security"])
            }
        ]
        
        for policy in sample_policies:
            self.cursor.execute("""
                INSERT OR IGNORE INTO enterprise_policies 
                (category, policy_name, description, severity, implementation_guide, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (policy["category"], policy["policy_name"], policy["description"],
                  policy["severity"], policy["implementation_guide"], policy["tags"]))
        
        # Sample coding standards
        sample_standards = [
            {
                "language": "Python",
                "standard_name": "No Hardcoded Secrets",
                "rule_description": "Never hardcode API keys, passwords, or secrets in source code",
                "example_good": "api_key = os.getenv('API_KEY')",
                "example_bad": "api_key = 'AIzaSyC...'",
                "severity": "ERROR",
                "linter_rule": "bandit:B105",
                "tags": json.dumps(["security", "secrets"])
            },
            {
                "language": "Terraform",
                "standard_name": "Resource Tagging",
                "rule_description": "All resources must have required tags: environment, owner, cost-center",
                "example_good": "tags = {\n  environment = var.environment\n  owner = var.owner\n  cost-center = var.cost_center\n}",
                "example_bad": "# No tags defined",
                "severity": "WARNING",
                "tags": json.dumps(["governance", "cost"])
            }
        ]
        
        for standard in sample_standards:
            self.cursor.execute("""
                INSERT OR IGNORE INTO coding_standards
                (language, standard_name, rule_description, example_good, example_bad, severity, linter_rule, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (standard["language"], standard["standard_name"], standard["rule_description"],
                  standard["example_good"], standard["example_bad"], standard["severity"],
                  standard.get("linter_rule"), standard["tags"]))
        
        # Sample compliance frameworks
        sample_compliance = [
            {
                "framework_name": "SOC2",
                "requirement_id": "CC6.1",
                "requirement_text": "Logical and Physical Access Controls",
                "description": "The entity implements logical access security software, infrastructure, and architectures",
                "gcp_mapping": json.dumps(["IAM", "Cloud Identity", "VPC Service Controls"]),
                "compliance_status": "PARTIAL",
                "tags": json.dumps(["soc2", "access-control"])
            },
            {
                "framework_name": "PCI-DSS",
                "requirement_id": "2.2.1",
                "requirement_text": "Implement only one primary function per server",
                "description": "Implement only one primary function per server to prevent functions that require different security levels from co-existing on the same server",
                "gcp_mapping": json.dumps(["Compute Engine", "GKE", "Cloud Run"]),
                "compliance_status": "COMPLIANT",
                "tags": json.dumps(["pci", "infrastructure"])
            }
        ]
        
        for compliance in sample_compliance:
            self.cursor.execute("""
                INSERT OR IGNORE INTO compliance_frameworks
                (framework_name, requirement_id, requirement_text, description, gcp_mapping, compliance_status, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (compliance["framework_name"], compliance["requirement_id"], compliance["requirement_text"],
                  compliance["description"], compliance["gcp_mapping"], compliance["compliance_status"],
                  compliance["tags"]))
        
        # Sample best practices
        sample_practices = [
            {
                "service": "Cloud Storage",
                "practice_name": "Enable Versioning",
                "category": "Data Protection",
                "rationale": "Protects against accidental deletion and provides data recovery capabilities",
                "implementation_guide": "gsutil versioning set on gs://bucket-name",
                "risk_if_not_followed": "Permanent data loss if objects are accidentally deleted",
                "gcloud_command": "gsutil versioning set on gs://bucket-name",
                "tags": json.dumps(["storage", "backup", "recovery"])
            },
            {
                "service": "Compute Engine",
                "practice_name": "Use Shielded VMs",
                "category": "Security",
                "rationale": "Protects against rootkits and bootkits",
                "implementation_guide": "Enable shielded VM features: Secure Boot, vTPM, Integrity Monitoring",
                "risk_if_not_followed": "VMs vulnerable to persistent malware and rootkits",
                "terraform_snippet": "shielded_instance_config {\n  enable_secure_boot = true\n  enable_vtpm = true\n  enable_integrity_monitoring = true\n}",
                "tags": json.dumps(["compute", "security", "compliance"])
            }
        ]
        
        for practice in sample_practices:
            self.cursor.execute("""
                INSERT OR IGNORE INTO best_practices
                (service, practice_name, category, rationale, implementation_guide, 
                 risk_if_not_followed, gcloud_command, terraform_snippet, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (practice["service"], practice["practice_name"], practice["category"],
                  practice["rationale"], practice["implementation_guide"],
                  practice["risk_if_not_followed"], practice.get("gcloud_command"),
                  practice.get("terraform_snippet"), practice["tags"]))
        
        self.conn.commit()
        logger.info("Sample data inserted successfully")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Initialize the knowledge base database"""
    kb = KnowledgeBaseSetup()
    kb.create_schema()
    kb.insert_sample_data()
    kb.close()
    print("[OK] Knowledge base database initialized successfully")


if __name__ == "__main__":
    main()
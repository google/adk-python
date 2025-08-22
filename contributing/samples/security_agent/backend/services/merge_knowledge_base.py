#!/usr/bin/env python3
"""
Merge Knowledge Base into Main Database
========================================

This script merges the knowledge base tables from knowledge_base.db 
into the main gcp_data.db to enable direct querying from the chat agent.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_databases():
    """Merge knowledge base tables into main database"""
    
    # Database paths
    main_db_path = Path("backend/cache/gcp_data.db")
    kb_db_path = Path("backend/cache/knowledge_base.db")
    
    if not kb_db_path.exists():
        logger.error(f"Knowledge base database not found at {kb_db_path}")
        return False
    
    if not main_db_path.exists():
        logger.warning(f"Main database not found at {main_db_path}, will be created")
        main_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Merging {kb_db_path} into {main_db_path}")
    
    # Connect to both databases
    main_conn = sqlite3.connect(str(main_db_path))
    main_cursor = main_conn.cursor()
    
    kb_conn = sqlite3.connect(str(kb_db_path))
    kb_conn.row_factory = sqlite3.Row
    kb_cursor = kb_conn.cursor()
    
    # Create knowledge base tables in main database
    logger.info("Creating knowledge base tables in main database...")
    
    # Enterprise Policies Table
    main_cursor.execute("""
        CREATE TABLE IF NOT EXISTS enterprise_policies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            policy_name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            severity TEXT CHECK(severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')) NOT NULL,
            implementation_guide TEXT,
            exceptions TEXT,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            version INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    
    # Coding Standards Table
    main_cursor.execute("""
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
    main_cursor.execute("""
        CREATE TABLE IF NOT EXISTS compliance_frameworks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            framework_name TEXT NOT NULL,
            requirement_id TEXT NOT NULL,
            requirement_text TEXT NOT NULL,
            description TEXT,
            gcp_mapping TEXT,
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
    main_cursor.execute("""
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
            reference_links TEXT,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    
    # Policy Violations Table
    main_cursor.execute("""
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
    
    # Copy data from knowledge base to main database
    tables_to_copy = [
        'enterprise_policies',
        'coding_standards', 
        'compliance_frameworks',
        'best_practices',
        'policy_violations'
    ]
    
    for table in tables_to_copy:
        try:
            # Check if table exists in source
            kb_cursor.execute(f"SELECT * FROM {table}")
            rows = kb_cursor.fetchall()
            
            if rows:
                logger.info(f"Copying {len(rows)} rows from {table}...")
                
                # Get column names
                columns = [desc[0] for desc in kb_cursor.description]
                columns_str = ', '.join(columns)
                placeholders = ', '.join(['?' for _ in columns])
                
                # Clear existing data to avoid duplicates
                main_cursor.execute(f"DELETE FROM {table}")
                
                # Insert data
                for row in rows:
                    values = [row[col] for col in columns]
                    main_cursor.execute(
                        f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})",
                        values
                    )
                
                logger.info(f"✅ Copied {len(rows)} rows to {table}")
            else:
                logger.info(f"No data in {table}")
                
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                logger.warning(f"Table {table} doesn't exist in source database")
            else:
                logger.error(f"Error copying {table}: {e}")
    
    # Add test-specific coding standards
    logger.info("Adding test-specific coding standards...")
    
    test_standards = [
        {
            "language": "Python",
            "standard_name": "Test Coverage Requirement",
            "rule_description": "All new features must have >80% test coverage",
            "example_good": "# Run: pytest --cov=mymodule --cov-report=html\n# Coverage: 85%",
            "example_bad": "# No tests written for new feature",
            "severity": "ERROR",
            "linter_rule": "coverage:80",
            "tags": json.dumps(["testing", "quality", "coverage"])
        },
        {
            "language": "Python",
            "standard_name": "Test Naming Convention",
            "rule_description": "Test functions must start with 'test_' and describe what they test",
            "example_good": "def test_user_authentication_with_valid_credentials():",
            "example_bad": "def check_login():",
            "severity": "WARNING",
            "tags": json.dumps(["testing", "naming", "conventions"])
        },
        {
            "language": "Python",
            "standard_name": "Mock External Services",
            "rule_description": "External API calls must be mocked in unit tests",
            "example_good": "@patch('requests.get')\ndef test_api_call(mock_get):\n    mock_get.return_value.json.return_value = {'status': 'ok'}",
            "example_bad": "def test_api_call():\n    response = requests.get('https://api.example.com')",
            "severity": "ERROR",
            "tags": json.dumps(["testing", "mocking", "isolation"])
        },
        {
            "language": "Python", 
            "standard_name": "Test Data Management",
            "rule_description": "Test data should be isolated and not affect production",
            "example_good": "def test_with_fixture(test_database):\n    # Use test-specific database",
            "example_bad": "def test_production_data():\n    # Directly modifies production database",
            "severity": "ERROR",
            "tags": json.dumps(["testing", "data", "isolation"])
        },
        {
            "language": "Python",
            "standard_name": "Assert Meaningful Messages",
            "rule_description": "Assertions should include descriptive failure messages",
            "example_good": "assert user.is_active, f'User {user.id} should be active after registration'",
            "example_bad": "assert user.is_active",
            "severity": "INFO",
            "tags": json.dumps(["testing", "debugging", "assertions"])
        }
    ]
    
    for standard in test_standards:
        try:
            main_cursor.execute("""
                INSERT OR IGNORE INTO coding_standards
                (language, standard_name, rule_description, example_good, example_bad, severity, linter_rule, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                standard["language"], standard["standard_name"], standard["rule_description"],
                standard.get("example_good"), standard.get("example_bad"), standard["severity"],
                standard.get("linter_rule"), standard["tags"]
            ))
        except sqlite3.IntegrityError:
            logger.info(f"Standard '{standard['standard_name']}' already exists")
    
    logger.info("✅ Added test-specific coding standards")
    
    # Create indexes for performance
    logger.info("Creating indexes...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_policies_category ON enterprise_policies(category)",
        "CREATE INDEX IF NOT EXISTS idx_policies_severity ON enterprise_policies(severity)",
        "CREATE INDEX IF NOT EXISTS idx_standards_language ON coding_standards(language)",
        "CREATE INDEX IF NOT EXISTS idx_standards_severity ON coding_standards(severity)",
        "CREATE INDEX IF NOT EXISTS idx_compliance_framework ON compliance_frameworks(framework_name)",
        "CREATE INDEX IF NOT EXISTS idx_practices_service ON best_practices(service)",
        "CREATE INDEX IF NOT EXISTS idx_violations_resolved ON policy_violations(resolved_at)"
    ]
    
    for index in indexes:
        main_cursor.execute(index)
    
    # Commit changes
    main_conn.commit()
    
    # Show statistics
    logger.info("\n📊 Database merge complete!")
    logger.info("Knowledge base tables in main database:")
    
    for table in ['enterprise_policies', 'coding_standards', 'compliance_frameworks', 'best_practices']:
        main_cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
        count = main_cursor.fetchone()[0]
        logger.info(f"  - {table}: {count} records")
    
    # Close connections
    main_conn.close()
    kb_conn.close()
    
    return True


def main():
    """Run the merge operation"""
    print("=" * 60)
    print("Merging Knowledge Base into Main Database")
    print("=" * 60)
    
    success = merge_databases()
    
    if success:
        print("\n✅ Successfully merged knowledge base into main database!")
        print("The chat agent can now query knowledge base data directly.")
    else:
        print("\n❌ Failed to merge databases")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
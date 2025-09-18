#!/usr/bin/env python3
"""Add test data to the SQLite database."""

import sqlite3
from datetime import datetime
from pathlib import Path
import json

# Get database path
db_path = Path(__file__).parent / "backend" / "cache" / "gcp_data.db"

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Add test security findings
test_findings = [
    {
        "finding_id": "high-risk-api-001",
        "finding_type": "API_KEY_EXPOSED",
        "severity": "HIGH",
        "resource_name": "//compute.googleapis.com/projects/mgm-digitalconcierge/zones/us-central1-a/instances/web-server-1",
        "category": "INSECURE_API_USAGE",
        "state": "ACTIVE",
        "create_time": datetime.now().isoformat(),
        "recommendation": "Rotate API keys immediately and use service accounts instead",
        "description": "API key found exposed in public repository",
        "project_id": "mgm-digitalconcierge"
    },
    {
        "finding_id": "open-firewall-002",
        "finding_type": "FIREWALL_RULE_TOO_PERMISSIVE",
        "severity": "CRITICAL",
        "resource_name": "//compute.googleapis.com/projects/mgm-digitalconcierge/global/firewalls/allow-all",
        "category": "FIREWALL_MISCONFIGURATION",
        "state": "ACTIVE",
        "create_time": datetime.now().isoformat(),
        "recommendation": "Restrict firewall rules to specific IP ranges and ports",
        "description": "Firewall rule allows unrestricted access from 0.0.0.0/0",
        "project_id": "mgm-digitalconcierge"
    },
    {
        "finding_id": "weak-iam-003",
        "finding_type": "OVERLY_PERMISSIVE_IAM",
        "severity": "MEDIUM",
        "resource_name": "//cloudresourcemanager.googleapis.com/projects/mgm-digitalconcierge",
        "category": "WEAK_IAM_POLICY",
        "state": "ACTIVE",
        "create_time": datetime.now().isoformat(),
        "recommendation": "Apply principle of least privilege to IAM roles",
        "description": "Service account has Owner role when Editor would suffice",
        "project_id": "mgm-digitalconcierge"
    }
]

# Clear existing data
cursor.execute("DELETE FROM security_findings")

# Insert test findings
for finding in test_findings:
    cursor.execute("""
        INSERT INTO security_findings
        (finding_id, finding_type, severity, resource_name, category, state,
         create_time, recommendation, description, project_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        finding['finding_id'], finding['finding_type'], finding['severity'],
        finding['resource_name'], finding['category'], finding['state'],
        finding['create_time'], finding['recommendation'], finding['description'],
        finding['project_id']
    ))

# Add test storage buckets
test_buckets = [
    {
        "name": "mgm-digitalconcierge-logs",
        "location": "US",
        "storage_class": "STANDARD",
        "versioning": 1,
        "encryption": "Google-managed",
        "public_access": "Not public",
        "created": datetime.now().isoformat(),
        "project_id": "mgm-digitalconcierge"
    },
    {
        "name": "mgm-digitalconcierge-backups",
        "location": "US-CENTRAL1",
        "storage_class": "NEARLINE",
        "versioning": 1,
        "encryption": "Customer-managed",
        "public_access": "Not public",
        "created": datetime.now().isoformat(),
        "project_id": "mgm-digitalconcierge"
    },
    {
        "name": "mgm-digitalconcierge-public-assets",
        "location": "US",
        "storage_class": "STANDARD",
        "versioning": 0,
        "encryption": "Google-managed",
        "public_access": "Public to internet",
        "created": datetime.now().isoformat(),
        "project_id": "mgm-digitalconcierge"
    }
]

# Clear and insert buckets
cursor.execute("DELETE FROM storage_buckets")
for bucket in test_buckets:
    cursor.execute("""
        INSERT INTO storage_buckets
        (name, location, storage_class, versioning, encryption, public_access, created, project_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        bucket['name'], bucket['location'], bucket['storage_class'],
        bucket['versioning'], bucket['encryption'], bucket['public_access'],
        bucket['created'], bucket['project_id']
    ))

# Commit changes
conn.commit()

# Verify data was inserted
cursor.execute("SELECT COUNT(*) FROM security_findings")
findings_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM storage_buckets")
buckets_count = cursor.fetchone()[0]

print(f"✅ Successfully added test data:")
print(f"   - {findings_count} security findings")
print(f"   - {buckets_count} storage buckets")

conn.close()
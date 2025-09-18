#!/usr/bin/env python3
"""
Demo Data Setup for New Users
==============================

This script creates a working SQLite database with demo data
for users who don't have GCP credentials or want to test quickly.

Usage:
    python scripts/setup_demo_data.py

This will create a database with realistic but fake GCP security data.
"""

import os
import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_demo_database():
    """Create SQLite database with demo data for testing."""

    print("🚀 Setting up Demo Database")
    print("=" * 40)

    # Ensure backend directory exists
    backend_dir = project_root / "backend"
    backend_dir.mkdir(exist_ok=True)

    cache_dir = backend_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    db_path = cache_dir / "gcp_data.db"
    print(f"📂 Database path: {db_path}")

    # Remove existing database
    if db_path.exists():
        db_path.unlink()
        print("🗑️ Removed existing database")

    # Create new database with tables
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    print("📋 Creating database tables...")

    # Create all tables
    tables = {
        'assets': '''
            CREATE TABLE assets (
                id INTEGER PRIMARY KEY,
                name TEXT,
                asset_type TEXT,
                location TEXT,
                created_at TEXT,
                project_id TEXT
            )
        ''',
        'compute_instances': '''
            CREATE TABLE compute_instances (
                id INTEGER PRIMARY KEY,
                name TEXT,
                zone TEXT,
                machine_type TEXT,
                status TEXT,
                created_at TEXT,
                project_id TEXT
            )
        ''',
        'storage_buckets': '''
            CREATE TABLE storage_buckets (
                id INTEGER PRIMARY KEY,
                name TEXT,
                location TEXT,
                storage_class TEXT,
                versioning_enabled BOOLEAN,
                encryption_type TEXT,
                public_access_prevention TEXT,
                uniform_bucket_level_access BOOLEAN,
                created_at TEXT,
                project_id TEXT
            )
        ''',
        'iam_accounts': '''
            CREATE TABLE iam_accounts (
                id INTEGER PRIMARY KEY,
                email TEXT,
                role TEXT,
                account_type TEXT,
                created_at TEXT,
                project_id TEXT
            )
        ''',
        'security_findings': '''
            CREATE TABLE security_findings (
                id INTEGER PRIMARY KEY,
                name TEXT,
                category TEXT,
                severity TEXT,
                resource_name TEXT,
                description TEXT,
                recommendation TEXT,
                state TEXT,
                created_at TEXT,
                project_id TEXT
            )
        ''',
        'networks': '''
            CREATE TABLE networks (
                id INTEGER PRIMARY KEY,
                name TEXT,
                subnet_mode TEXT,
                routing_mode TEXT,
                created_at TEXT,
                project_id TEXT
            )
        ''',
        'firewall_rules': '''
            CREATE TABLE firewall_rules (
                id INTEGER PRIMARY KEY,
                name TEXT,
                direction TEXT,
                priority INTEGER,
                source_ranges TEXT,
                target_tags TEXT,
                allowed_ports TEXT,
                created_at TEXT,
                project_id TEXT
            )
        ''',
        'databases': '''
            CREATE TABLE databases (
                id INTEGER PRIMARY KEY,
                name TEXT,
                database_type TEXT,
                version TEXT,
                region TEXT,
                tier TEXT,
                backup_enabled BOOLEAN,
                ssl_required BOOLEAN,
                created_at TEXT,
                project_id TEXT
            )
        '''
    }

    for table_name, sql in tables.items():
        cursor.execute(sql)
        print(f"✅ Created table: {table_name}")

    print("\n📝 Inserting demo data...")

    # Insert demo storage buckets (main test data)
    storage_buckets = [
        ("demo-storage-bucket-1", "us-central1", "STANDARD", True, "Google-managed", "enforced", True, "2024-01-15", "demo-project"),
        ("demo-storage-bucket-2", "us-east1", "NEARLINE", False, "Customer-managed", "inherited", False, "2024-02-01", "demo-project"),
        ("demo-public-bucket", "us-west1", "STANDARD", False, "None", "inherited", False, "2024-01-20", "demo-project"),
        ("demo-secure-bucket", "europe-west1", "STANDARD", True, "Customer-managed", "enforced", True, "2024-01-25", "demo-project"),
        ("demo-archive-bucket", "asia-east1", "ARCHIVE", True, "Google-managed", "enforced", True, "2024-02-10", "demo-project"),
        ("demo-backup-bucket", "us-central1", "COLDLINE", True, "Customer-managed", "enforced", True, "2024-02-15", "demo-project"),
        ("demo-logs-bucket", "us-east1", "STANDARD", False, "Google-managed", "inherited", False, "2024-01-30", "demo-project"),
        ("demo-temp-bucket", "us-west1", "STANDARD", False, "None", "inherited", False, "2024-02-05", "demo-project"),
        ("demo-prod-bucket", "europe-west1", "STANDARD", True, "Customer-managed", "enforced", True, "2024-01-10", "demo-project"),
        ("demo-dev-bucket", "us-central1", "STANDARD", False, "Google-managed", "inherited", False, "2024-02-20", "demo-project"),
        ("demo-staging-bucket", "us-east1", "STANDARD", True, "Customer-managed", "enforced", True, "2024-01-18", "demo-project"),
        ("demo-analytics-bucket", "us-west1", "NEARLINE", True, "Google-managed", "enforced", True, "2024-02-12", "demo-project"),
        ("demo-media-bucket", "asia-east1", "STANDARD", False, "None", "inherited", False, "2024-01-28", "demo-project"),
        ("demo-config-bucket", "europe-west1", "STANDARD", True, "Customer-managed", "enforced", True, "2024-02-08", "demo-project")
    ]

    for bucket in storage_buckets:
        cursor.execute("""
            INSERT INTO storage_buckets
            (name, location, storage_class, versioning_enabled, encryption_type,
             public_access_prevention, uniform_bucket_level_access, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, bucket)

    print(f"✅ Inserted {len(storage_buckets)} storage buckets")

    # Insert demo security findings
    security_findings = [
        ("Public Storage Bucket", "STORAGE", "HIGH", "demo-public-bucket", "Storage bucket allows public read access", "Restrict public access to prevent data exposure", "ACTIVE", "2024-02-01", "demo-project"),
        ("Unencrypted Bucket", "STORAGE", "MEDIUM", "demo-temp-bucket", "Storage bucket lacks encryption at rest", "Enable encryption to protect sensitive data", "ACTIVE", "2024-02-02", "demo-project"),
        ("Overprivileged IAM Role", "IAM", "HIGH", "demo-service-account", "Service account has excessive permissions", "Apply principle of least privilege", "ACTIVE", "2024-02-03", "demo-project"),
        ("Open Firewall Rule", "NETWORK", "CRITICAL", "demo-firewall-allow-all", "Firewall rule allows unrestricted access", "Restrict source IP ranges and ports", "ACTIVE", "2024-02-04", "demo-project"),
        ("Weak Database Security", "DATABASE", "MEDIUM", "demo-sql-instance", "SQL instance allows non-SSL connections", "Require SSL connections for all database access", "ACTIVE", "2024-02-05", "demo-project")
    ]

    for finding in security_findings:
        cursor.execute("""
            INSERT INTO security_findings
            (name, category, severity, resource_name, description, recommendation, state, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, finding)

    print(f"✅ Inserted {len(security_findings)} security findings")

    # Insert demo compute instances
    compute_instances = [
        ("demo-web-server-1", "us-central1-a", "n1-standard-1", "RUNNING", "2024-01-15", "demo-project"),
        ("demo-web-server-2", "us-central1-b", "n1-standard-1", "RUNNING", "2024-01-16", "demo-project"),
        ("demo-db-server", "us-east1-a", "n1-standard-2", "RUNNING", "2024-01-20", "demo-project"),
        ("demo-worker-1", "us-west1-a", "n1-standard-1", "STOPPED", "2024-02-01", "demo-project"),
        ("demo-analytics", "europe-west1-a", "n1-highmem-2", "RUNNING", "2024-01-25", "demo-project")
    ]

    for instance in compute_instances:
        cursor.execute("""
            INSERT INTO compute_instances
            (name, zone, machine_type, status, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, instance)

    print(f"✅ Inserted {len(compute_instances)} compute instances")

    # Insert demo IAM accounts
    iam_accounts = [
        ("admin@demo-project.iam.gserviceaccount.com", "roles/owner", "service_account", "2024-01-10", "demo-project"),
        ("storage-service@demo-project.iam.gserviceaccount.com", "roles/storage.admin", "service_account", "2024-01-15", "demo-project"),
        ("compute-service@demo-project.iam.gserviceaccount.com", "roles/compute.admin", "service_account", "2024-01-20", "demo-project"),
        ("demo-user@example.com", "roles/viewer", "user", "2024-02-01", "demo-project"),
        ("security-scanner@demo-project.iam.gserviceaccount.com", "roles/securitycenter.admin", "service_account", "2024-01-25", "demo-project")
    ]

    for account in iam_accounts:
        cursor.execute("""
            INSERT INTO iam_accounts
            (email, role, account_type, created_at, project_id)
            VALUES (?, ?, ?, ?, ?)
        """, account)

    print(f"✅ Inserted {len(iam_accounts)} IAM accounts")

    # Insert demo networks
    networks = [
        ("default", "AUTO", "REGIONAL", "2024-01-01", "demo-project"),
        ("demo-vpc", "CUSTOM", "GLOBAL", "2024-01-15", "demo-project"),
        ("secure-network", "CUSTOM", "REGIONAL", "2024-02-01", "demo-project")
    ]

    for network in networks:
        cursor.execute("""
            INSERT INTO networks
            (name, subnet_mode, routing_mode, created_at, project_id)
            VALUES (?, ?, ?, ?, ?)
        """, network)

    print(f"✅ Inserted {len(networks)} networks")

    # Insert demo firewall rules
    firewall_rules = [
        ("default-allow-http", "INGRESS", 1000, "0.0.0.0/0", "http-server", "tcp:80", "2024-01-01", "demo-project"),
        ("default-allow-https", "INGRESS", 1000, "0.0.0.0/0", "https-server", "tcp:443", "2024-01-01", "demo-project"),
        ("demo-allow-ssh", "INGRESS", 1000, "10.0.0.0/8", "ssh-access", "tcp:22", "2024-01-15", "demo-project"),
        ("demo-allow-all", "INGRESS", 1000, "0.0.0.0/0", "all", "tcp:0-65535", "2024-02-01", "demo-project")
    ]

    for rule in firewall_rules:
        cursor.execute("""
            INSERT INTO firewall_rules
            (name, direction, priority, source_ranges, target_tags, allowed_ports, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rule)

    print(f"✅ Inserted {len(firewall_rules)} firewall rules")

    # Insert demo databases
    databases = [
        ("demo-mysql-db", "MySQL", "8.0", "us-central1", "db-n1-standard-1", True, True, "2024-01-20", "demo-project"),
        ("demo-postgres-db", "PostgreSQL", "13", "us-east1", "db-n1-standard-2", True, False, "2024-01-25", "demo-project"),
        ("demo-redis-cache", "Redis", "6.0", "us-west1", "basic", False, False, "2024-02-01", "demo-project")
    ]

    for db in databases:
        cursor.execute("""
            INSERT INTO databases
            (name, database_type, version, region, tier, backup_enabled, ssl_required, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, db)

    print(f"✅ Inserted {len(databases)} databases")

    conn.commit()
    conn.close()

    print(f"\n🎉 Demo database created successfully!")
    print(f"📂 Location: {db_path}")
    print(f"📊 Total records: {sum([
        len(storage_buckets), len(security_findings), len(compute_instances),
        len(iam_accounts), len(networks), len(firewall_rules), len(databases)
    ])}")

    return str(db_path)

if __name__ == "__main__":
    create_demo_database()
#!/usr/bin/env python3
"""
Populate GCP database with realistic demo data for security agent testing
"""

import sqlite3
import os
from datetime import datetime, timedelta
import random

# Database path
DB_PATH = "backend/cache/gcp_data.db"
PROJECT_ID = "mgm-digitalconcierge"

def clear_tables(cursor):
    """Clear all existing data"""
    tables = ['assets', 'compute_instances', 'storage_buckets', 'iam_accounts',
              'security_findings', 'networks', 'firewall_rules', 'databases']

    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
    print("✅ Cleared existing data")

def populate_storage_buckets(cursor):
    """Add realistic storage buckets"""
    buckets = [
        # Production buckets - more secure
        ("mgm-digitalconcierge-prod-logs", "us-central1", "STANDARD", True, "Google-managed", "enforced", True),
        ("mgm-digitalconcierge-prod-backups", "us-east1", "COLDLINE", True, "Customer-managed", "enforced", True),
        ("mgm-digitalconcierge-terraform-state", "us-central1", "STANDARD", True, "Google-managed", "enforced", True),

        # Development buckets - some security issues
        ("mgm-digitalconcierge-dev-uploads", "us-west1", "STANDARD", False, "Google-managed", "inherited", False),
        ("mgm-digitalconcierge-temp-data", "us-east4", "STANDARD", False, "Google-managed", "inherited", False),

        # Problematic buckets - security risks
        ("mgm-public-assets", "us", "STANDARD", False, "None", "inherited", False),
        ("legacy-backup-bucket", "us-east1", "NEARLINE", False, "Google-managed", "inherited", False),
        ("mgm-digitalconcierge-logs-old", "us-central1", "STANDARD", False, "Google-managed", "inherited", True),

        # Cloud Functions and services
        ("gcf-v2-sources-419850945193-us-central1", "us-central1", "STANDARD", False, "Google-managed", "enforced", False),
        ("mgm-digitalconcierge_cloudbuild", "us", "STANDARD", False, "Google-managed", "enforced", False),
    ]

    for i, (name, location, storage_class, versioning, encryption, public_access, uniform_access) in enumerate(buckets):
        created_at = (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat()
        cursor.execute("""
            INSERT INTO storage_buckets
            (name, location, storage_class, versioning_enabled, encryption_type,
             public_access_prevention, uniform_bucket_level_access, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, location, storage_class, versioning, encryption, public_access, uniform_access, created_at, PROJECT_ID))

    print(f"✅ Added {len(buckets)} storage buckets")

def populate_compute_instances(cursor):
    """Add realistic compute instances"""
    instances = [
        ("web-server-prod-1", "us-central1-a", "e2-standard-4", "RUNNING"),
        ("web-server-prod-2", "us-central1-b", "e2-standard-4", "RUNNING"),
        ("api-server-prod", "us-central1-c", "e2-standard-2", "RUNNING"),
        ("database-replica", "us-east1-a", "n1-standard-8", "RUNNING"),
        ("dev-instance", "us-west1-a", "e2-micro", "STOPPED"),
        ("test-vm-legacy", "us-central1-a", "n1-standard-1", "RUNNING"),
        ("analytics-worker", "us-central1-b", "c2-standard-16", "RUNNING"),
        ("backup-processor", "us-east1-b", "e2-standard-2", "STOPPED"),
    ]

    for name, zone, machine_type, status in instances:
        created_at = (datetime.now() - timedelta(days=random.randint(7, 180))).isoformat()
        cursor.execute("""
            INSERT INTO compute_instances
            (name, zone, machine_type, status, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, zone, machine_type, status, created_at, PROJECT_ID))

    print(f"✅ Added {len(instances)} compute instances")

def populate_iam_accounts(cursor):
    """Add realistic IAM accounts"""
    accounts = [
        # Service accounts
        ("mgm-digitalconcierge@mgm-digitalconcierge.iam.gserviceaccount.com", "roles/editor", "serviceAccount"),
        ("terraform@mgm-digitalconcierge.iam.gserviceaccount.com", "roles/owner", "serviceAccount"),
        ("cloudbuild@mgm-digitalconcierge.iam.gserviceaccount.com", "roles/cloudbuild.builds.builder", "serviceAccount"),
        ("storage-admin@mgm-digitalconcierge.iam.gserviceaccount.com", "roles/storage.admin", "serviceAccount"),

        # User accounts
        ("stuart.gano@mgm.com", "roles/owner", "user"),
        ("dev-team@mgm.com", "roles/editor", "user"),
        ("security-team@mgm.com", "roles/securitycenter.admin", "user"),
        ("readonly-analyst@mgm.com", "roles/viewer", "user"),

        # Problematic accounts
        ("legacy-admin@old-domain.com", "roles/owner", "user"),  # External domain
        ("temp-contractor@gmail.com", "roles/editor", "user"),   # Personal email
        ("service-xyz@appspot.gserviceaccount.com", "roles/editor", "serviceAccount"),  # Broad permissions
    ]

    for email, role, account_type in accounts:
        created_at = (datetime.now() - timedelta(days=random.randint(1, 500))).isoformat()
        cursor.execute("""
            INSERT INTO iam_accounts
            (email, role, account_type, created_at, project_id)
            VALUES (?, ?, ?, ?, ?)
        """, (email, role, account_type, created_at, PROJECT_ID))

    print(f"✅ Added {len(accounts)} IAM accounts")

def populate_security_findings(cursor):
    """Add realistic security findings"""
    findings = [
        ("PUBLIC_BUCKET_ACL", "STORAGE", "HIGH", "mgm-public-assets",
         "Storage bucket allows public read access",
         "Remove public access and implement proper IAM controls", "ACTIVE"),

        ("WEAK_SSL_POLICY", "NETWORK", "MEDIUM", "legacy-load-balancer",
         "Load balancer uses outdated SSL policy with weak ciphers",
         "Update SSL policy to use TLS 1.2+ only", "ACTIVE"),

        ("ADMIN_SERVICE_ACCOUNT", "IAM", "HIGH", "service-xyz@appspot.gserviceaccount.com",
         "Service account has excessive editor permissions",
         "Apply principle of least privilege and reduce permissions", "ACTIVE"),

        ("UNENCRYPTED_DISK", "COMPUTE", "MEDIUM", "test-vm-legacy",
         "Compute instance disk is not encrypted with customer-managed keys",
         "Enable customer-managed encryption keys (CMEK)", "ACTIVE"),

        ("OPEN_FIREWALL_RULE", "NETWORK", "HIGH", "allow-all-http",
         "Firewall rule allows HTTP traffic from any source (0.0.0.0/0)",
         "Restrict source IP ranges to necessary addresses only", "ACTIVE"),

        ("EXTERNAL_IAM_USER", "IAM", "MEDIUM", "legacy-admin@old-domain.com",
         "User account from external domain has owner permissions",
         "Review and remove external accounts or limit permissions", "ACTIVE"),

        ("OUTDATED_INSTANCE", "COMPUTE", "LOW", "test-vm-legacy",
         "Compute instance running outdated machine type",
         "Migrate to newer generation machine types", "MUTED"),

        ("VERSIONING_DISABLED", "STORAGE", "MEDIUM", "mgm-digitalconcierge-temp-data",
         "Storage bucket has versioning disabled",
         "Enable object versioning for data protection", "ACTIVE"),

        ("BROAD_NETWORK_ACCESS", "NETWORK", "MEDIUM", "allow-ssh-from-internet",
         "SSH access allowed from any IP address",
         "Restrict SSH access to specific IP ranges or use IAP", "ACTIVE"),

        ("API_KEY_UNRESTRICTED", "IAM", "HIGH", "browser-key-1",
         "API key has no application restrictions",
         "Add HTTP referrer or IP address restrictions", "ACTIVE"),
    ]

    for name, category, severity, resource, description, recommendation, state in findings:
        created_at = (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat()
        cursor.execute("""
            INSERT INTO security_findings
            (name, category, severity, resource_name, description, recommendation, state, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, category, severity, resource, description, recommendation, state, created_at, PROJECT_ID))

    print(f"✅ Added {len(findings)} security findings")

def populate_networks(cursor):
    """Add realistic networks"""
    networks = [
        ("default", "auto", "regional"),
        ("mgm-prod-vpc", "custom", "global"),
        ("mgm-dev-vpc", "custom", "regional"),
        ("legacy-network", "legacy", "global"),
    ]

    for name, subnet_mode, routing_mode in networks:
        created_at = (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat()
        cursor.execute("""
            INSERT INTO networks
            (name, subnet_mode, routing_mode, created_at, project_id)
            VALUES (?, ?, ?, ?, ?)
        """, (name, subnet_mode, routing_mode, created_at, PROJECT_ID))

    print(f"✅ Added {len(networks)} networks")

def populate_firewall_rules(cursor):
    """Add realistic firewall rules"""
    rules = [
        ("default-allow-http", "INGRESS", 1000, "0.0.0.0/0", "web-servers", "tcp:80"),
        ("default-allow-https", "INGRESS", 1000, "0.0.0.0/0", "web-servers", "tcp:443"),
        ("allow-ssh-from-office", "INGRESS", 1000, "203.0.113.0/24", "ssh-allowed", "tcp:22"),
        ("allow-internal", "INGRESS", 1000, "10.0.0.0/8", "", "tcp:1-65535"),
        ("allow-all-http", "INGRESS", 1000, "0.0.0.0/0", "", "tcp:80"),  # Problematic - too open
        ("allow-ssh-from-internet", "INGRESS", 1000, "0.0.0.0/0", "legacy-servers", "tcp:22"),  # Problematic
        ("deny-all-egress", "EGRESS", 65534, "0.0.0.0/0", "", ""),
        ("allow-api-access", "INGRESS", 1000, "192.168.1.0/24", "api-servers", "tcp:8080"),
        ("allow-database", "INGRESS", 1000, "10.1.0.0/16", "db-servers", "tcp:5432"),
        ("allow-monitoring", "INGRESS", 1000, "10.2.0.0/16", "monitoring", "tcp:9090"),
    ]

    for name, direction, priority, source_ranges, target_tags, allowed_ports in rules:
        created_at = (datetime.now() - timedelta(days=random.randint(7, 200))).isoformat()
        cursor.execute("""
            INSERT INTO firewall_rules
            (name, direction, priority, source_ranges, target_tags, allowed_ports, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, direction, priority, source_ranges, target_tags, allowed_ports, created_at, PROJECT_ID))

    print(f"✅ Added {len(rules)} firewall rules")

def populate_databases(cursor):
    """Add realistic databases"""
    databases = [
        ("mgm-prod-db", "POSTGRES", "13", "us-central1", "db-standard-4", True, True),
        ("mgm-analytics-db", "MYSQL", "8.0", "us-east1", "db-standard-8", True, True),
        ("dev-database", "POSTGRES", "12", "us-west1", "db-f1-micro", False, False),
        ("legacy-mysql", "MYSQL", "5.7", "us-central1", "db-standard-1", False, False),
        ("cache-redis", "REDIS", "6.0", "us-central1", "basic", False, True),
        ("test-mongodb", "MONGODB", "4.4", "us-west1", "shared-core", False, False),
    ]

    for name, db_type, version, region, tier, backup_enabled, ssl_required in databases:
        created_at = (datetime.now() - timedelta(days=random.randint(14, 300))).isoformat()
        cursor.execute("""
            INSERT INTO databases
            (name, database_type, version, region, tier, backup_enabled, ssl_required, created_at, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, db_type, version, region, tier, backup_enabled, ssl_required, created_at, PROJECT_ID))

    print(f"✅ Added {len(databases)} databases")

def populate_assets(cursor):
    """Add realistic assets"""
    assets = [
        ("mgm-prod-db", "sql_instance", "us-central1"),
        ("web-server-prod-1", "compute_instance", "us-central1-a"),
        ("mgm-digitalconcierge-prod-logs", "storage_bucket", "us-central1"),
        ("mgm-prod-vpc", "network", "global"),
        ("production-kms-key", "kms_key", "global"),
        ("mgm-ssl-cert", "ssl_certificate", "global"),
        ("prod-load-balancer", "load_balancer", "global"),
        ("api-gateway-prod", "api_gateway", "us-central1"),
        ("cloud-scheduler-job", "scheduler_job", "us-central1"),
        ("pub-sub-topic", "pubsub_topic", "global"),
    ]

    for name, asset_type, location in assets:
        created_at = (datetime.now() - timedelta(days=random.randint(7, 400))).isoformat()
        cursor.execute("""
            INSERT INTO assets
            (name, asset_type, location, created_at, project_id)
            VALUES (?, ?, ?, ?, ?)
        """, (name, asset_type, location, created_at, PROJECT_ID))

    print(f"✅ Added {len(assets)} assets")

def main():
    """Populate database with realistic demo data"""
    print("🔄 Populating GCP database with realistic demo data...")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Clear existing data
        clear_tables(cursor)

        # Populate all tables
        populate_storage_buckets(cursor)
        populate_compute_instances(cursor)
        populate_iam_accounts(cursor)
        populate_security_findings(cursor)
        populate_networks(cursor)
        populate_firewall_rules(cursor)
        populate_databases(cursor)
        populate_assets(cursor)

        # Commit changes
        conn.commit()
        print("\n🎉 Successfully populated database with realistic demo data!")
        print(f"📍 Database location: {DB_PATH}")

        # Show summary
        cursor.execute("SELECT COUNT(*) FROM storage_buckets")
        bucket_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM security_findings WHERE state='ACTIVE'")
        active_findings = cursor.fetchone()[0]

        print(f"📊 Summary:")
        print(f"   • {bucket_count} storage buckets")
        print(f"   • {active_findings} active security findings")
        print(f"   • Data spans realistic time ranges")
        print(f"   • Mix of secure and problematic configurations")

    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
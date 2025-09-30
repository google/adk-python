#!/usr/bin/env python3
"""
Script to populate Confluence cache with sample documentation data for testing.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Configuration
CACHE_DB_PATH = "backend/cache/confluence_cache.db"

# Sample documents
SAMPLE_DOCUMENTS = [
    {
        "document_id": "DOC001",
        "space_key": "SEC",
        "title": "GCP Security Best Practices Guide",
        "content": """
        <h1>GCP Security Best Practices</h1>
        <p>This guide covers essential security practices for Google Cloud Platform.</p>
        <h2>1. Identity and Access Management (IAM)</h2>
        <p>Implement least privilege access controls and use service accounts appropriately.</p>
        <h2>2. Network Security</h2>
        <p>Configure VPC firewall rules, use Private Google Access, and implement Cloud Armor.</p>
        <h2>3. Data Protection</h2>
        <p>Enable encryption at rest and in transit. Use Cloud KMS for key management.</p>
        <p>Tags: PCI-DSS, HIPAA, SOC2 compliant</p>
        """,
        "url": "https://confluence.example.com/display/SEC/gcp-security-best-practices",
        "created_by": "Security Team",
        "modified_by": "John Doe",
        "labels": ["security", "gcp", "best-practices", "compliance"],
    },
    {
        "document_id": "DOC002",
        "space_key": "POLICY",
        "title": "Data Classification and Handling Policy",
        "content": """
        <h1>Data Classification Policy</h1>
        <p>This policy defines how to classify and handle sensitive data.</p>
        <h2>Classification Levels</h2>
        <ul>
            <li>Public - No restrictions</li>
            <li>Internal - Company use only</li>
            <li>Confidential - Restricted access</li>
            <li>Highly Confidential - Need-to-know basis</li>
        </ul>
        <p>All data must be classified according to GDPR and CCPA requirements.</p>
        """,
        "url": "https://confluence.example.com/display/POLICY/data-classification",
        "created_by": "Compliance Team",
        "modified_by": "Jane Smith",
        "labels": ["policy", "data-classification", "gdpr", "ccpa"],
    },
    {
        "document_id": "DOC003",
        "space_key": "SEC",
        "title": "Incident Response Playbook",
        "content": """
        <h1>Security Incident Response Procedures</h1>
        <h2>1. Detection and Analysis</h2>
        <p>Monitor security alerts and analyze potential incidents.</p>
        <h2>2. Containment</h2>
        <p>Isolate affected systems to prevent spread.</p>
        <h2>3. Eradication and Recovery</h2>
        <p>Remove threat and restore systems to normal operation.</p>
        <h2>4. Post-Incident Review</h2>
        <p>Document lessons learned and update procedures.</p>
        """,
        "url": "https://confluence.example.com/display/SEC/incident-response",
        "created_by": "SOC Team",
        "modified_by": "Security Admin",
        "labels": ["incident-response", "security", "playbook"],
    },
    {
        "document_id": "DOC004",
        "space_key": "GCP",
        "title": "Cloud Storage Security Configuration",
        "content": """
        <h1>Securing Cloud Storage Buckets</h1>
        <h2>Access Controls</h2>
        <p>Configure uniform bucket-level access and remove public access where not needed.</p>
        <h2>Encryption</h2>
        <p>Enable default encryption using Google-managed or customer-managed keys.</p>
        <h2>Audit Logging</h2>
        <p>Enable Cloud Audit Logs for all data access.</p>
        <p>Compliance: SOC2, ISO 27001 certified configuration.</p>
        """,
        "url": "https://confluence.example.com/display/GCP/cloud-storage-security",
        "created_by": "Cloud Team",
        "modified_by": "DevOps Engineer",
        "labels": ["cloud-storage", "security", "encryption", "gcp"],
    },
    {
        "document_id": "DOC005",
        "space_key": "POLICY",
        "title": "Access Control and Authentication Standards",
        "content": """
        <h1>Authentication and Access Control Policy</h1>
        <h2>Multi-Factor Authentication</h2>
        <p>MFA is mandatory for all privileged accounts and recommended for all users.</p>
        <h2>Password Requirements</h2>
        <p>Minimum 14 characters, complexity requirements, 90-day rotation.</p>
        <h2>Service Account Management</h2>
        <p>Regular key rotation, least privilege, monitoring of usage.</p>
        <p>Compliance with PCI-DSS and HIPAA requirements.</p>
        """,
        "url": "https://confluence.example.com/display/POLICY/access-control",
        "created_by": "Security Team",
        "modified_by": "Compliance Officer",
        "labels": ["authentication", "access-control", "mfa", "policy"],
    },
    {
        "document_id": "DOC006",
        "space_key": "SEC",
        "title": "Network Security Architecture Guide",
        "content": """
        <h1>GCP Network Security Architecture</h1>
        <h2>VPC Design</h2>
        <p>Implement hub-and-spoke architecture with shared VPC.</p>
        <h2>Firewall Rules</h2>
        <p>Default deny-all with explicit allow rules. Use service accounts for targeting.</p>
        <h2>Cloud Armor</h2>
        <p>Deploy Cloud Armor policies for DDoS protection and WAF rules.</p>
        <h2>Private Google Access</h2>
        <p>Enable Private Google Access for all subnets to avoid public IPs.</p>
        """,
        "url": "https://confluence.example.com/display/SEC/network-security",
        "created_by": "Network Team",
        "modified_by": "Security Architect",
        "labels": ["network", "security", "vpc", "firewall"],
    },
]

def populate_cache():
    """Populate the cache database with sample documents."""
    # Create database directory if it doesn't exist
    db_path = Path(CACHE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS confluence_documents (
            document_id TEXT PRIMARY KEY,
            space_key TEXT,
            title TEXT,
            content TEXT,
            url TEXT,
            created_date TIMESTAMP,
            modified_date TIMESTAMP,
            created_by TEXT,
            modified_by TEXT,
            parent_id TEXT,
            labels TEXT,
            content_hash TEXT,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS confluence_search_cache (
            query_hash TEXT PRIMARY KEY,
            query TEXT,
            spaces TEXT,
            results TEXT,
            result_count INTEGER,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert sample documents
    now = datetime.now()
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        # Generate dates
        created_date = now - timedelta(days=30 + i * 5)
        modified_date = now - timedelta(days=i * 2)

        # Calculate content hash
        content_hash = hashlib.md5(doc["content"].encode()).hexdigest()

        # Insert document
        cursor.execute("""
            INSERT OR REPLACE INTO confluence_documents
            (document_id, space_key, title, content, url,
             created_date, modified_date, created_by, modified_by,
             parent_id, labels, content_hash, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc["document_id"],
            doc["space_key"],
            doc["title"],
            doc["content"],
            doc["url"],
            created_date.isoformat(),
            modified_date.isoformat(),
            doc["created_by"],
            doc["modified_by"],
            None,  # parent_id
            json.dumps(doc["labels"]),
            content_hash,
            now.isoformat()
        ))

    # Create some search cache entries
    search_queries = [
        ("security", ["SEC", "POLICY", "GCP"], ["DOC001", "DOC003", "DOC004", "DOC006"]),
        ("IAM", ["SEC", "POLICY"], ["DOC001", "DOC005"]),
        ("encryption", ["SEC", "GCP"], ["DOC001", "DOC004"]),
        ("compliance", ["POLICY"], ["DOC001", "DOC002", "DOC005"]),
    ]

    for query, spaces, doc_ids in search_queries:
        query_hash = hashlib.md5(f"{query}:{','.join(spaces)}:10".encode()).hexdigest()

        # Get matching documents
        results = []
        for doc_id in doc_ids:
            cursor.execute("""
                SELECT document_id, space_key, title, content, url, modified_date
                FROM confluence_documents
                WHERE document_id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            if row:
                results.append({
                    "id": row[0],
                    "space_key": row[1],
                    "title": row[2],
                    "excerpt": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                    "url": row[4],
                    "modified_date": row[5]
                })

        cursor.execute("""
            INSERT OR REPLACE INTO confluence_search_cache
            (query_hash, query, spaces, results, result_count, cached_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            query_hash,
            query,
            ",".join(spaces),
            json.dumps(results),
            len(results),
            now.isoformat()
        ))

    conn.commit()
    conn.close()

    print(f"✅ Successfully populated cache with {len(SAMPLE_DOCUMENTS)} sample documents")
    print(f"📁 Cache database: {db_path.absolute()}")
    print("\nSample documents added:")
    for doc in SAMPLE_DOCUMENTS:
        print(f"  - [{doc['space_key']}] {doc['title']}")

if __name__ == "__main__":
    populate_cache()
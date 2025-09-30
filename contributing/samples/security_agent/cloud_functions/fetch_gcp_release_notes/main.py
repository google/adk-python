#!/usr/bin/env python3
"""
Cloud Function to fetch Google Cloud release notes from RSS feeds
Runs independently on a schedule (every 4 hours)
"""

import os
import json
import hashlib
import re
from datetime import datetime, timezone
from typing import List, Dict, Any
import requests
import feedparser
from google.cloud import bigquery
from dateutil import parser as date_parser


def extract_security_keywords(text: str) -> List[str]:
    """Extract security-related keywords from text"""
    if not text:
        return []

    text_lower = text.lower()
    security_keywords = [
        'security', 'vulnerability', 'cve', 'patch', 'fix', 'bug fix',
        'authentication', 'authorization', 'encryption', 'ssl', 'tls',
        'firewall', 'iam', 'access control', 'privilege', 'permission',
        'audit', 'compliance', 'gdpr', 'sox', 'hipaa', 'pci',
        'threat', 'attack', 'malware', 'intrusion', 'breach',
        'certificate', 'key management', 'oauth', 'saml',
        'network security', 'data protection', 'privacy'
    ]

    found_keywords = []
    for keyword in security_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)

    return list(set(found_keywords))


def categorize_service(title: str, description: str) -> str:
    """Categorize the service based on title and description"""
    content = f"{title} {description}".lower()

    # Define service categories
    categories = {
        'compute': ['compute engine', 'gke', 'kubernetes', 'vm', 'instance', 'container'],
        'storage': ['cloud storage', 'persistent disk', 'filestore', 'backup'],
        'networking': ['vpc', 'cloud nat', 'load balancer', 'cdn', 'dns', 'firewall'],
        'database': ['cloud sql', 'firestore', 'bigtable', 'spanner', 'redis'],
        'security': ['iam', 'security command center', 'kms', 'secret manager', 'certificate'],
        'ai_ml': ['ai platform', 'automl', 'vertex ai', 'tensorflow', 'machine learning'],
        'analytics': ['bigquery', 'dataflow', 'dataproc', 'pub/sub', 'analytics'],
        'serverless': ['cloud functions', 'cloud run', 'app engine', 'workflows'],
        'monitoring': ['cloud monitoring', 'logging', 'trace', 'profiler', 'error reporting'],
        'data': ['dataflow', 'dataproc', 'dataprep', 'data fusion', 'data catalog']
    }

    for category, keywords in categories.items():
        if any(keyword in content for keyword in keywords):
            return category

    return 'other'


def calculate_security_score(title: str, description: str, keywords: List[str]) -> int:
    """Calculate a security relevance score (0-10)"""
    score = 0
    title_lower = title.lower()
    desc_lower = description.lower() if description else ""

    # High priority security terms
    high_priority = ['cve', 'vulnerability', 'security fix', 'patch', 'critical']
    medium_priority = ['authentication', 'authorization', 'encryption', 'firewall', 'iam']
    low_priority = ['compliance', 'audit', 'privacy', 'access control']

    # Score based on keywords
    for keyword in keywords:
        if any(hp in keyword for hp in high_priority):
            score += 3
        elif any(mp in keyword for mp in medium_priority):
            score += 2
        elif any(lp in keyword for lp in low_priority):
            score += 1

    # Additional scoring for title/description context
    if any(term in title_lower for term in high_priority):
        score += 2
    if any(term in desc_lower for term in high_priority):
        score += 2

    return min(score, 10)  # Cap at 10


def fetch_gcp_release_notes(request):
    """
    Cloud Function entry point - fetches GCP release notes from RSS feeds
    and loads to BigQuery

    Args:
        request: HTTP request object (can contain force_refresh flag)

    Returns:
        JSON response with status and record count
    """

    # Initialize BigQuery client
    bq_client = bigquery.Client()

    # Get configuration from environment
    project_id = os.environ.get('PROJECT_ID', 'mgm-digitalconcierge')
    dataset_id = os.environ.get('BQ_DATASET_ID', 'security_insights')

    print(f"Starting GCP release notes RSS feed refresh for project: {project_id}")

    # GCP Release Notes RSS Feeds
    rss_feeds = [
        {
            'url': 'https://cloud.google.com/feeds/gcp-release-notes.xml',
            'source': 'gcp_general',
            'name': 'Google Cloud Platform General Release Notes'
        },
        {
            'url': 'https://cloud.google.com/feeds/compute-release-notes.xml',
            'source': 'gcp_compute',
            'name': 'Google Compute Engine Release Notes'
        },
        {
            'url': 'https://cloud.google.com/feeds/gke-release-notes.xml',
            'source': 'gcp_gke',
            'name': 'Google Kubernetes Engine Release Notes'
        },
        {
            'url': 'https://cloud.google.com/feeds/iam-release-notes.xml',
            'source': 'gcp_iam',
            'name': 'Google Cloud IAM Release Notes'
        },
        {
            'url': 'https://cloud.google.com/feeds/security-command-center-release-notes.xml',
            'source': 'gcp_scc',
            'name': 'Security Command Center Release Notes'
        }
    ]

    all_releases = []

    try:
        for feed_config in rss_feeds:
            try:
                print(f"Fetching feed: {feed_config['name']}")

                # Fetch RSS feed
                response = requests.get(feed_config['url'], timeout=30)
                response.raise_for_status()

                # Parse RSS feed
                feed = feedparser.parse(response.content)

                for entry in feed.entries:
                    # Create unique ID for deduplication
                    entry_id = hashlib.md5(
                        f"{entry.link}_{entry.title}".encode()
                    ).hexdigest()

                    # Parse publish date
                    published_date = None
                    if hasattr(entry, 'published'):
                        try:
                            published_date = date_parser.parse(entry.published).isoformat()
                        except:
                            pass

                    # Extract and clean description
                    description = ""
                    if hasattr(entry, 'description'):
                        # Remove HTML tags
                        description = re.sub('<[^<]+?>', '', entry.description)
                        description = description.strip()
                    elif hasattr(entry, 'summary'):
                        description = re.sub('<[^<]+?>', '', entry.summary)
                        description = description.strip()

                    # Extract security keywords
                    security_keywords = extract_security_keywords(f"{entry.title} {description}")

                    # Calculate security relevance score
                    security_score = calculate_security_score(entry.title, description, security_keywords)

                    # Categorize the service
                    service_category = categorize_service(entry.title, description)

                    release_record = {
                        'entry_id': entry_id,
                        'title': entry.title,
                        'description': description[:4000],  # Limit description length
                        'link': entry.link,
                        'source_feed': feed_config['source'],
                        'feed_name': feed_config['name'],
                        'published_date': published_date,
                        'service_category': service_category,
                        'security_keywords': security_keywords,
                        'security_score': security_score,
                        'is_security_related': security_score >= 3,
                        'created_at': datetime.utcnow().isoformat(),
                        'last_refreshed': datetime.utcnow().isoformat(),
                        'refresh_job': 'scheduled_4h'
                    }

                    all_releases.append(release_record)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching feed {feed_config['url']}: {e}")
                continue
            except Exception as e:
                print(f"Error processing feed {feed_config['name']}: {e}")
                continue

        # Load data to BigQuery
        if all_releases:
            table_id = f"{project_id}.{dataset_id}.gcp_release_notes"

            # Define schema for the table
            schema = [
                bigquery.SchemaField("entry_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("title", "STRING"),
                bigquery.SchemaField("description", "STRING"),
                bigquery.SchemaField("link", "STRING"),
                bigquery.SchemaField("source_feed", "STRING"),
                bigquery.SchemaField("feed_name", "STRING"),
                bigquery.SchemaField("published_date", "TIMESTAMP"),
                bigquery.SchemaField("service_category", "STRING"),
                bigquery.SchemaField("security_keywords", "STRING", mode="REPEATED"),
                bigquery.SchemaField("security_score", "INTEGER"),
                bigquery.SchemaField("is_security_related", "BOOLEAN"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
                bigquery.SchemaField("last_refreshed", "TIMESTAMP"),
                bigquery.SchemaField("refresh_job", "STRING"),
            ]

            # Configure load job with MERGE behavior for deduplication
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition="WRITE_APPEND",  # Append new data
                create_disposition="CREATE_IF_NEEDED",
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
            )

            # Load data
            job = bq_client.load_table_from_json(
                all_releases,
                table_id,
                job_config=job_config
            )
            job.result()  # Wait for job to complete

            # After loading, deduplicate by keeping the latest entry for each entry_id
            dedupe_query = f"""
            CREATE OR REPLACE TABLE `{table_id}` AS
            SELECT * EXCEPT(row_num)
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY entry_id
                        ORDER BY last_refreshed DESC
                    ) as row_num
                FROM `{table_id}`
            )
            WHERE row_num = 1
            """

            dedupe_job = bq_client.query(dedupe_query)
            dedupe_job.result()

            print(f"Successfully loaded {len(all_releases)} release notes to BigQuery")

            # Log refresh metadata
            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            metadata_record = [{
                'table_name': 'gcp_release_notes',
                'refresh_time': datetime.utcnow().isoformat(),
                'record_count': len(all_releases),
                'status': 'success',
                'refresh_type': 'scheduled',
                'details': json.dumps({
                    'feeds_processed': len(rss_feeds),
                    'security_related_count': sum(1 for r in all_releases if r['is_security_related'])
                }),
                'error_message': None
            }]

            try:
                metadata_job = bq_client.load_table_from_json(
                    metadata_record,
                    metadata_table_id,
                    job_config=bigquery.LoadJobConfig(
                        write_disposition="WRITE_APPEND",
                        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
                    )
                )
                metadata_job.result()
            except Exception as e:
                print(f"Warning: Could not update refresh metadata: {e}")

            return {
                'status': 'success',
                'records': len(all_releases),
                'security_related': sum(1 for r in all_releases if r['is_security_related']),
                'feeds_processed': len(rss_feeds),
                'table': table_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            print("No release notes found to load")
            return {
                'status': 'success',
                'records': 0,
                'message': 'No release notes found',
                'timestamp': datetime.utcnow().isoformat()
            }

    except Exception as e:
        error_msg = f"Error in fetch_gcp_release_notes: {str(e)}"
        print(error_msg)

        # Try to log error to metadata table
        try:
            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            error_record = [{
                'table_name': 'gcp_release_notes',
                'refresh_time': datetime.utcnow().isoformat(),
                'record_count': 0,
                'status': 'failed',
                'refresh_type': 'scheduled',
                'error_message': str(e)[:1000]
            }]

            bq_client.load_table_from_json(
                error_record,
                metadata_table_id,
                job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            ).result()
        except:
            pass  # Silent fail on metadata logging

        return {
            'status': 'error',
            'error': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        }, 500


# For local testing
if __name__ == "__main__":
    class MockRequest:
        def __init__(self):
            self.json = {}

    result = fetch_gcp_release_notes(MockRequest())
    print(json.dumps(result, indent=2))
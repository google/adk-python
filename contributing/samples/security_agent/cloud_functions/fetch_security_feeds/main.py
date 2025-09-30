#!/usr/bin/env python3
"""
Cloud Function to fetch security feeds (CVE, advisories, threat intelligence)
Runs independently on a schedule (every 2 hours for critical security updates)
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


def parse_cvss_score(text: str) -> float:
    """Extract CVSS score from text"""
    if not text:
        return 0.0

    # Look for CVSS score patterns
    cvss_patterns = [
        r'cvss[:\s]*(\d+\.?\d*)',
        r'score[:\s]*(\d+\.?\d*)',
        r'(\d+\.?\d*)[:\s]*cvss'
    ]

    text_lower = text.lower()
    for pattern in cvss_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
            except ValueError:
                continue

    return 0.0


def extract_cve_ids(text: str) -> List[str]:
    """Extract CVE IDs from text"""
    if not text:
        return []

    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    cve_ids = re.findall(cve_pattern, text, re.IGNORECASE)
    return list(set(cve_ids))


def categorize_threat_type(title: str, description: str) -> str:
    """Categorize the type of security threat"""
    content = f"{title} {description}".lower()

    threat_categories = {
        'vulnerability': ['vulnerability', 'cve', 'security flaw', 'exploit'],
        'malware': ['malware', 'virus', 'trojan', 'ransomware', 'backdoor'],
        'phishing': ['phishing', 'social engineering', 'credential theft'],
        'ddos': ['ddos', 'denial of service', 'amplification'],
        'data_breach': ['data breach', 'leak', 'exposure', 'unauthorized access'],
        'supply_chain': ['supply chain', 'third party', 'dependency'],
        'zero_day': ['zero day', 'zero-day', '0-day'],
        'apt': ['apt', 'advanced persistent threat', 'nation state'],
        'botnet': ['botnet', 'bot network', 'zombie'],
        'insider_threat': ['insider threat', 'rogue employee', 'privileged access']
    }

    for category, keywords in threat_categories.items():
        if any(keyword in content for keyword in keywords):
            return category

    return 'other'


def calculate_threat_severity(cvss_score: float, title: str, description: str, cve_count: int) -> str:
    """Calculate threat severity based on various factors"""
    content = f"{title} {description}".lower()

    # CVSS-based severity
    if cvss_score >= 9.0:
        base_severity = 'critical'
    elif cvss_score >= 7.0:
        base_severity = 'high'
    elif cvss_score >= 4.0:
        base_severity = 'medium'
    elif cvss_score > 0:
        base_severity = 'low'
    else:
        # Text-based severity detection
        if any(term in content for term in ['critical', 'emergency', 'urgent', 'zero day']):
            base_severity = 'critical'
        elif any(term in content for term in ['high', 'severe', 'important']):
            base_severity = 'high'
        elif any(term in content for term in ['medium', 'moderate']):
            base_severity = 'medium'
        else:
            base_severity = 'low'

    # Boost severity for multiple CVEs
    if cve_count > 3 and base_severity == 'medium':
        base_severity = 'high'
    elif cve_count > 1 and base_severity == 'low':
        base_severity = 'medium'

    return base_severity


def fetch_security_feeds(request):
    """
    Cloud Function entry point - fetches security feeds from multiple sources
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

    print(f"Starting security feeds refresh for project: {project_id}")

    # Security Feed Sources
    security_feeds = [
        {
            'url': 'https://nvd.nist.gov/feeds/xml/cve/misc/nvd-rss.xml',
            'source': 'nvd_cve',
            'name': 'NIST National Vulnerability Database',
            'feed_type': 'cve'
        },
        {
            'url': 'https://tools.cisco.com/security/center/psirt_rss20.xml',
            'source': 'cisco_psirt',
            'name': 'Cisco Product Security Incident Response Team',
            'feed_type': 'vendor_advisory'
        },
        {
            'url': 'https://support.microsoft.com/app/content/api/content/feeds/smc-en-us-rss',
            'source': 'microsoft_security',
            'name': 'Microsoft Security Response Center',
            'feed_type': 'vendor_advisory'
        },
        {
            'url': 'https://www.us-cert.gov/ncas/current-activity.xml',
            'source': 'us_cert',
            'name': 'US-CERT Current Activity',
            'feed_type': 'government_advisory'
        },
        {
            'url': 'https://feeds.feedburner.com/TheHackersNews',
            'source': 'hacker_news',
            'name': 'The Hacker News',
            'feed_type': 'news'
        },
        {
            'url': 'https://feeds.fortinet.com/fortinet/blog/threat-research',
            'source': 'fortinet_threat',
            'name': 'Fortinet Threat Research',
            'feed_type': 'threat_intelligence'
        },
        {
            'url': 'https://blog.rapid7.com/rss/',
            'source': 'rapid7_blog',
            'name': 'Rapid7 Security Blog',
            'feed_type': 'threat_intelligence'
        }
    ]

    all_security_items = []

    try:
        for feed_config in security_feeds:
            try:
                print(f"Fetching security feed: {feed_config['name']}")

                # Fetch RSS feed with timeout
                response = requests.get(feed_config['url'], timeout=45)
                response.raise_for_status()

                # Parse RSS feed
                feed = feedparser.parse(response.content)

                for entry in feed.entries:
                    # Create unique ID for deduplication
                    entry_id = hashlib.md5(
                        f"{entry.link}_{entry.title}_{feed_config['source']}".encode()
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
                        description = re.sub('<[^<]+?>', '', entry.description)
                        description = description.strip()
                    elif hasattr(entry, 'summary'):
                        description = re.sub('<[^<]+?>', '', entry.summary)
                        description = description.strip()

                    # Extract security-specific information
                    cvss_score = parse_cvss_score(f"{entry.title} {description}")
                    cve_ids = extract_cve_ids(f"{entry.title} {description}")
                    threat_type = categorize_threat_type(entry.title, description)
                    severity = calculate_threat_severity(cvss_score, entry.title, description, len(cve_ids))

                    # Determine if this is cloud-related
                    cloud_keywords = ['cloud', 'aws', 'azure', 'gcp', 'google cloud', 'kubernetes', 'docker', 'container']
                    is_cloud_related = any(keyword in f"{entry.title} {description}".lower() for keyword in cloud_keywords)

                    # Extract affected products/vendors
                    affected_products = []
                    if 'cisco' in feed_config['source']:
                        affected_products.append('cisco')
                    elif 'microsoft' in feed_config['source']:
                        affected_products.append('microsoft')

                    # Additional product detection from content
                    product_keywords = {
                        'google': ['google', 'gcp', 'google cloud'],
                        'microsoft': ['microsoft', 'azure', 'windows'],
                        'amazon': ['amazon', 'aws'],
                        'kubernetes': ['kubernetes', 'k8s'],
                        'docker': ['docker', 'container'],
                        'linux': ['linux', 'ubuntu', 'centos', 'rhel'],
                        'apache': ['apache'],
                        'nginx': ['nginx']
                    }

                    content_lower = f"{entry.title} {description}".lower()
                    for product, keywords in product_keywords.items():
                        if any(keyword in content_lower for keyword in keywords):
                            affected_products.append(product)

                    security_record = {
                        'entry_id': entry_id,
                        'title': entry.title,
                        'description': description[:4000],  # Limit description length
                        'link': entry.link,
                        'source_feed': feed_config['source'],
                        'feed_name': feed_config['name'],
                        'feed_type': feed_config['feed_type'],
                        'published_date': published_date,
                        'cvss_score': cvss_score,
                        'cve_ids': cve_ids,
                        'cve_count': len(cve_ids),
                        'threat_type': threat_type,
                        'severity': severity,
                        'is_cloud_related': is_cloud_related,
                        'affected_products': list(set(affected_products)),
                        'requires_immediate_action': severity in ['critical', 'high'] and len(cve_ids) > 0,
                        'created_at': datetime.utcnow().isoformat(),
                        'last_refreshed': datetime.utcnow().isoformat(),
                        'refresh_job': 'scheduled_2h'
                    }

                    all_security_items.append(security_record)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching feed {feed_config['url']}: {e}")
                continue
            except Exception as e:
                print(f"Error processing feed {feed_config['name']}: {e}")
                continue

        # Load data to BigQuery
        if all_security_items:
            table_id = f"{project_id}.{dataset_id}.security_threat_feeds"

            # Define schema for the table
            schema = [
                bigquery.SchemaField("entry_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("title", "STRING"),
                bigquery.SchemaField("description", "STRING"),
                bigquery.SchemaField("link", "STRING"),
                bigquery.SchemaField("source_feed", "STRING"),
                bigquery.SchemaField("feed_name", "STRING"),
                bigquery.SchemaField("feed_type", "STRING"),
                bigquery.SchemaField("published_date", "TIMESTAMP"),
                bigquery.SchemaField("cvss_score", "FLOAT"),
                bigquery.SchemaField("cve_ids", "STRING", mode="REPEATED"),
                bigquery.SchemaField("cve_count", "INTEGER"),
                bigquery.SchemaField("threat_type", "STRING"),
                bigquery.SchemaField("severity", "STRING"),
                bigquery.SchemaField("is_cloud_related", "BOOLEAN"),
                bigquery.SchemaField("affected_products", "STRING", mode="REPEATED"),
                bigquery.SchemaField("requires_immediate_action", "BOOLEAN"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
                bigquery.SchemaField("last_refreshed", "TIMESTAMP"),
                bigquery.SchemaField("refresh_job", "STRING"),
            ]

            # Configure load job
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition="WRITE_APPEND",
                create_disposition="CREATE_IF_NEEDED",
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
            )

            # Load data
            job = bq_client.load_table_from_json(
                all_security_items,
                table_id,
                job_config=job_config
            )
            job.result()  # Wait for job to complete

            # Deduplicate by keeping the latest entry for each entry_id
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

            print(f"Successfully loaded {len(all_security_items)} security items to BigQuery")

            # Log refresh metadata with detailed statistics
            critical_count = sum(1 for item in all_security_items if item['severity'] == 'critical')
            high_count = sum(1 for item in all_security_items if item['severity'] == 'high')
            cve_count = sum(1 for item in all_security_items if item['cve_count'] > 0)
            cloud_related_count = sum(1 for item in all_security_items if item['is_cloud_related'])

            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            metadata_record = [{
                'table_name': 'security_threat_feeds',
                'refresh_time': datetime.utcnow().isoformat(),
                'record_count': len(all_security_items),
                'status': 'success',
                'refresh_type': 'scheduled',
                'details': json.dumps({
                    'feeds_processed': len(security_feeds),
                    'critical_threats': critical_count,
                    'high_threats': high_count,
                    'cve_items': cve_count,
                    'cloud_related': cloud_related_count
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
                'records': len(all_security_items),
                'critical_threats': critical_count,
                'high_threats': high_count,
                'cve_items': cve_count,
                'cloud_related': cloud_related_count,
                'feeds_processed': len(security_feeds),
                'table': table_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            print("No security items found to load")
            return {
                'status': 'success',
                'records': 0,
                'message': 'No security items found',
                'timestamp': datetime.utcnow().isoformat()
            }

    except Exception as e:
        error_msg = f"Error in fetch_security_feeds: {str(e)}"
        print(error_msg)

        # Try to log error to metadata table
        try:
            metadata_table_id = f"{project_id}.{dataset_id}.refresh_metadata"
            error_record = [{
                'table_name': 'security_threat_feeds',
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

    result = fetch_security_feeds(MockRequest())
    print(json.dumps(result, indent=2))
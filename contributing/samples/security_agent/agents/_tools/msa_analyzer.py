"""
MSA (Multi-Service Analyzer) - Release Notes Impact Assessment
Monitors GCP release notes and analyzes security, billing, and compliance impacts
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import feedparser
from google.cloud import bigquery
from google.cloud import storage
import hashlib

logger = logging.getLogger(__name__)

class MSAAnalyzer:
    """
    Multi-Service Analyzer for GCP release notes and service changes.
    Monitors official GCP feeds and assesses impacts on security, billing, and compliance.
    """

    def __init__(self, project_id: str = None):
        """Initialize the MSA Analyzer"""
        import os
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')

        # Initialize clients
        self.bq_client = None
        self.storage_client = None

        try:
            self.bq_client = bigquery.Client(project=self.project_id)
            logger.info(f"✅ BigQuery client initialized for MSA")
        except Exception as e:
            logger.warning(f"BigQuery not available for MSA: {e}")

        try:
            self.storage_client = storage.Client(project=self.project_id)
            logger.info(f"✅ Storage client initialized for MSA")
        except Exception as e:
            logger.warning(f"Storage not available for MSA: {e}")

        # GCP Release Notes feed
        self.release_notes_url = "https://cloud.google.com/feeds/gcp-release-notes.xml"

        # Cache for processed items
        self.processed_cache = set()
        self._load_processed_cache()

        # Active services (would come from asset inventory in production)
        self.active_services = self._get_active_services()

    def analyze_release_notes(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Main entry point: Analyze recent GCP release notes for impacts.

        Args:
            days_back: Number of days to look back for release notes

        Returns:
            Comprehensive impact analysis report
        """
        logger.info(f"🔍 Analyzing GCP release notes from last {days_back} days")

        # Step 1: Fetch release notes
        release_notes = self._fetch_release_notes(days_back)

        # Step 2: Filter for active services only
        relevant_notes = self._filter_active_services(release_notes)

        # Step 3: Categorize changes
        categorized_changes = self._categorize_changes(relevant_notes)

        # Step 4: Analyze security impact
        security_impact = self._analyze_security_impact(categorized_changes)

        # Step 5: Analyze billing impact
        billing_impact = self._analyze_billing_impact(categorized_changes)

        # Step 6: Analyze compliance impact
        compliance_impact = self._analyze_compliance_impact(categorized_changes)

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            security_impact,
            billing_impact,
            compliance_impact
        )

        # Step 8: Create analysis report
        report = self._create_analysis_report(
            relevant_notes,
            categorized_changes,
            security_impact,
            billing_impact,
            compliance_impact,
            recommendations
        )

        # Step 9: Store in BigQuery
        self._store_analysis(report)

        return report

    def _fetch_release_notes(self, days_back: int) -> List[Dict]:
        """Fetch release notes from GCP RSS feed"""
        release_notes = []

        try:
            # Parse RSS feed
            feed = feedparser.parse(self.release_notes_url)

            cutoff_date = datetime.now() - timedelta(days=days_back)

            for entry in feed.entries:
                # Parse publication date
                pub_date = datetime(*entry.published_parsed[:6])

                if pub_date >= cutoff_date:
                    # Extract service name from title or tags
                    service = self._extract_service_name(entry)

                    # Parse content for details
                    content = entry.get('summary', entry.get('description', ''))

                    note = {
                        'id': entry.get('id', hashlib.md5(entry.title.encode()).hexdigest()),
                        'title': entry.title,
                        'service': service,
                        'published': pub_date.isoformat(),
                        'link': entry.link,
                        'content': content,
                        'tags': [tag.term for tag in entry.get('tags', [])],
                        'category': self._determine_change_category(content),
                        'processed': False
                    }

                    # Check if already processed
                    if note['id'] not in self.processed_cache:
                        release_notes.append(note)

            logger.info(f"📰 Fetched {len(release_notes)} new release notes")

        except Exception as e:
            logger.error(f"Error fetching release notes: {e}")

            # Fallback to web scraping if RSS fails
            release_notes = self._fetch_release_notes_web(days_back)

        return release_notes

    def _fetch_release_notes_web(self, days_back: int) -> List[Dict]:
        """Fallback: Fetch release notes via web scraping"""
        release_notes = []

        try:
            url = "https://cloud.google.com/release-notes"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find release note entries
            entries = soup.find_all('div', class_='release-note-entry') or \
                     soup.find_all('article', class_='devsite-article')

            cutoff_date = datetime.now() - timedelta(days=days_back)

            for entry in entries[:50]:  # Limit to recent entries
                title_elem = entry.find('h2') or entry.find('h3')
                if not title_elem:
                    continue

                title = title_elem.text.strip()
                service = self._extract_service_name_from_text(title)

                # Try to find date
                date_elem = entry.find('time') or entry.find('span', class_='date')
                if date_elem:
                    try:
                        pub_date = datetime.strptime(date_elem.text.strip(), '%Y-%m-%d')
                        if pub_date < cutoff_date:
                            continue
                    except:
                        pub_date = datetime.now()
                else:
                    pub_date = datetime.now()

                content = entry.text.strip()

                note = {
                    'id': hashlib.md5(title.encode()).hexdigest(),
                    'title': title,
                    'service': service,
                    'published': pub_date.isoformat(),
                    'link': url,
                    'content': content,
                    'tags': [],
                    'category': self._determine_change_category(content),
                    'processed': False
                }

                if note['id'] not in self.processed_cache:
                    release_notes.append(note)

        except Exception as e:
            logger.error(f"Error with web scraping fallback: {e}")

        return release_notes

    def _extract_service_name(self, entry) -> str:
        """Extract service name from RSS entry"""
        # Try tags first
        for tag in entry.get('tags', []):
            if 'product' in tag.term.lower():
                return tag.term.replace('product/', '')

        # Try title
        return self._extract_service_name_from_text(entry.title)

    def _extract_service_name_from_text(self, text: str) -> str:
        """Extract service name from text"""
        text_lower = text.lower()

        # Common GCP service patterns
        services = {
            'bigquery': 'BigQuery',
            'cloud storage': 'Cloud Storage',
            'compute engine': 'Compute Engine',
            'cloud run': 'Cloud Run',
            'cloud functions': 'Cloud Functions',
            'vertex ai': 'Vertex AI',
            'cloud sql': 'Cloud SQL',
            'gke': 'Google Kubernetes Engine',
            'pub/sub': 'Pub/Sub',
            'firestore': 'Firestore',
            'spanner': 'Cloud Spanner',
            'dataflow': 'Dataflow',
            'cloud kms': 'Cloud KMS',
            'secret manager': 'Secret Manager',
            'vpc': 'VPC',
            'cloud armor': 'Cloud Armor',
            'identity platform': 'Identity Platform'
        }

        for pattern, service_name in services.items():
            if pattern in text_lower:
                return service_name

        # Generic extraction
        if ':' in text:
            return text.split(':')[0].strip()

        return 'Unknown Service'

    def _determine_change_category(self, content: str) -> str:
        """Determine the category of change from content"""
        content_lower = content.lower()

        if any(word in content_lower for word in ['security', 'encryption', 'auth', 'iam', 'permission', 'vulnerability', 'cve']):
            return 'security'
        elif any(word in content_lower for word in ['pricing', 'cost', 'billing', 'price', 'free tier', 'discount']):
            return 'billing'
        elif any(word in content_lower for word in ['deprecat', 'sunset', 'end of life', 'eol', 'discontinu']):
            return 'deprecation'
        elif any(word in content_lower for word in ['compliance', 'certification', 'audit', 'gdpr', 'hipaa', 'pci', 'sox']):
            return 'compliance'
        elif any(word in content_lower for word in ['performance', 'speed', 'latency', 'throughput', 'optimization']):
            return 'performance'
        elif any(word in content_lower for word in ['new feature', 'launch', 'introduce', 'announce', 'preview', 'beta', 'alpha']):
            return 'feature'
        elif any(word in content_lower for word in ['bug', 'fix', 'patch', 'resolve', 'issue']):
            return 'bugfix'
        else:
            return 'general'

    def _get_active_services(self) -> Set[str]:
        """Get list of services currently in use"""
        active_services = set()

        if self.bq_client:
            try:
                query = """
                SELECT DISTINCT service_name
                FROM `{}.security_data.active_services`
                WHERE status = 'active'
                """.format(self.project_id)

                results = self.bq_client.query(query)
                for row in results:
                    active_services.add(row.service_name.lower())

            except Exception as e:
                logger.warning(f"Could not fetch active services: {e}")

        # Fallback to common services if no BigQuery data
        if not active_services:
            active_services = {
                'bigquery', 'cloud storage', 'compute engine',
                'cloud run', 'cloud functions', 'cloud sql',
                'pub/sub', 'vertex ai', 'cloud kms'
            }

        logger.info(f"📊 Tracking {len(active_services)} active services")
        return active_services

    def _filter_active_services(self, release_notes: List[Dict]) -> List[Dict]:
        """Filter release notes to only include active services"""
        relevant_notes = []

        for note in release_notes:
            service_lower = note['service'].lower()

            # Check if service is active
            if any(active_svc in service_lower for active_svc in self.active_services):
                relevant_notes.append(note)
            # Also include critical security updates regardless
            elif note['category'] == 'security' and 'critical' in note['content'].lower():
                relevant_notes.append(note)

        logger.info(f"🎯 Filtered to {len(relevant_notes)} relevant notes for active services")
        return relevant_notes

    def _categorize_changes(self, release_notes: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize release notes by type of change"""
        categories = {
            'security': [],
            'billing': [],
            'deprecation': [],
            'compliance': [],
            'feature': [],
            'performance': [],
            'bugfix': [],
            'general': []
        }

        for note in release_notes:
            category = note['category']
            if category in categories:
                categories[category].append(note)
            else:
                categories['general'].append(note)

        # Log summary
        for cat, notes in categories.items():
            if notes:
                logger.info(f"  {cat}: {len(notes)} changes")

        return categories

    def _analyze_security_impact(self, categorized_changes: Dict) -> Dict[str, Any]:
        """Analyze security impact of changes"""
        security_impact = {
            'risk_level': 'low',
            'critical_updates': [],
            'authentication_changes': [],
            'encryption_changes': [],
            'permission_changes': [],
            'vulnerabilities_fixed': [],
            'action_required': []
        }

        security_notes = categorized_changes.get('security', [])

        for note in security_notes:
            content_lower = note['content'].lower()

            # Check for critical security updates
            if any(word in content_lower for word in ['critical', 'urgent', 'immediately']):
                security_impact['critical_updates'].append({
                    'service': note['service'],
                    'title': note['title'],
                    'link': note['link']
                })
                security_impact['risk_level'] = 'high'
                security_impact['action_required'].append(f"Review critical update for {note['service']}")

            # Check for authentication changes
            if any(word in content_lower for word in ['authentication', 'oauth', 'saml', 'identity']):
                security_impact['authentication_changes'].append({
                    'service': note['service'],
                    'description': note['title']
                })
                security_impact['action_required'].append(f"Verify authentication still works for {note['service']}")

            # Check for encryption changes
            if any(word in content_lower for word in ['encryption', 'tls', 'ssl', 'cipher']):
                security_impact['encryption_changes'].append({
                    'service': note['service'],
                    'description': note['title']
                })
                if security_impact['risk_level'] == 'low':
                    security_impact['risk_level'] = 'medium'

            # Check for permission/IAM changes
            if any(word in content_lower for word in ['permission', 'iam', 'role', 'policy']):
                security_impact['permission_changes'].append({
                    'service': note['service'],
                    'description': note['title']
                })
                security_impact['action_required'].append(f"Review IAM changes for {note['service']}")

            # Check for vulnerability fixes
            if any(word in content_lower for word in ['cve', 'vulnerability', 'patch', 'security fix']):
                security_impact['vulnerabilities_fixed'].append({
                    'service': note['service'],
                    'description': note['title']
                })

        # Also check deprecation for security implications
        for note in categorized_changes.get('deprecation', []):
            if 'security' in note['content'].lower():
                security_impact['action_required'].append(
                    f"Deprecated security feature in {note['service']} - migration required"
                )
                if security_impact['risk_level'] == 'low':
                    security_impact['risk_level'] = 'medium'

        return security_impact

    def _analyze_billing_impact(self, categorized_changes: Dict) -> Dict[str, Any]:
        """Analyze billing/cost impact of changes"""
        billing_impact = {
            'estimated_impact': 'neutral',
            'pricing_changes': [],
            'free_tier_changes': [],
            'new_charges': [],
            'discontinued_discounts': [],
            'cost_optimization_opportunities': [],
            'action_required': []
        }

        billing_notes = categorized_changes.get('billing', [])

        for note in billing_notes:
            content_lower = note['content'].lower()

            # Check for price increases
            if any(word in content_lower for word in ['increase', 'raise', 'higher']):
                billing_impact['pricing_changes'].append({
                    'service': note['service'],
                    'type': 'increase',
                    'description': note['title']
                })
                billing_impact['estimated_impact'] = 'increase'
                billing_impact['action_required'].append(f"Review budget impact for {note['service']}")

            # Check for price decreases
            elif any(word in content_lower for word in ['decrease', 'reduce', 'lower', 'discount']):
                billing_impact['pricing_changes'].append({
                    'service': note['service'],
                    'type': 'decrease',
                    'description': note['title']
                })
                billing_impact['cost_optimization_opportunities'].append(
                    f"Cost savings available in {note['service']}"
                )
                if billing_impact['estimated_impact'] == 'neutral':
                    billing_impact['estimated_impact'] = 'decrease'

            # Check for free tier changes
            if 'free tier' in content_lower or 'free usage' in content_lower:
                billing_impact['free_tier_changes'].append({
                    'service': note['service'],
                    'description': note['title']
                })
                billing_impact['action_required'].append(f"Review free tier usage for {note['service']}")

            # Check for new charges
            if any(word in content_lower for word in ['new charge', 'now charge', 'will charge', 'fee']):
                billing_impact['new_charges'].append({
                    'service': note['service'],
                    'description': note['title']
                })
                billing_impact['estimated_impact'] = 'increase'

        # Check deprecations for billing impact
        for note in categorized_changes.get('deprecation', []):
            if note['service'] in [pc['service'] for pc in billing_impact['pricing_changes']]:
                continue
            billing_impact['action_required'].append(
                f"Service deprecation for {note['service']} - may require migration costs"
            )

        return billing_impact

    def _analyze_compliance_impact(self, categorized_changes: Dict) -> Dict[str, Any]:
        """Analyze compliance and regulatory impact"""
        compliance_impact = {
            'impact_level': 'low',
            'new_certifications': [],
            'lost_certifications': [],
            'regulation_changes': [],
            'audit_requirements': [],
            'data_residency_changes': [],
            'action_required': []
        }

        compliance_notes = categorized_changes.get('compliance', [])

        for note in compliance_notes:
            content_lower = note['content'].lower()

            # Check for new certifications
            if any(word in content_lower for word in ['certified', 'achieved', 'compliant', 'attestation']):
                cert_type = 'unknown'
                for cert in ['soc2', 'iso', 'hipaa', 'pci', 'gdpr', 'ccpa']:
                    if cert in content_lower:
                        cert_type = cert.upper()
                        break

                compliance_impact['new_certifications'].append({
                    'service': note['service'],
                    'certification': cert_type,
                    'description': note['title']
                })

            # Check for lost certifications or compliance issues
            if any(word in content_lower for word in ['no longer', 'discontinued', 'withdrawn']):
                compliance_impact['lost_certifications'].append({
                    'service': note['service'],
                    'description': note['title']
                })
                compliance_impact['impact_level'] = 'high'
                compliance_impact['action_required'].append(
                    f"Urgent: Review compliance status for {note['service']}"
                )

            # Check for regulation changes
            if any(word in content_lower for word in ['regulation', 'requirement', 'mandate']):
                compliance_impact['regulation_changes'].append({
                    'service': note['service'],
                    'description': note['title']
                })
                if compliance_impact['impact_level'] == 'low':
                    compliance_impact['impact_level'] = 'medium'

            # Check for data residency
            if any(word in content_lower for word in ['data residency', 'data location', 'region', 'sovereignty']):
                compliance_impact['data_residency_changes'].append({
                    'service': note['service'],
                    'description': note['title']
                })
                compliance_impact['action_required'].append(
                    f"Review data residency requirements for {note['service']}"
                )

        # Check security changes for compliance impact
        for note in categorized_changes.get('security', []):
            if any(word in note['content'].lower() for word in ['compliance', 'audit', 'regulation']):
                compliance_impact['audit_requirements'].append({
                    'service': note['service'],
                    'description': note['title']
                })

        return compliance_impact

    def _generate_recommendations(self, security_impact: Dict,
                                 billing_impact: Dict,
                                 compliance_impact: Dict) -> List[Dict]:
        """Generate actionable recommendations based on all impacts"""
        recommendations = []

        # Priority 1: Critical security issues
        if security_impact['critical_updates']:
            for update in security_impact['critical_updates']:
                recommendations.append({
                    'priority': 'critical',
                    'category': 'security',
                    'action': f"Apply critical security update for {update['service']}",
                    'deadline': 'immediate',
                    'link': update['link']
                })

        # Priority 2: Compliance issues
        if compliance_impact['lost_certifications']:
            for cert in compliance_impact['lost_certifications']:
                recommendations.append({
                    'priority': 'high',
                    'category': 'compliance',
                    'action': f"Address compliance gap in {cert['service']}",
                    'deadline': '7 days',
                    'details': cert['description']
                })

        # Priority 3: Cost increases
        if billing_impact['estimated_impact'] == 'increase':
            recommendations.append({
                'priority': 'medium',
                'category': 'billing',
                'action': 'Review and adjust budget forecasts',
                'deadline': '30 days',
                'details': f"{len(billing_impact['pricing_changes'])} services with pricing changes"
            })

        # Priority 4: Authentication/permission changes
        if security_impact['authentication_changes'] or security_impact['permission_changes']:
            recommendations.append({
                'priority': 'medium',
                'category': 'security',
                'action': 'Test authentication and permissions',
                'deadline': '14 days',
                'details': 'Verify access controls still function correctly'
            })

        # Priority 5: Deprecations
        deprecation_count = len([n for n in security_impact.get('action_required', [])
                               if 'deprecated' in n.lower()])
        if deprecation_count > 0:
            recommendations.append({
                'priority': 'low',
                'category': 'migration',
                'action': f"Plan migration for {deprecation_count} deprecated features",
                'deadline': '90 days',
                'details': 'Create migration timeline before end-of-life'
            })

        # Add cost optimization opportunities
        if billing_impact['cost_optimization_opportunities']:
            recommendations.append({
                'priority': 'low',
                'category': 'optimization',
                'action': 'Review cost optimization opportunities',
                'deadline': '60 days',
                'details': ', '.join(billing_impact['cost_optimization_opportunities'])
            })

        return recommendations

    def _create_analysis_report(self, relevant_notes: List[Dict],
                               categorized_changes: Dict,
                               security_impact: Dict,
                               billing_impact: Dict,
                               compliance_impact: Dict,
                               recommendations: List[Dict]) -> Dict[str, Any]:
        """Create comprehensive MSA analysis report"""

        # Calculate overall risk score
        risk_score = 0
        if security_impact['risk_level'] == 'high':
            risk_score += 40
        elif security_impact['risk_level'] == 'medium':
            risk_score += 20

        if billing_impact['estimated_impact'] == 'increase':
            risk_score += 20

        if compliance_impact['impact_level'] == 'high':
            risk_score += 30
        elif compliance_impact['impact_level'] == 'medium':
            risk_score += 15

        risk_level = 'low' if risk_score < 30 else 'medium' if risk_score < 60 else 'high'

        report = {
            'analysis_id': hashlib.md5(
                f"msa-{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_changes_analyzed': len(relevant_notes),
                'active_services_affected': len(set(n['service'] for n in relevant_notes)),
                'overall_risk_score': risk_score,
                'overall_risk_level': risk_level,
                'critical_issues': len(security_impact['critical_updates']),
                'recommendations_count': len(recommendations)
            },
            'breakdown_by_category': {
                cat: len(notes) for cat, notes in categorized_changes.items() if notes
            },
            'security_impact': security_impact,
            'billing_impact': billing_impact,
            'compliance_impact': compliance_impact,
            'recommendations': sorted(
                recommendations,
                key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}.get(x['priority'], 4)
            ),
            'affected_services': list(set(n['service'] for n in relevant_notes)),
            'action_items': {
                'immediate': [r for r in recommendations if r['priority'] == 'critical'],
                'within_7_days': [r for r in recommendations if r['priority'] == 'high'],
                'within_30_days': [r for r in recommendations if r['priority'] == 'medium'],
                'within_90_days': [r for r in recommendations if r['priority'] == 'low']
            },
            'release_notes_analyzed': [
                {
                    'service': n['service'],
                    'title': n['title'],
                    'category': n['category'],
                    'published': n['published'],
                    'link': n['link']
                } for n in relevant_notes[:10]  # Include top 10 for reference
            ]
        }

        return report

    def _store_analysis(self, report: Dict[str, Any]):
        """Store MSA analysis in BigQuery"""
        if not self.bq_client:
            return

        try:
            dataset_id = f"{self.project_id}.security_data"
            table_id = f"{dataset_id}.msa_analysis_history"

            # Ensure dataset exists
            try:
                self.bq_client.get_dataset(dataset_id)
            except:
                dataset = bigquery.Dataset(dataset_id)
                dataset.location = "US"
                self.bq_client.create_dataset(dataset, timeout=30)

            # Prepare row for insertion
            row = {
                'analysis_id': report['analysis_id'],
                'timestamp': report['timestamp'],
                'total_changes': report['summary']['total_changes_analyzed'],
                'services_affected': report['summary']['active_services_affected'],
                'risk_score': report['summary']['overall_risk_score'],
                'risk_level': report['summary']['overall_risk_level'],
                'critical_issues': report['summary']['critical_issues'],
                'security_risk': report['security_impact']['risk_level'],
                'billing_impact': report['billing_impact']['estimated_impact'],
                'compliance_impact': report['compliance_impact']['impact_level'],
                'recommendations': json.dumps(report['recommendations']),
                'full_report': json.dumps(report)
            }

            errors = self.bq_client.insert_rows_json(
                table_id, [row], ignore_unknown_values=True
            )

            if errors:
                logger.error(f"Failed to store MSA analysis: {errors}")
            else:
                logger.info(f"✅ MSA analysis stored: {report['analysis_id']}")

        except Exception as e:
            logger.error(f"Error storing MSA analysis: {e}")

    def _load_processed_cache(self):
        """Load cache of already processed release notes"""
        if self.storage_client:
            try:
                bucket_name = f"{self.project_id}-msa-cache"
                blob_name = "processed_notes.json"

                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                if blob.exists():
                    cache_data = json.loads(blob.download_as_text())
                    self.processed_cache = set(cache_data.get('processed_ids', []))
                    logger.info(f"📚 Loaded {len(self.processed_cache)} processed notes from cache")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")

    def _save_processed_cache(self, new_ids: List[str]):
        """Save processed note IDs to cache"""
        if self.storage_client and new_ids:
            try:
                bucket_name = f"{self.project_id}-msa-cache"
                blob_name = "processed_notes.json"

                # Add new IDs to cache
                self.processed_cache.update(new_ids)

                # Keep only last 1000 IDs to prevent unlimited growth
                cache_list = list(self.processed_cache)[-1000:]

                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                cache_data = {
                    'processed_ids': cache_list,
                    'last_updated': datetime.now().isoformat()
                }

                blob.upload_from_string(json.dumps(cache_data))
                logger.info(f"💾 Saved {len(new_ids)} new IDs to cache")

            except Exception as e:
                logger.warning(f"Could not save cache: {e}")


# Cloud Function entry point
def analyze_releases(request):
    """
    Cloud Function entry point for automated MSA analysis.
    Can be triggered by Cloud Scheduler or HTTP request.
    """
    # Parse request for parameters
    request_json = request.get_json(silent=True)
    days_back = 7  # Default

    if request_json and 'days_back' in request_json:
        days_back = int(request_json['days_back'])

    # Run analysis
    analyzer = MSAAnalyzer()
    report = analyzer.analyze_release_notes(days_back)

    # Mark notes as processed
    processed_ids = [n['id'] for n in report.get('release_notes_analyzed', [])]
    analyzer._save_processed_cache(processed_ids)

    # Return summary
    return {
        'success': True,
        'analysis_id': report['analysis_id'],
        'summary': report['summary'],
        'top_recommendations': report['recommendations'][:3] if report['recommendations'] else []
    }


# ADK Tool function
def analyze_gcp_releases(days_back: int = 7) -> Dict[str, Any]:
    """
    ADK tool function for MSA analysis.

    Args:
        days_back: Number of days to analyze

    Returns:
        MSA analysis report
    """
    analyzer = MSAAnalyzer()
    return analyzer.analyze_release_notes(days_back)


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("   🔍 MSA (Multi-Service Analyzer) - Release Notes Impact")
    print("=" * 80)
    print()

    analyzer = MSAAnalyzer()

    # Analyze last 7 days
    report = analyzer.analyze_release_notes(days_back=7)

    print(f"\n📊 ANALYSIS SUMMARY")
    print(f"   Analysis ID: {report['analysis_id']}")
    print(f"   Changes Analyzed: {report['summary']['total_changes_analyzed']}")
    print(f"   Services Affected: {report['summary']['active_services_affected']}")
    print(f"   Overall Risk: {report['summary']['overall_risk_level'].upper()} (Score: {report['summary']['overall_risk_score']})")
    print(f"   Critical Issues: {report['summary']['critical_issues']}")

    print(f"\n🔒 SECURITY IMPACT")
    print(f"   Risk Level: {report['security_impact']['risk_level'].upper()}")
    if report['security_impact']['critical_updates']:
        print(f"   ⚠️ Critical Updates: {len(report['security_impact']['critical_updates'])}")
    if report['security_impact']['authentication_changes']:
        print(f"   🔐 Auth Changes: {len(report['security_impact']['authentication_changes'])}")
    if report['security_impact']['vulnerabilities_fixed']:
        print(f"   ✅ Vulnerabilities Fixed: {len(report['security_impact']['vulnerabilities_fixed'])}")

    print(f"\n💰 BILLING IMPACT")
    print(f"   Estimated Impact: {report['billing_impact']['estimated_impact'].upper()}")
    if report['billing_impact']['pricing_changes']:
        print(f"   💵 Pricing Changes: {len(report['billing_impact']['pricing_changes'])}")
    if report['billing_impact']['cost_optimization_opportunities']:
        print(f"   💡 Optimization Opportunities: {len(report['billing_impact']['cost_optimization_opportunities'])}")

    print(f"\n📋 COMPLIANCE IMPACT")
    print(f"   Impact Level: {report['compliance_impact']['impact_level'].upper()}")
    if report['compliance_impact']['new_certifications']:
        print(f"   ✅ New Certifications: {len(report['compliance_impact']['new_certifications'])}")
    if report['compliance_impact']['lost_certifications']:
        print(f"   ❌ Lost Certifications: {len(report['compliance_impact']['lost_certifications'])}")

    print(f"\n🎯 TOP RECOMMENDATIONS")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(rec['priority'], '⚪')
        print(f"   {i}. {priority_emoji} [{rec['priority'].upper()}] {rec['action']}")
        print(f"      Deadline: {rec['deadline']}")

    if report['affected_services']:
        print(f"\n📦 AFFECTED SERVICES")
        for service in report['affected_services'][:10]:
            print(f"   • {service}")

    print("\n" + "=" * 80)
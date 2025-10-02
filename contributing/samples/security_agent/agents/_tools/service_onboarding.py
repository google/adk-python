"""
Service Onboarding Tool - URL-Based Service Discovery and Compliance
Allows freehand input via GCP documentation URLs
"""

import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from google.cloud import bigquery

logger = logging.getLogger(__name__)

class ServiceOnboardingTool:
    """
    Onboard new GCP services by pasting documentation URLs.
    Automatically extracts service details and performs compliance checks.
    """

    def __init__(self, project_id: str = None):
        """Initialize the service onboarding tool"""
        import os
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
        self.bq_client = None
        try:
            self.bq_client = bigquery.Client(project=self.project_id)
            logger.info(f"✅ BigQuery client initialized for project: {self.project_id}")
        except Exception as e:
            logger.warning(f"BigQuery not available: {e}")

        # Common GCP service URL patterns
        self.url_patterns = {
            'cloud.google.com/([^/]+)/docs': 'service_from_docs',
            'cloud.google.com/products/([^/]+)': 'service_from_products',
            'developers.google.com/([^/]+)': 'service_from_developers',
            'firebase.google.com/docs/([^/]+)': 'firebase_service'
        }

        # Cache for previously analyzed services
        self.service_cache = {}

    def onboard_service_from_url(self, doc_url: str) -> Dict[str, Any]:
        """
        Main entry point: Onboard a service from its documentation URL.

        Args:
            doc_url: GCP documentation URL (e.g., https://cloud.google.com/bigquery/docs)

        Returns:
            Complete onboarding analysis with recommendations
        """
        logger.info(f"🚀 Starting service onboarding from URL: {doc_url}")

        # Step 1: Extract service information from URL
        service_info = self._extract_service_from_url(doc_url)

        if not service_info['success']:
            return {
                'success': False,
                'error': service_info.get('error', 'Could not extract service from URL'),
                'suggestion': 'Please provide a valid GCP service documentation URL'
            }

        # Step 2: Fetch and analyze documentation
        doc_analysis = self._analyze_documentation(doc_url, service_info)

        # Step 3: Find similar previously approved services
        similar_services = self._find_similar_services(service_info)

        # Step 4: Generate security recommendations
        security_recommendations = self._generate_security_recommendations(
            service_info,
            doc_analysis,
            similar_services
        )

        # Step 5: Compliance checks
        compliance_status = self._check_compliance(service_info, doc_analysis)

        # Step 6: Generate least-privilege IAM recommendations
        iam_recommendations = self._generate_iam_recommendations(
            service_info,
            similar_services
        )

        # Step 7: Create onboarding report
        onboarding_report = self._create_onboarding_report(
            service_info,
            doc_analysis,
            similar_services,
            security_recommendations,
            compliance_status,
            iam_recommendations
        )

        # Step 8: Store in BigQuery for future reference
        self._store_onboarding_analysis(onboarding_report)

        return onboarding_report

    def _extract_service_from_url(self, url: str) -> Dict[str, Any]:
        """Extract service name and details from documentation URL"""
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()

            # Try to match known patterns
            for pattern, extractor in self.url_patterns.items():
                match = re.search(pattern, parsed_url.netloc + path)
                if match:
                    service_name = match.group(1)

                    # Clean up service name
                    service_name = service_name.replace('-', ' ').replace('_', ' ')

                    # Map common service names
                    service_mapping = {
                        'bigquery': 'BigQuery',
                        'compute': 'Compute Engine',
                        'storage': 'Cloud Storage',
                        'functions': 'Cloud Functions',
                        'run': 'Cloud Run',
                        'gke': 'Google Kubernetes Engine',
                        'pubsub': 'Pub/Sub',
                        'firestore': 'Firestore',
                        'spanner': 'Cloud Spanner',
                        'vision': 'Vision AI',
                        'natural language': 'Natural Language AI',
                        'vertex ai': 'Vertex AI',
                        'dataflow': 'Dataflow',
                        'dataproc': 'Dataproc',
                        'composer': 'Cloud Composer',
                        'memorystore': 'Memorystore',
                        'sql': 'Cloud SQL',
                        'vpc': 'Virtual Private Cloud',
                        'iam': 'Identity and Access Management',
                        'kms': 'Cloud Key Management Service',
                        'secret manager': 'Secret Manager',
                        'dlp': 'Cloud Data Loss Prevention'
                    }

                    # Find best match in mapping
                    for key, proper_name in service_mapping.items():
                        if key in service_name.lower():
                            service_name = proper_name
                            break

                    return {
                        'success': True,
                        'service_name': service_name,
                        'service_id': service_name.lower().replace(' ', '_'),
                        'documentation_url': url,
                        'category': self._determine_category(service_name),
                        'extraction_method': extractor
                    }

            # If no pattern matches, try to extract from page title
            return self._extract_from_page_content(url)

        except Exception as e:
            logger.error(f"Error extracting service from URL: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_from_page_content(self, url: str) -> Dict[str, Any]:
        """Fallback: Extract service name from page content"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try to find service name in title or h1
            title = soup.find('title')
            if title:
                title_text = title.text
                # Extract service name from patterns like "BigQuery | Google Cloud"
                if '|' in title_text:
                    service_name = title_text.split('|')[0].strip()
                else:
                    service_name = title_text.replace('Documentation', '').replace('Docs', '').strip()

                return {
                    'success': True,
                    'service_name': service_name,
                    'service_id': service_name.lower().replace(' ', '_'),
                    'documentation_url': url,
                    'category': self._determine_category(service_name),
                    'extraction_method': 'page_content'
                }

        except Exception as e:
            logger.error(f"Could not extract from page content: {e}")

        return {
            'success': False,
            'error': 'Could not extract service name from URL or page content'
        }

    def _determine_category(self, service_name: str) -> str:
        """Determine the service category"""
        service_lower = service_name.lower()

        categories = {
            'compute': ['compute', 'gke', 'kubernetes', 'run', 'functions', 'app engine'],
            'storage': ['storage', 'filestore', 'persistent disk'],
            'database': ['sql', 'spanner', 'firestore', 'datastore', 'bigtable', 'memorystore', 'redis'],
            'analytics': ['bigquery', 'dataflow', 'dataproc', 'composer', 'dataprep', 'pub/sub'],
            'ai_ml': ['ai', 'ml', 'vertex', 'vision', 'speech', 'language', 'translation', 'automl'],
            'networking': ['vpc', 'load balancing', 'cdn', 'interconnect', 'network', 'dns', 'nat'],
            'security': ['iam', 'kms', 'secret', 'security', 'dlp', 'certificate', 'binary authorization'],
            'management': ['monitoring', 'logging', 'trace', 'profiler', 'debugger', 'operations'],
            'developer': ['build', 'deploy', 'source', 'container', 'artifact', 'code']
        }

        for category, keywords in categories.items():
            if any(keyword in service_lower for keyword in keywords):
                return category

        return 'other'

    def _analyze_documentation(self, url: str, service_info: Dict) -> Dict[str, Any]:
        """Analyze the documentation page for security and compliance info"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            text_content = soup.get_text().lower()

            analysis = {
                'has_iam_section': 'iam' in text_content or 'identity' in text_content,
                'has_security_section': 'security' in text_content,
                'has_encryption': 'encrypt' in text_content,
                'has_audit_logging': 'audit' in text_content or 'logging' in text_content,
                'has_vpc_controls': 'vpc' in text_content or 'private' in text_content,
                'has_compliance_info': any(term in text_content for term in
                    ['compliance', 'hipaa', 'pci', 'sox', 'gdpr', 'iso']),
                'has_sla': 'sla' in text_content or 'availability' in text_content,
                'has_quotas': 'quota' in text_content or 'limit' in text_content,
                'has_pricing': 'pricing' in text_content or 'cost' in text_content
            }

            # Extract mentioned IAM roles
            roles_mentioned = []
            role_pattern = r'roles/[a-zA-Z]+\.[a-zA-Z]+'
            roles_found = re.findall(role_pattern, response.text)
            if roles_found:
                roles_mentioned = list(set(roles_found))

            analysis['iam_roles_mentioned'] = roles_mentioned

            # Check for dangerous permissions
            dangerous_terms = ['owner', 'admin', 'editor', '*', 'setIamPolicy']
            analysis['has_dangerous_permissions'] = any(term in text_content for term in dangerous_terms)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing documentation: {e}")
            return {
                'error': str(e),
                'has_iam_section': False,
                'has_security_section': False
            }

    def _find_similar_services(self, service_info: Dict) -> List[Dict]:
        """Find previously onboarded services that are similar"""
        similar_services = []

        if self.bq_client:
            try:
                query = f"""
                SELECT
                    service_name,
                    service_id,
                    category,
                    approved_date,
                    iam_roles_used,
                    security_score,
                    compliance_status
                FROM `{self.project_id}.security_data.onboarded_services`
                WHERE category = @category
                    OR service_name LIKE @name_pattern
                ORDER BY approved_date DESC
                LIMIT 5
                """

                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("category", "STRING", service_info['category']),
                        bigquery.ScalarQueryParameter("name_pattern", "STRING", f"%{service_info['service_name'][:4]}%")
                    ]
                )

                results = self.bq_client.query(query, job_config=job_config)

                for row in results:
                    similar_services.append({
                        'service_name': row.service_name,
                        'service_id': row.service_id,
                        'category': row.category,
                        'approved_date': row.approved_date,
                        'iam_roles': row.iam_roles_used,
                        'security_score': row.security_score,
                        'compliance_status': row.compliance_status
                    })

            except Exception as e:
                logger.warning(f"Could not query similar services: {e}")

        # Fallback to hardcoded examples if no BigQuery
        if not similar_services:
            if service_info['category'] == 'analytics':
                similar_services = [
                    {
                        'service_name': 'BigQuery',
                        'iam_roles': ['roles/bigquery.dataViewer', 'roles/bigquery.jobUser'],
                        'security_score': 85,
                        'compliance_status': 'approved'
                    }
                ]
            elif service_info['category'] == 'storage':
                similar_services = [
                    {
                        'service_name': 'Cloud Storage',
                        'iam_roles': ['roles/storage.objectViewer'],
                        'security_score': 90,
                        'compliance_status': 'approved'
                    }
                ]

        return similar_services

    def _generate_security_recommendations(self, service_info: Dict, doc_analysis: Dict,
                                          similar_services: List[Dict]) -> Dict[str, Any]:
        """Generate security recommendations based on analysis"""
        recommendations = {
            'priority': 'high',
            'required_controls': [],
            'recommended_controls': [],
            'warnings': [],
            'best_practices': []
        }

        # Required controls based on documentation analysis
        if not doc_analysis.get('has_encryption'):
            recommendations['required_controls'].append({
                'control': 'Enable encryption at rest',
                'reason': 'No encryption mentioned in documentation',
                'implementation': 'Enable default Google-managed encryption keys (GMEK) or customer-managed encryption keys (CMEK)'
            })

        if not doc_analysis.get('has_audit_logging'):
            recommendations['required_controls'].append({
                'control': 'Enable audit logging',
                'reason': 'Audit logging not documented',
                'implementation': 'Enable Cloud Audit Logs for data access and admin activity'
            })

        if not doc_analysis.get('has_vpc_controls'):
            recommendations['recommended_controls'].append({
                'control': 'Implement VPC Service Controls',
                'reason': 'No VPC controls mentioned',
                'implementation': 'Create VPC service perimeter if handling sensitive data'
            })

        # Warnings based on dangerous permissions
        if doc_analysis.get('has_dangerous_permissions'):
            recommendations['warnings'].append({
                'type': 'dangerous_permissions',
                'message': 'Documentation mentions admin/owner roles',
                'recommendation': 'Use least-privilege roles instead of admin/owner'
            })

        # Best practices from similar services
        if similar_services:
            common_roles = set()
            for service in similar_services:
                if service.get('iam_roles'):
                    common_roles.update(service['iam_roles'])

            if common_roles:
                recommendations['best_practices'].append({
                    'practice': 'Use similar IAM roles',
                    'details': f"Similar services use: {', '.join(list(common_roles)[:3])}"
                })

        # Category-specific recommendations
        if service_info['category'] == 'database':
            recommendations['required_controls'].append({
                'control': 'Configure automated backups',
                'reason': 'Database service requires backup strategy',
                'implementation': 'Set up automated backups with appropriate retention'
            })
        elif service_info['category'] == 'ai_ml':
            recommendations['recommended_controls'].append({
                'control': 'Implement data privacy controls',
                'reason': 'AI/ML services process potentially sensitive data',
                'implementation': 'Use Cloud DLP to scan and redact sensitive information'
            })

        return recommendations

    def _check_compliance(self, service_info: Dict, doc_analysis: Dict) -> Dict[str, Any]:
        """Check compliance against enterprise standards"""
        compliance_checks = {
            'overall_status': 'pending',
            'checks_passed': [],
            'checks_failed': [],
            'checks_warning': [],
            'remediation_required': []
        }

        # Check 1: Encryption requirement
        if doc_analysis.get('has_encryption'):
            compliance_checks['checks_passed'].append('Encryption supported')
        else:
            compliance_checks['checks_failed'].append('Encryption not verified')
            compliance_checks['remediation_required'].append({
                'issue': 'Missing encryption',
                'action': 'Verify and enable encryption at rest and in transit'
            })

        # Check 2: Audit logging
        if doc_analysis.get('has_audit_logging'):
            compliance_checks['checks_passed'].append('Audit logging available')
        else:
            compliance_checks['checks_warning'].append('Audit logging not confirmed')

        # Check 3: No dangerous permissions
        if not doc_analysis.get('has_dangerous_permissions'):
            compliance_checks['checks_passed'].append('No admin/owner roles recommended')
        else:
            compliance_checks['checks_failed'].append('Documentation suggests admin/owner roles')
            compliance_checks['remediation_required'].append({
                'issue': 'Overly permissive roles',
                'action': 'Use least-privilege roles instead'
            })

        # Check 4: Compliance certifications
        if doc_analysis.get('has_compliance_info'):
            compliance_checks['checks_passed'].append('Compliance certifications documented')
        else:
            compliance_checks['checks_warning'].append('No compliance certifications found')

        # Determine overall status
        if compliance_checks['checks_failed']:
            compliance_checks['overall_status'] = 'failed'
        elif compliance_checks['checks_warning']:
            compliance_checks['overall_status'] = 'review_required'
        else:
            compliance_checks['overall_status'] = 'approved'

        return compliance_checks

    def _generate_iam_recommendations(self, service_info: Dict,
                                     similar_services: List[Dict]) -> Dict[str, Any]:
        """Generate least-privilege IAM recommendations"""
        iam_recommendations = {
            'recommended_roles': [],
            'avoid_roles': [],
            'custom_role_needed': False,
            'justification': []
        }

        # NEVER recommend admin/owner/editor roles
        iam_recommendations['avoid_roles'] = [
            'roles/owner',
            'roles/editor',
            f"roles/{service_info['service_id']}.admin"
        ]

        # Recommend based on category
        if service_info['category'] == 'analytics':
            iam_recommendations['recommended_roles'] = [
                'roles/bigquery.dataViewer',
                'roles/bigquery.jobUser'
            ]
            iam_recommendations['justification'].append(
                'Analytics services typically need data viewing and job execution'
            )
        elif service_info['category'] == 'storage':
            iam_recommendations['recommended_roles'] = [
                'roles/storage.objectViewer',
                'roles/storage.objectCreator'
            ]
            iam_recommendations['justification'].append(
                'Storage access should be limited to specific operations'
            )
        elif service_info['category'] == 'compute':
            iam_recommendations['recommended_roles'] = [
                'roles/compute.viewer',
                'roles/compute.instanceAdmin.v1'
            ]
            iam_recommendations['justification'].append(
                'Compute resources need instance-level permissions'
            )
        else:
            # Generic least-privilege recommendations
            iam_recommendations['recommended_roles'] = [
                f"roles/{service_info['service_id']}.viewer",
                f"roles/{service_info['service_id']}.user"
            ]
            iam_recommendations['justification'].append(
                'Start with viewer/user roles and expand as needed'
            )

        # Learn from similar services
        if similar_services:
            for service in similar_services:
                if service.get('iam_roles'):
                    for role in service['iam_roles']:
                        if role not in iam_recommendations['recommended_roles'] and \
                           'admin' not in role.lower() and 'owner' not in role.lower():
                            iam_recommendations['recommended_roles'].append(role)

            iam_recommendations['justification'].append(
                f"Based on {len(similar_services)} similar approved services"
            )

        # Check if custom role might be needed
        if len(iam_recommendations['recommended_roles']) > 3:
            iam_recommendations['custom_role_needed'] = True
            iam_recommendations['justification'].append(
                'Consider creating a custom role to combine permissions'
            )

        return iam_recommendations

    def _create_onboarding_report(self, service_info: Dict, doc_analysis: Dict,
                                  similar_services: List[Dict],
                                  security_recommendations: Dict,
                                  compliance_status: Dict,
                                  iam_recommendations: Dict) -> Dict[str, Any]:
        """Create comprehensive onboarding report"""

        # Calculate risk score
        risk_score = 0
        if doc_analysis.get('has_dangerous_permissions'):
            risk_score += 30
        if not doc_analysis.get('has_encryption'):
            risk_score += 25
        if not doc_analysis.get('has_audit_logging'):
            risk_score += 20
        if compliance_status['overall_status'] == 'failed':
            risk_score += 25

        risk_level = 'low' if risk_score < 30 else 'medium' if risk_score < 60 else 'high'

        report = {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'service_info': service_info,
            'risk_assessment': {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'factors': []
            },
            'compliance': compliance_status,
            'security_recommendations': security_recommendations,
            'iam_recommendations': iam_recommendations,
            'similar_services': similar_services[:3] if similar_services else [],
            'documentation_analysis': {
                'url_analyzed': service_info['documentation_url'],
                'security_features': {
                    'encryption': doc_analysis.get('has_encryption', False),
                    'audit_logging': doc_analysis.get('has_audit_logging', False),
                    'vpc_controls': doc_analysis.get('has_vpc_controls', False),
                    'compliance_certs': doc_analysis.get('has_compliance_info', False)
                }
            },
            'next_steps': [],
            'approval_workflow': {
                'status': 'pending_review',
                'required_approvals': [],
                'auto_approved': False
            }
        }

        # Add risk factors
        if doc_analysis.get('has_dangerous_permissions'):
            report['risk_assessment']['factors'].append('Documentation mentions admin/owner roles')
        if not doc_analysis.get('has_encryption'):
            report['risk_assessment']['factors'].append('Encryption not documented')
        if not doc_analysis.get('has_audit_logging'):
            report['risk_assessment']['factors'].append('Audit logging not confirmed')

        # Determine next steps
        if compliance_status['remediation_required']:
            report['next_steps'].append({
                'priority': 'high',
                'action': 'Address compliance issues',
                'details': compliance_status['remediation_required']
            })

        if iam_recommendations['custom_role_needed']:
            report['next_steps'].append({
                'priority': 'medium',
                'action': 'Create custom IAM role',
                'details': 'Combine permissions for least privilege'
            })

        report['next_steps'].append({
            'priority': 'low',
            'action': 'Schedule security review',
            'details': 'Review after 30 days of usage'
        })

        # Determine approval requirements
        if risk_level == 'high':
            report['approval_workflow']['required_approvals'] = ['security_team', 'architecture_board']
        elif risk_level == 'medium':
            report['approval_workflow']['required_approvals'] = ['security_team']
        else:
            report['approval_workflow']['auto_approved'] = True
            report['approval_workflow']['status'] = 'auto_approved'

        return report

    def _store_onboarding_analysis(self, report: Dict[str, Any]):
        """Store the onboarding analysis in BigQuery for future reference"""
        if not self.bq_client:
            return

        try:
            # Create dataset if it doesn't exist
            dataset_id = f"{self.project_id}.security_data"
            try:
                self.bq_client.get_dataset(dataset_id)
            except:
                dataset = bigquery.Dataset(dataset_id)
                dataset.location = "US"
                self.bq_client.create_dataset(dataset, timeout=30)

            # Store the analysis
            table_id = f"{dataset_id}.service_onboarding_history"

            row = {
                'service_name': report['service_info']['service_name'],
                'service_id': report['service_info']['service_id'],
                'category': report['service_info']['category'],
                'documentation_url': report['service_info']['documentation_url'],
                'risk_score': report['risk_assessment']['risk_score'],
                'risk_level': report['risk_assessment']['risk_level'],
                'compliance_status': report['compliance']['overall_status'],
                'recommended_roles': json.dumps(report['iam_recommendations']['recommended_roles']),
                'analysis_timestamp': report['timestamp'],
                'full_report': json.dumps(report)
            }

            errors = self.bq_client.insert_rows_json(table_id, [row], ignore_unknown_values=True)

            if errors:
                logger.error(f"Failed to store onboarding analysis: {errors}")
            else:
                logger.info(f"✅ Onboarding analysis stored for {report['service_info']['service_name']}")

        except Exception as e:
            logger.error(f"Error storing onboarding analysis: {e}")


# ADK Tool function
def onboard_service(doc_url: str) -> Dict[str, Any]:
    """
    ADK tool function for service onboarding via documentation URL.

    Args:
        doc_url: GCP service documentation URL

    Returns:
        Onboarding analysis and recommendations
    """
    tool = ServiceOnboardingTool()
    return tool.onboard_service_from_url(doc_url)


# Example usage
if __name__ == "__main__":
    # Test with various GCP service documentation URLs
    test_urls = [
        "https://cloud.google.com/bigquery/docs",
        "https://cloud.google.com/vertex-ai/docs",
        "https://cloud.google.com/secret-manager/docs",
        "https://cloud.google.com/dataflow/docs"
    ]

    tool = ServiceOnboardingTool()

    for url in test_urls[:1]:  # Test with first URL
        print(f"\n{'='*80}")
        print(f"Testing Service Onboarding with: {url}")
        print('='*80)

        result = tool.onboard_service_from_url(url)

        if result['success']:
            print(f"\n✅ Service: {result['service_info']['service_name']}")
            print(f"   Category: {result['service_info']['category']}")
            print(f"   Risk Level: {result['risk_assessment']['risk_level']} (Score: {result['risk_assessment']['risk_score']})")
            print(f"   Compliance: {result['compliance']['overall_status']}")

            print(f"\n📋 Recommended IAM Roles:")
            for role in result['iam_recommendations']['recommended_roles'][:3]:
                print(f"   • {role}")

            print(f"\n⚠️ Avoid These Roles:")
            for role in result['iam_recommendations']['avoid_roles'][:3]:
                print(f"   • {role}")

            if result['security_recommendations']['required_controls']:
                print(f"\n🔒 Required Security Controls:")
                for control in result['security_recommendations']['required_controls']:
                    print(f"   • {control['control']}: {control['reason']}")

            if result['next_steps']:
                print(f"\n📌 Next Steps:")
                for step in result['next_steps']:
                    print(f"   [{step['priority']}] {step['action']}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
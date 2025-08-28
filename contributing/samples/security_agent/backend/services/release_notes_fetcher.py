#!/usr/bin/env python3
"""
Google Cloud Release Notes Fetcher
==================================

Fetches and parses release notes from Google Cloud services
for security and billing impact analysis.
"""

import os
import json
import logging
import sqlite3
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReleaseNotesFetcher:
    """Fetches and analyzes Google Cloud release notes."""
    
    # Services currently used by organization (configurable)
    ORGANIZATION_SERVICES = [
        "bigquery",
        "compute-engine",
        "cloud-storage",
        "cloud-sql",
        "cloud-iam",
        "vpc",
        "cloud-monitoring",
        "cloud-logging",
        "cloud-kms",
        "secret-manager",
        "cloud-asset-inventory",
        "security-command-center",
        "cloud-functions",
        "cloud-run",
        "gke",
        "dataflow",
        "pub-sub",
        "cloud-spanner"
    ]
    
    # Release notes base URLs
    RELEASE_NOTES_URLS = {
        "bigquery": "https://cloud.google.com/bigquery/docs/release-notes",
        "compute-engine": "https://cloud.google.com/compute/docs/release-notes",
        "cloud-storage": "https://cloud.google.com/storage/docs/release-notes",
        "cloud-sql": "https://cloud.google.com/sql/docs/release-notes",
        "cloud-iam": "https://cloud.google.com/iam/docs/release-notes",
        "vpc": "https://cloud.google.com/vpc/docs/release-notes",
        "cloud-monitoring": "https://cloud.google.com/monitoring/api/release-notes",
        "cloud-logging": "https://cloud.google.com/logging/docs/release-notes",
        "cloud-kms": "https://cloud.google.com/kms/docs/release-notes",
        "secret-manager": "https://cloud.google.com/secret-manager/docs/release-notes",
        "cloud-asset-inventory": "https://cloud.google.com/asset-inventory/docs/release-notes",
        "security-command-center": "https://cloud.google.com/security-command-center/docs/release-notes",
        "cloud-functions": "https://cloud.google.com/functions/docs/release-notes",
        "cloud-run": "https://cloud.google.com/run/docs/release-notes",
        "gke": "https://cloud.google.com/kubernetes-engine/docs/release-notes",
        "dataflow": "https://cloud.google.com/dataflow/docs/release-notes",
        "pub-sub": "https://cloud.google.com/pubsub/docs/release-notes",
        "cloud-spanner": "https://cloud.google.com/spanner/docs/release-notes"
    }
    
    def __init__(self, db_path: str = None):
        """Initialize the release notes fetcher."""
        self.db_path = db_path or os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        self.setup_database()
        
    def setup_database(self):
        """Create database tables for storing release notes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Table for storing release notes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS release_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL,
                    release_date TEXT,
                    title TEXT,
                    description TEXT,
                    note_type TEXT,  -- feature, fix, deprecation, security, pricing
                    security_impact TEXT,  -- JSON with security details
                    billing_impact TEXT,   -- JSON with billing details
                    affected_features TEXT, -- JSON array
                    raw_content TEXT,
                    source_url TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analyzed_at TIMESTAMP
                )
            """)
            
            # Table for security impact analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_impacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_note_id INTEGER,
                    service TEXT NOT NULL,
                    impact_type TEXT,  -- encryption, authentication, authorization, vulnerability, compliance
                    severity TEXT,     -- critical, high, medium, low
                    description TEXT,
                    remediation TEXT,
                    cve_ids TEXT,      -- JSON array of CVE IDs if applicable
                    compliance_frameworks TEXT, -- JSON array (SOC2, HIPAA, etc.)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (release_note_id) REFERENCES release_notes(id)
                )
            """)
            
            # Table for billing impact analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS billing_impacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_note_id INTEGER,
                    service TEXT NOT NULL,
                    impact_type TEXT,  -- price_increase, price_decrease, new_charge, deprecated_sku
                    estimated_impact_percent REAL,
                    old_pricing TEXT,
                    new_pricing TEXT,
                    effective_date TEXT,
                    affected_skus TEXT,  -- JSON array
                    cost_optimization_tips TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (release_note_id) REFERENCES release_notes(id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_release_notes_service ON release_notes(service)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_release_notes_date ON release_notes(release_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_security_impacts_severity ON security_impacts(severity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_billing_impacts_type ON billing_impacts(impact_type)")
            
            conn.commit()
            logger.info("✅ Release notes database tables created")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def fetch_release_notes(self, service: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch release notes for a specific service.
        
        Args:
            service: The GCP service name
            days_back: Number of days to look back for release notes
            
        Returns:
            List of parsed release notes
        """
        if service not in self.RELEASE_NOTES_URLS:
            logger.warning(f"No release notes URL for service: {service}")
            return []
        
        url = self.RELEASE_NOTES_URLS[service]
        release_notes = []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Parse HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find release note entries (patterns vary by service)
                notes = self._parse_release_notes_html(soup, service)
                
                # Filter by date
                cutoff_date = datetime.now() - timedelta(days=days_back)
                for note in notes:
                    if note.get('release_date'):
                        try:
                            note_date = datetime.strptime(note['release_date'], '%Y-%m-%d')
                            if note_date >= cutoff_date:
                                release_notes.append(note)
                        except:
                            # Include notes without valid dates
                            release_notes.append(note)
                    else:
                        release_notes.append(note)
                
                logger.info(f"✅ Fetched {len(release_notes)} release notes for {service}")
                
        except Exception as e:
            logger.error(f"Error fetching release notes for {service}: {e}")
        
        return release_notes
    
    def _parse_release_notes_html(self, soup: BeautifulSoup, service: str) -> List[Dict[str, Any]]:
        """Parse release notes from HTML content."""
        notes = []
        
        # Look for common release note patterns
        # Pattern 1: Date headers with content
        date_headers = soup.find_all(['h2', 'h3'], text=re.compile(r'\d{4}-\d{2}-\d{2}'))
        
        for header in date_headers:
            date_text = header.get_text().strip()
            release_date = self._extract_date(date_text)
            
            # Get content following the header
            content_parts = []
            sibling = header.find_next_sibling()
            
            while sibling and sibling.name not in ['h2', 'h3']:
                if sibling.name in ['p', 'ul', 'ol']:
                    content_parts.append(sibling.get_text().strip())
                sibling = sibling.find_next_sibling()
            
            if content_parts:
                description = '\n'.join(content_parts)
                note_type = self._classify_note_type(description)
                
                notes.append({
                    'service': service,
                    'release_date': release_date,
                    'title': f"{service} update - {release_date}",
                    'description': description,
                    'note_type': note_type,
                    'raw_content': str(header) + ''.join(str(s) for s in content_parts[:3])
                })
        
        # Pattern 2: Release note sections with class names
        release_sections = soup.find_all('div', class_=re.compile('release|changelog|update'))
        
        for section in release_sections[:10]:  # Limit to prevent excessive parsing
            date_elem = section.find(text=re.compile(r'\d{4}-\d{2}-\d{2}'))
            if date_elem:
                release_date = self._extract_date(date_elem)
                description = section.get_text().strip()[:1000]  # Limit length
                note_type = self._classify_note_type(description)
                
                notes.append({
                    'service': service,
                    'release_date': release_date,
                    'title': f"{service} update",
                    'description': description,
                    'note_type': note_type,
                    'raw_content': str(section)[:500]
                })
        
        return notes
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text in YYYY-MM-DD format."""
        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
        if date_match:
            return date_match.group(0)
        return None
    
    def _classify_note_type(self, description: str) -> str:
        """Classify the type of release note based on content."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['security', 'vulnerability', 'cve', 'patch']):
            return 'security'
        elif any(word in description_lower for word in ['price', 'cost', 'billing', 'charge', 'fee']):
            return 'pricing'
        elif any(word in description_lower for word in ['deprecat', 'sunset', 'end of life', 'eol']):
            return 'deprecation'
        elif any(word in description_lower for word in ['fix', 'bug', 'issue', 'resolve']):
            return 'fix'
        elif any(word in description_lower for word in ['feature', 'new', 'introduce', 'launch']):
            return 'feature'
        else:
            return 'general'
    
    def analyze_security_impact(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze security impact of a release note.
        
        Returns:
            Dictionary with security impact details
        """
        security_impact = {
            'has_impact': False,
            'impact_type': None,
            'severity': 'low',
            'details': [],
            'remediation': [],
            'compliance_affected': []
        }
        
        description = note.get('description', '').lower()
        
        # Check for encryption-related changes
        if any(word in description for word in ['encrypt', 'tls', 'ssl', 'cipher', 'aes', 'rsa']):
            security_impact['has_impact'] = True
            security_impact['impact_type'] = 'encryption'
            security_impact['details'].append('Encryption settings or algorithms have been updated')
            security_impact['remediation'].append('Review encryption configurations and update if necessary')
            security_impact['compliance_affected'].extend(['SOC2', 'HIPAA', 'PCI-DSS'])
            
            if 'additional' in description or 'new' in description:
                security_impact['severity'] = 'medium'
                security_impact['details'].append('New encryption options available')
            elif 'deprecat' in description:
                security_impact['severity'] = 'high'
                security_impact['details'].append('Encryption method being deprecated')
        
        # Check for authentication/authorization changes
        if any(word in description for word in ['auth', 'iam', 'permission', 'role', 'access', 'identity']):
            security_impact['has_impact'] = True
            security_impact['impact_type'] = 'authentication' if 'auth' in description else 'authorization'
            security_impact['details'].append('Authentication or authorization mechanisms updated')
            security_impact['remediation'].append('Audit IAM policies and service account permissions')
            security_impact['compliance_affected'].extend(['SOC2', 'ISO27001'])
            
            if 'required' in description or 'mandatory' in description:
                security_impact['severity'] = 'high'
        
        # Check for vulnerability fixes
        if any(word in description for word in ['vulnerability', 'cve', 'security fix', 'patch']):
            security_impact['has_impact'] = True
            security_impact['impact_type'] = 'vulnerability'
            security_impact['severity'] = 'critical' if 'critical' in description else 'high'
            security_impact['details'].append('Security vulnerability has been addressed')
            security_impact['remediation'].append('Apply security patches immediately')
            
            # Extract CVE IDs
            cve_matches = re.findall(r'CVE-\d{4}-\d+', description.upper())
            if cve_matches:
                security_impact['cve_ids'] = cve_matches
        
        # Check for compliance-related changes
        if any(word in description for word in ['compliance', 'regulatory', 'gdpr', 'hipaa', 'sox', 'pci']):
            security_impact['has_impact'] = True
            security_impact['impact_type'] = 'compliance'
            security_impact['details'].append('Compliance-related changes implemented')
            security_impact['remediation'].append('Review compliance documentation and attestations')
            
            # Identify specific frameworks
            frameworks = []
            if 'gdpr' in description:
                frameworks.append('GDPR')
            if 'hipaa' in description:
                frameworks.append('HIPAA')
            if 'sox' in description or 'sarbanes' in description:
                frameworks.append('SOX')
            if 'pci' in description:
                frameworks.append('PCI-DSS')
            if 'soc' in description and '2' in description:
                frameworks.append('SOC2')
            
            security_impact['compliance_affected'].extend(frameworks)
        
        # Check for network security changes
        if any(word in description for word in ['firewall', 'vpc', 'network', 'subnet', 'ingress', 'egress']):
            security_impact['has_impact'] = True
            security_impact['impact_type'] = 'network_security'
            security_impact['details'].append('Network security configuration changes')
            security_impact['remediation'].append('Review firewall rules and network policies')
        
        return security_impact
    
    def analyze_billing_impact(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze billing impact of a release note.
        
        Returns:
            Dictionary with billing impact details
        """
        billing_impact = {
            'has_impact': False,
            'impact_type': None,
            'estimated_change_percent': 0,
            'details': [],
            'optimization_tips': [],
            'affected_skus': []
        }
        
        description = note.get('description', '').lower()
        
        # Check for pricing changes
        if any(word in description for word in ['price', 'pricing', 'cost', 'charge', 'fee', 'rate']):
            billing_impact['has_impact'] = True
            
            # Determine type of pricing change
            if any(word in description for word in ['increase', 'raise', 'higher']):
                billing_impact['impact_type'] = 'price_increase'
                billing_impact['details'].append('Pricing has been increased')
                billing_impact['optimization_tips'].append('Review resource usage and consider optimization')
                
                # Try to extract percentage
                percent_match = re.search(r'(\d+)%', description)
                if percent_match:
                    billing_impact['estimated_change_percent'] = float(percent_match.group(1))
                    
            elif any(word in description for word in ['decrease', 'reduce', 'lower', 'discount']):
                billing_impact['impact_type'] = 'price_decrease'
                billing_impact['details'].append('Pricing has been reduced')
                billing_impact['optimization_tips'].append('Consider expanding usage to take advantage of lower prices')
                
                percent_match = re.search(r'(\d+)%', description)
                if percent_match:
                    billing_impact['estimated_change_percent'] = -float(percent_match.group(1))
                    
            elif 'new' in description:
                billing_impact['impact_type'] = 'new_charge'
                billing_impact['details'].append('New charges or billing items introduced')
                billing_impact['optimization_tips'].append('Evaluate if new features justify additional costs')
        
        # Check for SKU changes
        if 'sku' in description or 'product' in description:
            billing_impact['has_impact'] = True
            if 'deprecat' in description or 'retire' in description:
                billing_impact['impact_type'] = 'deprecated_sku'
                billing_impact['details'].append('Some SKUs are being deprecated')
                billing_impact['optimization_tips'].append('Migrate to recommended replacement SKUs')
        
        # Check for quota or limit changes
        if any(word in description for word in ['quota', 'limit', 'threshold']):
            billing_impact['has_impact'] = True
            billing_impact['details'].append('Quota or limit changes may affect billing')
            billing_impact['optimization_tips'].append('Review quota usage and adjust resource allocation')
        
        # Check for free tier changes
        if 'free tier' in description or 'free usage' in description:
            billing_impact['has_impact'] = True
            billing_impact['impact_type'] = 'free_tier_change'
            billing_impact['details'].append('Free tier or trial offerings have been modified')
            billing_impact['optimization_tips'].append('Review free tier limits and adjust usage patterns')
        
        # Check for commitment/reservation changes
        if any(word in description for word in ['commitment', 'reservation', 'sustained', 'discount']):
            billing_impact['has_impact'] = True
            billing_impact['details'].append('Commitment or reservation pricing updated')
            billing_impact['optimization_tips'].append('Evaluate commitment options for cost savings')
        
        return billing_impact
    
    async def fetch_and_analyze_all_services(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Fetch and analyze release notes for all organization services.
        
        Returns:
            Summary of security and billing impacts
        """
        results = {
            'services_analyzed': [],
            'total_notes': 0,
            'security_impacts': [],
            'billing_impacts': [],
            'summary': {}
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for service in self.ORGANIZATION_SERVICES:
            logger.info(f"📊 Analyzing release notes for {service}...")
            
            # Fetch release notes
            notes = await self.fetch_release_notes(service, days_back)
            results['total_notes'] += len(notes)
            
            if notes:
                results['services_analyzed'].append(service)
                
                for note in notes:
                    # Analyze security impact
                    security_impact = self.analyze_security_impact(note)
                    if security_impact['has_impact']:
                        security_impact['service'] = service
                        security_impact['note'] = note
                        results['security_impacts'].append(security_impact)
                        
                        # Store in database
                        try:
                            cursor.execute("""
                                INSERT INTO release_notes (
                                    service, release_date, title, description, note_type,
                                    security_impact, source_url, analyzed_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                service,
                                note.get('release_date'),
                                note.get('title'),
                                note.get('description'),
                                note.get('note_type'),
                                json.dumps(security_impact),
                                self.RELEASE_NOTES_URLS.get(service),
                                datetime.now()
                            ))
                            
                            release_note_id = cursor.lastrowid
                            
                            cursor.execute("""
                                INSERT INTO security_impacts (
                                    release_note_id, service, impact_type, severity,
                                    description, remediation, cve_ids, compliance_frameworks
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                release_note_id,
                                service,
                                security_impact.get('impact_type'),
                                security_impact.get('severity'),
                                '\n'.join(security_impact.get('details', [])),
                                '\n'.join(security_impact.get('remediation', [])),
                                json.dumps(security_impact.get('cve_ids', [])),
                                json.dumps(security_impact.get('compliance_affected', []))
                            ))
                        except Exception as e:
                            logger.error(f"Error storing security impact: {e}")
                    
                    # Analyze billing impact
                    billing_impact = self.analyze_billing_impact(note)
                    if billing_impact['has_impact']:
                        billing_impact['service'] = service
                        billing_impact['note'] = note
                        results['billing_impacts'].append(billing_impact)
                        
                        # Store in database
                        try:
                            if not security_impact['has_impact']:
                                # Insert release note if not already inserted
                                cursor.execute("""
                                    INSERT INTO release_notes (
                                        service, release_date, title, description, note_type,
                                        billing_impact, source_url, analyzed_at
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    service,
                                    note.get('release_date'),
                                    note.get('title'),
                                    note.get('description'),
                                    note.get('note_type'),
                                    json.dumps(billing_impact),
                                    self.RELEASE_NOTES_URLS.get(service),
                                    datetime.now()
                                ))
                                release_note_id = cursor.lastrowid
                            
                            cursor.execute("""
                                INSERT INTO billing_impacts (
                                    release_note_id, service, impact_type, estimated_impact_percent,
                                    cost_optimization_tips
                                ) VALUES (?, ?, ?, ?, ?)
                            """, (
                                release_note_id,
                                service,
                                billing_impact.get('impact_type'),
                                billing_impact.get('estimated_change_percent'),
                                '\n'.join(billing_impact.get('optimization_tips', []))
                            ))
                        except Exception as e:
                            logger.error(f"Error storing billing impact: {e}")
        
        conn.commit()
        conn.close()
        
        # Generate summary
        results['summary'] = {
            'services_with_updates': len(results['services_analyzed']),
            'critical_security_impacts': sum(1 for s in results['security_impacts'] if s.get('severity') == 'critical'),
            'high_security_impacts': sum(1 for s in results['security_impacts'] if s.get('severity') == 'high'),
            'price_increases': sum(1 for b in results['billing_impacts'] if b.get('impact_type') == 'price_increase'),
            'price_decreases': sum(1 for b in results['billing_impacts'] if b.get('impact_type') == 'price_decrease'),
            'compliance_impacts': len(set(
                framework 
                for s in results['security_impacts'] 
                for framework in s.get('compliance_affected', [])
            ))
        }
        
        logger.info(f"✅ Analysis complete: {results['summary']}")
        return results


if __name__ == "__main__":
    import asyncio
    
    async def main():
        fetcher = ReleaseNotesFetcher()
        
        # Test fetching for a single service
        logger.info("🔍 Testing release notes fetch for BigQuery...")
        notes = await fetcher.fetch_release_notes("bigquery", days_back=30)
        
        if notes:
            logger.info(f"✅ Found {len(notes)} release notes")
            for note in notes[:3]:  # Show first 3
                logger.info(f"  - {note.get('release_date')}: {note.get('title')[:50]}...")
        
        # Test full analysis
        logger.info("\n📊 Running full analysis for all services...")
        results = await fetcher.fetch_and_analyze_all_services(days_back=30)
        
        logger.info("\n📈 Analysis Summary:")
        logger.info(f"  - Services analyzed: {len(results['services_analyzed'])}")
        logger.info(f"  - Total release notes: {results['total_notes']}")
        logger.info(f"  - Security impacts: {len(results['security_impacts'])}")
        logger.info(f"  - Billing impacts: {len(results['billing_impacts'])}")
        logger.info(f"  - Critical security issues: {results['summary']['critical_security_impacts']}")
        logger.info(f"  - Price increases: {results['summary']['price_increases']}")
    
    asyncio.run(main())
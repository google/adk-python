#!/usr/bin/env python3
"""
Integration Tests for Release Notes Fetcher
===========================================

Tests the complete flow of fetching, analyzing, and storing release notes
with security and billing impact analysis.
"""

import pytest
import asyncio
import os
import sys
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from services.release_notes_fetcher import ReleaseNotesFetcher
from services.msa_database_setup import create_msa_tables


class TestReleaseNotesIntegration:
    """Integration tests for release notes fetching and analysis."""
    
    @pytest.fixture
    def test_db_path(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_msa.db"
        create_msa_tables(str(db_path))
        return str(db_path)
    
    @pytest.fixture
    def fetcher(self, test_db_path):
        """Create a ReleaseNotesFetcher instance with test database."""
        return ReleaseNotesFetcher(db_path=test_db_path)
    
    @pytest.mark.asyncio
    async def test_database_setup(self, fetcher, test_db_path):
        """Test that database tables are created correctly."""
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # Check all required tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN (
                'release_notes', 'security_impacts', 'billing_impacts',
                'msa_emails', 'msa_changes', 'msa_impact_assessments'
            )
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'release_notes' in tables
        assert 'security_impacts' in tables
        assert 'billing_impacts' in tables
        assert 'msa_emails' in tables
        assert 'msa_changes' in tables
        
        conn.close()
        print("✅ Database tables created correctly")
    
    @pytest.mark.asyncio
    async def test_fetch_single_service_notes(self, fetcher):
        """Test fetching release notes for a single service."""
        # Test with BigQuery (usually has frequent updates)
        notes = await fetcher.fetch_release_notes("bigquery", days_back=30)
        
        # Should return a list
        assert isinstance(notes, list)
        
        # If notes are found, verify structure
        if notes:
            first_note = notes[0]
            assert 'service' in first_note
            assert 'release_date' in first_note
            assert 'description' in first_note
            assert 'note_type' in first_note
            
            print(f"✅ Fetched {len(notes)} release notes for BigQuery")
        else:
            print("⚠️ No release notes found (this may be normal)")
    
    @pytest.mark.asyncio
    async def test_security_impact_analysis(self, fetcher):
        """Test security impact analysis of release notes."""
        # Create a mock release note with security content
        mock_note = {
            'service': 'test-service',
            'release_date': datetime.now().strftime('%Y-%m-%d'),
            'title': 'Security Update',
            'description': 'Fixed critical vulnerability CVE-2024-1234. Enhanced encryption with AES-256. Updated authentication mechanisms for improved security.',
            'note_type': 'security'
        }
        
        # Analyze security impact
        impact = fetcher.analyze_security_impact(mock_note)
        
        # Verify security impact detection
        assert impact['has_impact'] == True
        assert impact['impact_type'] in ['encryption', 'vulnerability', 'authentication']
        assert impact['severity'] in ['critical', 'high', 'medium', 'low']
        assert len(impact['details']) > 0
        assert len(impact['remediation']) > 0
        
        # Should detect CVE
        if 'cve_ids' in impact:
            assert 'CVE-2024-1234' in impact['cve_ids']
        
        print("✅ Security impact analysis works correctly")
    
    @pytest.mark.asyncio
    async def test_billing_impact_analysis(self, fetcher):
        """Test billing impact analysis of release notes."""
        # Test price increase detection
        price_increase_note = {
            'service': 'compute-engine',
            'release_date': datetime.now().strftime('%Y-%m-%d'),
            'title': 'Pricing Update',
            'description': 'Pricing for n1-standard instances will increase by 10% starting next month.',
            'note_type': 'pricing'
        }
        
        impact = fetcher.analyze_billing_impact(price_increase_note)
        
        assert impact['has_impact'] == True
        assert impact['impact_type'] == 'price_increase'
        assert impact['estimated_change_percent'] == 10.0
        assert len(impact['optimization_tips']) > 0
        
        # Test price decrease detection
        price_decrease_note = {
            'service': 'cloud-storage',
            'release_date': datetime.now().strftime('%Y-%m-%d'),
            'title': 'Price Reduction',
            'description': 'Storage costs reduced by 15% for nearline storage class.',
            'note_type': 'pricing'
        }
        
        impact = fetcher.analyze_billing_impact(price_decrease_note)
        
        assert impact['has_impact'] == True
        assert impact['impact_type'] == 'price_decrease'
        assert impact['estimated_change_percent'] == -15.0
        
        print("✅ Billing impact analysis works correctly")
    
    @pytest.mark.asyncio
    async def test_compliance_impact_detection(self, fetcher):
        """Test compliance framework impact detection."""
        compliance_note = {
            'service': 'cloud-kms',
            'release_date': datetime.now().strftime('%Y-%m-%d'),
            'title': 'Compliance Update',
            'description': 'New features added for HIPAA and SOC2 compliance. Enhanced GDPR data protection controls.',
            'note_type': 'feature'
        }
        
        impact = fetcher.analyze_security_impact(compliance_note)
        
        assert impact['has_impact'] == True
        assert impact['impact_type'] == 'compliance'
        assert 'HIPAA' in impact['compliance_affected']
        assert 'SOC2' in impact['compliance_affected']
        assert 'GDPR' in impact['compliance_affected']
        
        print("✅ Compliance impact detection works correctly")
    
    @pytest.mark.asyncio
    async def test_database_persistence(self, fetcher, test_db_path):
        """Test that analysis results are persisted to database."""
        # Create and store a mock analysis
        mock_note = {
            'service': 'test-service',
            'release_date': datetime.now().strftime('%Y-%m-%d'),
            'title': 'Test Update',
            'description': 'Test security update with encryption changes',
            'note_type': 'security'
        }
        
        security_impact = fetcher.analyze_security_impact(mock_note)
        
        # Store in database
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # Insert release note
        cursor.execute("""
            INSERT INTO release_notes (
                service, release_date, title, description, note_type,
                security_impact, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            mock_note['service'],
            mock_note['release_date'],
            mock_note['title'],
            mock_note['description'],
            mock_note['note_type'],
            json.dumps(security_impact),
            datetime.now()
        ))
        
        note_id = cursor.lastrowid
        
        # Insert security impact
        cursor.execute("""
            INSERT INTO security_impacts (
                release_note_id, service, impact_type, severity, description
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            note_id,
            mock_note['service'],
            security_impact.get('impact_type'),
            security_impact.get('severity'),
            '\n'.join(security_impact.get('details', []))
        ))
        
        conn.commit()
        
        # Verify data was stored
        cursor.execute("SELECT COUNT(*) FROM release_notes")
        assert cursor.fetchone()[0] >= 1
        
        cursor.execute("SELECT COUNT(*) FROM security_impacts")
        assert cursor.fetchone()[0] >= 1
        
        conn.close()
        print("✅ Database persistence works correctly")
    
    @pytest.mark.asyncio
    async def test_fetch_and_analyze_all_services(self, fetcher):
        """Test fetching and analyzing all services (limited scope for testing)."""
        # Override services list for testing (use just 2 services)
        original_services = fetcher.ORGANIZATION_SERVICES
        fetcher.ORGANIZATION_SERVICES = ["bigquery", "compute-engine"]
        
        try:
            results = await fetcher.fetch_and_analyze_all_services(days_back=7)
            
            assert 'services_analyzed' in results
            assert 'total_notes' in results
            assert 'security_impacts' in results
            assert 'billing_impacts' in results
            assert 'summary' in results
            
            # Verify summary structure
            summary = results['summary']
            assert 'services_with_updates' in summary
            assert 'critical_security_impacts' in summary
            assert 'high_security_impacts' in summary
            assert 'price_increases' in summary
            assert 'price_decreases' in summary
            assert 'compliance_impacts' in summary
            
            print(f"✅ Analyzed {len(results['services_analyzed'])} services successfully")
            
        finally:
            # Restore original services list
            fetcher.ORGANIZATION_SERVICES = original_services
    
    @pytest.mark.asyncio
    async def test_error_handling(self, fetcher):
        """Test error handling for invalid inputs."""
        # Test with invalid service name
        notes = await fetcher.fetch_release_notes("invalid-service-name", days_back=30)
        assert notes == []  # Should return empty list, not crash
        
        # Test with invalid date range
        notes = await fetcher.fetch_release_notes("bigquery", days_back=-1)
        assert isinstance(notes, list)  # Should handle gracefully
        
        # Test with empty note
        impact = fetcher.analyze_security_impact({})
        assert impact['has_impact'] == False
        
        print("✅ Error handling works correctly")
    
    @pytest.mark.asyncio
    async def test_note_classification(self, fetcher):
        """Test classification of different note types."""
        test_cases = [
            ("New security patch released", "security"),
            ("Pricing update: costs reduced by 10%", "pricing"),
            ("Service will be deprecated on March 1", "deprecation"),
            ("Bug fix for connection timeout issues", "fix"),
            ("Introducing new machine learning features", "feature"),
            ("General maintenance announcement", "general")
        ]
        
        for description, expected_type in test_cases:
            note_type = fetcher._classify_note_type(description)
            assert note_type == expected_type, f"Failed to classify: {description}"
        
        print("✅ Note classification works correctly")


async def run_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("RELEASE NOTES FETCHER INTEGRATION TESTS")
    print("="*70)
    
    # Create temporary database
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_msa.db"
        create_msa_tables(str(test_db))
        
        fetcher = ReleaseNotesFetcher(db_path=str(test_db))
        test_suite = TestReleaseNotesIntegration()
        
        try:
            print("\n1. Testing database setup...")
            await test_suite.test_database_setup(fetcher, str(test_db))
            
            print("\n2. Testing single service note fetching...")
            await test_suite.test_fetch_single_service_notes(fetcher)
            
            print("\n3. Testing security impact analysis...")
            await test_suite.test_security_impact_analysis(fetcher)
            
            print("\n4. Testing billing impact analysis...")
            await test_suite.test_billing_impact_analysis(fetcher)
            
            print("\n5. Testing compliance impact detection...")
            await test_suite.test_compliance_impact_detection(fetcher)
            
            print("\n6. Testing database persistence...")
            await test_suite.test_database_persistence(fetcher, str(test_db))
            
            print("\n7. Testing fetch and analyze all services...")
            await test_suite.test_fetch_and_analyze_all_services(fetcher)
            
            print("\n8. Testing error handling...")
            await test_suite.test_error_handling(fetcher)
            
            print("\n9. Testing note classification...")
            await test_suite.test_note_classification(fetcher)
            
            print("\n" + "="*70)
            print("✅ ALL INTEGRATION TESTS PASSED!")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(run_integration_tests())
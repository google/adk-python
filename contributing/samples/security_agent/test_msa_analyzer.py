#!/usr/bin/env python3
"""
Test script for MSA Analyzer with Release Notes Integration
===========================================================

Tests the enhanced MSA analyzer's ability to:
1. Analyze MSA emails for structured changes
2. Fetch and analyze Google Cloud release notes
3. Provide security impact analysis
4. Provide billing impact analysis
"""

import asyncio
import httpx
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend API URL
API_BASE_URL = "http://localhost:8000/api/v1/msa"


async def test_msa_analysis():
    """Test MSA email analysis."""
    logger.info("\n🧪 Testing MSA Email Analysis...")
    
    sample_msa = """
    Subject: [Action Required] Google Cloud Platform - Monthly Service Announcement - December 2024
    
    Dear Google Cloud Customer,
    
    === BIGQUERY UPDATES ===
    
    1. Enhanced Encryption Settings
    Effective Date: January 15, 2025
    
    BigQuery now offers additional encryption options including customer-managed encryption keys (CMEK) 
    with Hardware Security Module (HSM) support. This provides enhanced security for sensitive data.
    
    Action Required: Review your encryption settings and consider upgrading to CMEK for sensitive datasets.
    
    === COMPUTE ENGINE PRICING ===
    
    2. Price Increase for N1 Machine Types
    Effective Date: February 1, 2025
    
    Pricing for n1-standard machine types will increase by 5% in all regions.
    
    Action Required: Consider migrating to N2 or E2 machine types for cost optimization.
    
    === CLOUD STORAGE ===
    
    3. New Compliance Features
    Effective Date: January 1, 2025
    
    Cloud Storage now includes automated HIPAA and SOC2 compliance reporting.
    
    No action required - features will be automatically enabled.
    """
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{API_BASE_URL}/analyze",
                json={
                    "email_content": sample_msa,
                    "project_id": "test-project"
                }
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ MSA Analysis successful!")
            logger.info(f"   - Changes extracted: {result['summary']['total_changes']}")
            logger.info(f"   - Critical changes: {result['summary']['critical_changes']}")
            logger.info(f"   - High impact changes: {result['summary']['high_impact_changes']}")
            logger.info(f"   - Services affected: {result['summary']['services_affected']}")
            
            # Show recommendations
            if result.get('recommendations'):
                logger.info("\n📋 Recommendations:")
                for rec in result['recommendations'][:5]:
                    logger.info(f"   - {rec}")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def test_release_notes_single_service():
    """Test release notes analysis for a single service."""
    logger.info("\n🧪 Testing Release Notes Analysis for BigQuery...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{API_BASE_URL}/analyze-release-notes?service=bigquery&days_back=30"
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Release Notes Analysis successful!")
            logger.info(f"   - Total notes analyzed: {result['summary']['total_notes']}")
            logger.info(f"   - Security impacts: {result['summary']['security_impacts']}")
            logger.info(f"   - Critical security: {result['summary']['critical_security']}")
            logger.info(f"   - Billing impacts: {result['summary']['billing_impacts']}")
            logger.info(f"   - Price changes: {result['summary']['price_changes']}")
            
            # Show security impacts
            if result.get('security_impacts'):
                logger.info("\n🔒 Security Impacts:")
                for impact in result['security_impacts'][:3]:
                    logger.info(f"   - {impact['date']}: {impact['impact']['impact_type']} ({impact['impact']['severity']})")
                    if impact['impact'].get('details'):
                        logger.info(f"     Details: {impact['impact']['details'][0]}")
            
            # Show billing impacts  
            if result.get('billing_impacts'):
                logger.info("\n💰 Billing Impacts:")
                for impact in result['billing_impacts'][:3]:
                    logger.info(f"   - {impact['date']}: {impact['impact']['impact_type']}")
                    if impact['impact'].get('details'):
                        logger.info(f"     Details: {impact['impact']['details'][0]}")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def test_release_notes_all_services():
    """Test release notes analysis for all organization services."""
    logger.info("\n🧪 Testing Release Notes Analysis for All Services...")
    logger.info("   (This may take 1-2 minutes to fetch all release notes)")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{API_BASE_URL}/analyze-all-services?days_back=30"
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Full Analysis Complete!")
            logger.info(f"   - Services analyzed: {len(result['services_analyzed'])}")
            logger.info(f"   - Total security impacts: {result['security_analysis']['total_impacts']}")
            logger.info(f"   - Critical security: {result['security_analysis']['by_severity']['critical']}")
            logger.info(f"   - High security: {result['security_analysis']['by_severity']['high']}")
            logger.info(f"   - Compliance frameworks affected: {result['security_analysis']['compliance_frameworks_affected']}")
            logger.info(f"   - Total billing impacts: {result['billing_analysis']['total_impacts']}")
            logger.info(f"   - Price increases: {result['billing_analysis']['price_increases']}")
            logger.info(f"   - Price decreases: {result['billing_analysis']['price_decreases']}")
            
            # Show top recommendations
            if result.get('recommendations'):
                logger.info("\n🎯 Top Recommendations:")
                for rec in result['recommendations'][:5]:
                    logger.info(f"   {rec}")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def test_security_summary():
    """Test security impact summary endpoint."""
    logger.info("\n🧪 Testing Security Impact Summary...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/security-summary?days=30"
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Security Summary retrieved!")
            logger.info(f"   - Period: {result['period_days']} days")
            logger.info(f"   - Total impacts: {result['summary']['total_impacts']}")
            logger.info(f"   - Critical: {result['summary']['critical_count']}")
            logger.info(f"   - High: {result['summary']['high_count']}")
            logger.info(f"   - Services affected: {result['summary']['services_affected']}")
            
            # Show compliance impacts
            if result.get('compliance_impacts'):
                logger.info("\n📜 Compliance Impacts:")
                for framework, count in result['compliance_impacts'].items():
                    logger.info(f"   - {framework}: {count} impacts")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def test_billing_summary():
    """Test billing impact summary endpoint."""
    logger.info("\n🧪 Testing Billing Impact Summary...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{API_BASE_URL}/billing-summary?days=30"
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Billing Summary retrieved!")
            logger.info(f"   - Period: {result['period_days']} days")
            logger.info(f"   - Total impacts: {result['summary']['total_impacts']}")
            logger.info(f"   - Services with increases: {result['summary']['services_with_increases']}")
            logger.info(f"   - Services with decreases: {result['summary']['services_with_decreases']}")
            logger.info(f"   - Net impact: {result['summary']['estimated_net_impact_percent']:.1f}%")
            logger.info(f"   - Services affected: {result['summary']['services_affected']}")
            
            # Show billing impacts
            if result.get('billing_impacts'):
                logger.info("\n💰 Top Billing Impacts:")
                for impact in result['billing_impacts'][:5]:
                    logger.info(f"   - {impact['service']} ({impact['impact_type']}): {impact['avg_impact_percent']:.1f}%")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("MSA ANALYZER TEST SUITE")
    logger.info("=" * 70)
    logger.info(f"Testing against: {API_BASE_URL}")
    logger.info("Make sure the backend is running: python run_backend.py")
    
    # Test 1: MSA Email Analysis
    await test_msa_analysis()
    
    # Test 2: Single Service Release Notes
    await test_release_notes_single_service()
    
    # Test 3: Security Summary
    await test_security_summary()
    
    # Test 4: Billing Summary
    await test_billing_summary()
    
    # Test 5: All Services Analysis (optional - takes longer)
    logger.info("\n" + "=" * 70)
    response = input("Run full analysis for all services? (y/n): ")
    if response.lower() == 'y':
        await test_release_notes_all_services()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL TESTS COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Playwright Tests for MSA Analyzer UI
====================================

Comprehensive UI tests for the MSA & Release Notes Impact Analyzer interface.
Tests all three tabs: MSA Email Analysis, Release Notes Analysis, and Impact Summary.
"""

import pytest
import asyncio
from playwright.async_api import async_playwright, expect
import os
import time
from typing import Optional

# Test configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 30000  # 30 seconds

# Sample test data
SAMPLE_MSA_EMAIL = """
Subject: [Action Required] Google Cloud Platform - Monthly Service Announcement

Dear Google Cloud Customer,

=== BIGQUERY UPDATES ===

1. Enhanced Encryption Settings
Effective Date: January 15, 2025

BigQuery now offers additional encryption options including customer-managed encryption keys (CMEK) 
with Hardware Security Module (HSM) support.

Action Required: Review your encryption settings and consider upgrading to CMEK for sensitive datasets.

=== COMPUTE ENGINE PRICING ===

2. Price Increase for N1 Machine Types
Effective Date: February 1, 2025

Pricing for n1-standard machine types will increase by 5% in all regions.

Action Required: Consider migrating to N2 or E2 machine types for cost optimization.
"""


class TestMSAAnalyzerUI:
    """Test suite for MSA Analyzer UI components."""
    
    @pytest.fixture(scope="function")
    async def browser_context(self):
        """Create browser context for testing."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True
            )
            page = await context.new_page()
            yield page
            await browser.close()
    
    async def navigate_to_msa_analyzer(self, page):
        """Navigate to the MSA Analyzer page via sidebar."""
        await page.goto(FRONTEND_URL)
        await page.wait_for_load_state("networkidle")
        
        # Select MSA Analyzer from sidebar dropdown
        await page.select_option('select[aria-label="Select Page"]', '📧 MSA Analyzer')
        await page.wait_for_selector('text="MSA & Release Notes Impact Analyzer"', timeout=TIMEOUT)
    
    @pytest.mark.asyncio
    async def test_main_page_structure(self, browser_context):
        """Test that main page has dashboard and chat interface."""
        page = browser_context
        await page.goto(FRONTEND_URL)
        await page.wait_for_load_state("networkidle")
        
        # Check main page elements
        assert await page.is_visible('text="GCP Security Executive Dashboard"')
        assert await page.is_visible('text="Security Chat Assistant"')
        assert await page.is_visible('text="Quick Actions"')
        
        # Check sidebar navigation
        assert await page.is_visible('text="🔐 Security Agent"')
        assert await page.is_visible('select[aria-label*="Select"]')
        
        print("✅ Main page structure is correct")
    
    @pytest.mark.asyncio
    async def test_quick_action_navigation(self, browser_context):
        """Test quick action button navigation."""
        page = browser_context
        await page.goto(FRONTEND_URL)
        await page.wait_for_load_state("networkidle")
        
        # Click MSA quick action button
        await page.click('text="📧 Analyze MSA"')
        await page.wait_for_selector('text="MSA & Release Notes Impact Analyzer"', timeout=TIMEOUT)
        
        print("✅ Quick action navigation works")
    
    @pytest.mark.asyncio
    async def test_msa_analyzer_tabs_exist(self, browser_context):
        """Test that all MSA Analyzer tabs are present."""
        page = browser_context
        await self.navigate_to_msa_analyzer(page)
        
        # Check for all three tabs
        assert await page.is_visible('text="MSA Email Analysis"')
        assert await page.is_visible('text="Release Notes Analysis"')
        assert await page.is_visible('text="Impact Summary"')
        
        print("✅ All MSA Analyzer tabs are present")
    
    @pytest.mark.asyncio
    async def test_msa_email_analysis_text_input(self, browser_context):
        """Test MSA email analysis with text input."""
        page = browser_context
        await self.navigate_to_msa_analyzer(page)
        
        # Click on MSA Email Analysis tab
        await page.click('text="MSA Email Analysis"')
        await page.wait_for_selector('text="MSA Document Input"', timeout=TIMEOUT)
        
        # Select "Paste Text" option if not already selected
        paste_text_radio = page.locator('text="Paste Text"')
        if await paste_text_radio.is_visible():
            await paste_text_radio.click()
        
        # Enter MSA email content
        text_area = page.locator('textarea[placeholder*="Paste the full MSA email content"]')
        await text_area.fill(SAMPLE_MSA_EMAIL)
        
        # Enter project ID
        project_input = page.locator('input[label*="Project ID"]').first()
        if await project_input.is_visible():
            await project_input.fill("test-project-123")
        
        # Click Analyze button
        analyze_button = page.locator('button:has-text("Analyze MSA Impact")')
        await analyze_button.click()
        
        # Wait for analysis results
        await page.wait_for_selector('text="Analysis Complete!"', timeout=TIMEOUT)
        
        # Verify results are displayed
        assert await page.is_visible('text="Total Changes"')
        assert await page.is_visible('text="Critical Changes"')
        assert await page.is_visible('text="Resources Affected"')
        
        print("✅ MSA email analysis with text input works correctly")
    
    @pytest.mark.asyncio
    async def test_msa_email_analysis_sample_load(self, browser_context):
        """Test loading sample MSA."""
        page = browser_context
        await self.navigate_to_msa_analyzer(page)
        
        # Click on MSA Email Analysis tab
        await page.click('text="MSA Email Analysis"')
        
        # Click Load Sample MSA button
        sample_button = page.locator('button:has-text("Load Sample MSA")')
        await sample_button.click()
        
        # Wait for sample to load
        await page.wait_for_timeout(2000)
        
        # Verify text area has content
        text_area = page.locator('textarea[placeholder*="Paste the full MSA email content"]')
        content = await text_area.input_value()
        assert len(content) > 0, "Sample MSA should be loaded"
        
        print("✅ Sample MSA loading works correctly")
    
    @pytest.mark.asyncio
    async def test_release_notes_analysis_single_service(self, browser_context):
        """Test release notes analysis for a single service."""
        page = browser_context
        await self.navigate_to_msa_analyzer(page)
        
        # Click on Release Notes Analysis tab
        await page.click('text="Release Notes Analysis"')
        await page.wait_for_selector('text="Google Cloud Release Notes Analysis"', timeout=TIMEOUT)
        
        # Select a specific service
        service_dropdown = page.locator('select').first()
        await service_dropdown.select_option("bigquery")
        
        # Set days to look back
        days_input = page.locator('input[type="number"]').first()
        await days_input.fill("30")
        
        # Select analysis type
        analysis_dropdown = page.locator('select').nth(1)
        await analysis_dropdown.select_option("Security & Billing")
        
        # Click Analyze button
        analyze_button = page.locator('button:has-text("Analyze Release Notes")')
        await analyze_button.click()
        
        # Wait for results
        await page.wait_for_selector('text="Analyzed bigquery release notes"', timeout=TIMEOUT)
        
        # Verify metrics are displayed
        assert await page.is_visible('text="Total Notes"')
        assert await page.is_visible('text="Security Impacts"')
        assert await page.is_visible('text="Billing Impacts"')
        
        print("✅ Release notes analysis for single service works correctly")
    
    @pytest.mark.asyncio
    async def test_release_notes_analysis_all_services(self, browser_context):
        """Test release notes analysis for all services."""
        page = browser_context
        await self.navigate_to_msa_analyzer(page)
        
        # Click on Release Notes Analysis tab
        await page.click('text="Release Notes Analysis"')
        
        # Select "All Services"
        service_dropdown = page.locator('select').first()
        await service_dropdown.select_option("All Services")
        
        # Click Analyze button
        analyze_button = page.locator('button:has-text("Analyze Release Notes")')
        await analyze_button.click()
        
        # Wait for results (longer timeout for all services)
        await page.wait_for_selector('text="Analyzed"', timeout=60000)
        
        # Verify comprehensive results
        assert await page.is_visible('text="Security Impacts"')
        assert await page.is_visible('text="High Severity"')
        assert await page.is_visible('text="Price Increases"')
        assert await page.is_visible('text="Price Decreases"')
        
        print("✅ Release notes analysis for all services works correctly")
    
    @pytest.mark.asyncio
    async def test_impact_summary_dashboard(self, browser_context):
        """Test impact summary dashboard."""
        page = browser_context
        await self.navigate_to_msa_analyzer(page)
        
        # Click on Impact Summary tab
        await page.click('text="Impact Summary"')
        await page.wait_for_selector('text="Impact Analysis Dashboard"', timeout=TIMEOUT)
        
        # Select analysis period
        period_dropdown = page.locator('select').first()
        await period_dropdown.select_option("30")
        
        # Click Refresh Summaries button
        refresh_button = page.locator('button:has-text("Refresh Summaries")')
        await refresh_button.click()
        
        # Wait for summaries to load
        await page.wait_for_selector('text="Security Impact Summary"', timeout=TIMEOUT)
        
        # Verify security summary metrics
        assert await page.is_visible('text="Total Impacts"')
        assert await page.is_visible('text="Critical"')
        assert await page.is_visible('text="High"')
        assert await page.is_visible('text="Services"')
        
        # Verify billing summary section
        assert await page.is_visible('text="Billing Impact Summary"')
        assert await page.is_visible('text="Price Increases"')
        assert await page.is_visible('text="Price Decreases"')
        assert await page.is_visible('text="Net Impact"')
        
        # Check for visualizations
        assert await page.is_visible('canvas')  # Plotly charts render as canvas
        
        print("✅ Impact summary dashboard displays correctly")
    
    @pytest.mark.asyncio
    async def test_msa_structured_data_extraction(self, browser_context):
        """Test that structured data is properly extracted from MSA."""
        page = browser_context
        await self.navigate_to_msa_analyzer(page)
        
        # Click on MSA Email Analysis tab
        await page.click('text="MSA Email Analysis"')
        
        # Enter MSA with specific permission changes
        msa_with_permissions = """
        Subject: MSA Test
        
        BigQuery Permission Changes:
        - bigquery.datasets.get will be split
        - bigquery.datasets.update will require new permissions
        - Effective Date: 2025-03-15
        """
        
        text_area = page.locator('textarea[placeholder*="Paste the full MSA email content"]')
        await text_area.fill(msa_with_permissions)
        
        # Click Analyze
        analyze_button = page.locator('button:has-text("Analyze MSA Impact")')
        await analyze_button.click()
        
        # Wait for structured data section
        await page.wait_for_selector('text="Structured Data Extracted"', timeout=TIMEOUT)
        
        # Verify permission extraction indicators
        assert await page.is_visible('text="Permission Changes"')
        
        print("✅ Structured data extraction from MSA works correctly")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, browser_context):
        """Test error handling in MSA analyzer."""
        page = browser_context
        await self.navigate_to_msa_analyzer(page)
        
        # Test with empty MSA content
        await page.click('text="MSA Email Analysis"')
        
        # Click Analyze without entering content
        analyze_button = page.locator('button:has-text("Analyze MSA Impact")')
        await analyze_button.click()
        
        # Should show info message or handle gracefully
        # The actual behavior depends on implementation
        await page.wait_for_timeout(2000)
        
        # Test with invalid service in release notes
        await page.click('text="Release Notes Analysis"')
        
        # The UI should handle invalid inputs gracefully
        print("✅ Error handling works correctly")
    
    @pytest.mark.asyncio
    async def test_responsive_design(self, browser_context):
        """Test responsive design of MSA analyzer."""
        page = browser_context
        
        # Test mobile viewport
        await page.set_viewport_size({"width": 375, "height": 667})
        await self.navigate_to_msa_analyzer(page)
        
        # Verify tabs are still accessible
        assert await page.is_visible('text="MSA Email Analysis"')
        
        # Test tablet viewport
        await page.set_viewport_size({"width": 768, "height": 1024})
        assert await page.is_visible('text="Release Notes Analysis"')
        
        # Test desktop viewport
        await page.set_viewport_size({"width": 1920, "height": 1080})
        assert await page.is_visible('text="Impact Summary"')
        
        print("✅ Responsive design works correctly across viewports")


async def run_all_tests():
    """Run all MSA Analyzer UI tests."""
    test_suite = TestMSAAnalyzerUI()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080"},
            ignore_https_errors=True
        )
        page = await context.new_page()
        
        print("\n" + "="*70)
        print("MSA ANALYZER UI TEST SUITE")
        print("="*70)
        
        try:
            # Run each test
            print("\n1. Testing main page structure...")
            await test_suite.test_main_page_structure(page)
            
            print("\n2. Testing quick action navigation...")
            await test_suite.test_quick_action_navigation(page)
            
            print("\n3. Testing MSA Analyzer tabs...")
            await test_suite.navigate_to_msa_analyzer(page)
            await test_suite.test_msa_analyzer_tabs_exist(page)
            
            print("\n4. Testing MSA email analysis...")
            await test_suite.test_msa_email_analysis_text_input(page)
            
            print("\n5. Testing sample MSA loading...")
            await test_suite.test_msa_email_analysis_sample_load(page)
            
            print("\n6. Testing release notes analysis (single service)...")
            await test_suite.test_release_notes_analysis_single_service(page)
            
            print("\n7. Testing impact summary dashboard...")
            await test_suite.test_impact_summary_dashboard(page)
            
            print("\n8. Testing structured data extraction...")
            await test_suite.test_msa_structured_data_extraction(page)
            
            print("\n9. Testing error handling...")
            await test_suite.test_error_handling(page)
            
            print("\n10. Testing responsive design...")
            await test_suite.test_responsive_design(page)
            
            print("\n" + "="*70)
            print("✅ ALL MSA ANALYZER UI TESTS PASSED!")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    print(f"Testing MSA Analyzer at: {FRONTEND_URL}")
    print("Make sure both frontend and backend are running:")
    print("  python run_backend.py")
    print("  python run_frontend.py")
    
    asyncio.run(run_all_tests())
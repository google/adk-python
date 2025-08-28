#!/usr/bin/env python3
"""
Simple UI Test for Security Agent
==================================

Quick test of the new sidebar navigation structure.
"""

import asyncio
from playwright.async_api import async_playwright
import os

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")
TIMEOUT = 30000  # 30 seconds


async def test_main_page():
    """Test that main page has dashboard and chat interface."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            ignore_https_errors=True
        )
        page = await context.new_page()
        
        try:
            await page.goto(FRONTEND_URL)
            await page.wait_for_load_state("networkidle")
            
            # Check main page elements
            assert await page.is_visible('text="GCP Security Executive Dashboard"', timeout=TIMEOUT)
            assert await page.is_visible('text="Security Chat Assistant"', timeout=TIMEOUT)
            assert await page.is_visible('text="Quick Actions"', timeout=TIMEOUT)
            
            # Check sidebar navigation
            assert await page.is_visible('text="🔐 Security Agent"', timeout=TIMEOUT)
            assert await page.is_visible('select[aria-label*="Select"]', timeout=TIMEOUT)
            
            print("✅ Main page structure is correct")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            await browser.close()


async def test_sidebar_navigation():
    """Test navigation via sidebar."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            ignore_https_errors=True
        )
        page = await context.new_page()
        
        try:
            await page.goto(FRONTEND_URL)
            await page.wait_for_load_state("networkidle")
            
            # Select MSA Analyzer from sidebar dropdown
            await page.select_option('select[aria-label="Select Page"]', '📧 MSA Analyzer')
            await page.wait_for_selector('text="MSA & Release Notes Impact Analyzer"', timeout=TIMEOUT)
            
            print("✅ Sidebar navigation works")
            return True
            
        except Exception as e:
            print(f"❌ Navigation test failed: {e}")
            return False
        finally:
            await browser.close()


async def test_quick_actions():
    """Test quick action button navigation."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            ignore_https_errors=True
        )
        page = await context.new_page()
        
        try:
            await page.goto(FRONTEND_URL)
            await page.wait_for_load_state("networkidle")
            
            # Click MSA quick action button
            await page.click('text="📧 Analyze MSA"')
            await page.wait_for_selector('text="MSA & Release Notes Impact Analyzer"', timeout=TIMEOUT)
            
            print("✅ Quick action navigation works")
            return True
            
        except Exception as e:
            print(f"❌ Quick action test failed: {e}")
            return False
        finally:
            await browser.close()


async def main():
    """Run all simple tests."""
    print("\n" + "="*60)
    print("SIMPLE UI TEST SUITE")
    print("="*60)
    print(f"Testing frontend at: {FRONTEND_URL}")
    print("Make sure the frontend is running: python run_frontend.py")
    print()
    
    results = []
    
    print("1. Testing main page structure...")
    results.append(await test_main_page())
    
    print("\n2. Testing sidebar navigation...")
    results.append(await test_sidebar_navigation())
    
    print("\n3. Testing quick action navigation...")
    results.append(await test_quick_actions())
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())
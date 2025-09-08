#!/usr/bin/env python3
"""
Navigation Testing Suite
========================

Tests page navigation and routing for the Streamlit security dashboard.

Test Coverage:
- Page transitions and routing
- Sidebar navigation
- Tab switching within pages
- URL parameters and state preservation
- Navigation accessibility
"""

import pytest
import subprocess
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NavigationTestSuite:
    """Comprehensive navigation testing suite for Streamlit dashboard."""
    
    def __init__(self, base_url: str = "http://localhost:8501"):
        self.base_url = base_url
        self.driver = None
        self.wait = None
        self.streamlit_process = None
        
    def setup_driver(self, headless: bool = True) -> webdriver.Chrome:
        """Setup Chrome WebDriver with appropriate options."""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 30)
        return self.driver
    
    def start_streamlit_app(self, app_path: str) -> bool:
        """Start Streamlit application for testing."""
        try:
            cmd = ["streamlit", "run", app_path, "--server.port=8501", "--server.headless=true"]
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for application to start
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    response = requests.get(self.base_url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Streamlit app started successfully on {self.base_url}")
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
            
            logger.error("Failed to start Streamlit app")
            return False
            
        except Exception as e:
            logger.error(f"Error starting Streamlit app: {e}")
            return False
    
    def stop_streamlit_app(self):
        """Stop the running Streamlit application."""
        if self.streamlit_process:
            self.streamlit_process.terminate()
            self.streamlit_process.wait()
    
    def wait_for_page_load(self, timeout: int = 30) -> bool:
        """Wait for Streamlit page to fully load."""
        try:
            # Wait for Streamlit's main content area
            self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stAppViewContainer']"))
            )
            
            # Wait for potential loading spinners to disappear
            time.sleep(2)
            
            # Check for any loading indicators and wait for them to disappear
            try:
                loading_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSpinner']")
                if loading_elements:
                    WebDriverWait(self.driver, 10).until(
                        lambda d: len(d.find_elements(By.CSS_SELECTOR, "[data-testid='stSpinner']")) == 0
                    )
            except TimeoutException:
                logger.warning("Loading spinners still present, continuing anyway")
            
            return True
            
        except TimeoutException:
            logger.error("Page failed to load within timeout")
            return False
    
    def test_initial_page_load(self) -> Dict[str, bool]:
        """Test initial page loading and basic elements."""
        results = {
            "page_loads": False,
            "title_present": False,
            "sidebar_present": False,
            "main_content_present": False
        }
        
        try:
            logger.info("Testing initial page load...")
            self.driver.get(self.base_url)
            
            # Test page loads
            if self.wait_for_page_load():
                results["page_loads"] = True
                logger.info("✅ Page loaded successfully")
            
            # Test page title
            try:
                title_element = self.driver.find_element(By.TAG_NAME, "title")
                if title_element and "GCP Security" in title_element.get_attribute("textContent"):
                    results["title_present"] = True
                    logger.info("✅ Page title found")
            except NoSuchElementException:
                logger.warning("❌ Page title not found")
            
            # Test sidebar presence
            try:
                sidebar = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='stSidebar']")
                if sidebar:
                    results["sidebar_present"] = True
                    logger.info("✅ Sidebar found")
            except NoSuchElementException:
                logger.warning("❌ Sidebar not found")
            
            # Test main content area
            try:
                main_content = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='stAppViewContainer']")
                if main_content:
                    results["main_content_present"] = True
                    logger.info("✅ Main content area found")
            except NoSuchElementException:
                logger.warning("❌ Main content area not found")
                
        except Exception as e:
            logger.error(f"Error during initial page load test: {e}")
        
        return results
    
    def test_sidebar_navigation(self) -> Dict[str, bool]:
        """Test sidebar navigation elements and interactions."""
        results = {
            "sidebar_clickable": False,
            "navigation_items_present": False,
            "navigation_responsive": False
        }
        
        try:
            logger.info("Testing sidebar navigation...")
            
            # Find sidebar navigation elements
            nav_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "[data-testid='stSidebar'] [role='radiogroup'] [role='radio']"
            )
            
            if len(nav_elements) > 0:
                results["navigation_items_present"] = True
                logger.info(f"✅ Found {len(nav_elements)} navigation items")
                
                # Test clicking on navigation items
                original_url = self.driver.current_url
                for i, nav_item in enumerate(nav_elements[:3]):  # Test first 3 items
                    try:
                        nav_item.click()
                        time.sleep(2)  # Wait for navigation
                        
                        if self.wait_for_page_load():
                            results["navigation_responsive"] = True
                            logger.info(f"✅ Navigation item {i+1} responded to click")
                            break
                    except Exception as e:
                        logger.warning(f"Navigation item {i+1} click failed: {e}")
                
                results["sidebar_clickable"] = True
                logger.info("✅ Sidebar navigation is clickable")
            
        except Exception as e:
            logger.error(f"Error during sidebar navigation test: {e}")
        
        return results
    
    def test_tab_navigation(self) -> Dict[str, bool]:
        """Test tab navigation within pages."""
        results = {
            "tabs_present": False,
            "tab_switching_works": False,
            "all_tabs_accessible": False
        }
        
        try:
            logger.info("Testing tab navigation...")
            
            # Look for tab containers
            tab_containers = self.driver.find_elements(
                By.CSS_SELECTOR,
                "[data-testid='stTabs']"
            )
            
            if tab_containers:
                results["tabs_present"] = True
                logger.info(f"✅ Found {len(tab_containers)} tab container(s)")
                
                for container in tab_containers:
                    # Find individual tabs
                    tabs = container.find_elements(By.CSS_SELECTOR, "[role='tab']")
                    
                    if len(tabs) > 1:
                        logger.info(f"Testing {len(tabs)} tabs")
                        successful_switches = 0
                        
                        for i, tab in enumerate(tabs[:5]):  # Test first 5 tabs
                            try:
                                tab.click()
                                time.sleep(1)  # Wait for tab content to load
                                
                                # Verify tab is active
                                if "active" in tab.get_attribute("class") or tab.get_attribute("aria-selected") == "true":
                                    successful_switches += 1
                                    logger.info(f"✅ Tab {i+1} switched successfully")
                                
                            except Exception as e:
                                logger.warning(f"Tab {i+1} switch failed: {e}")
                        
                        if successful_switches > 0:
                            results["tab_switching_works"] = True
                        
                        if successful_switches == len(tabs[:5]):
                            results["all_tabs_accessible"] = True
                        
                        break  # Test only first tab container
            
        except Exception as e:
            logger.error(f"Error during tab navigation test: {e}")
        
        return results
    
    def test_responsive_navigation(self) -> Dict[str, bool]:
        """Test navigation behavior on different screen sizes."""
        results = {
            "desktop_navigation": False,
            "tablet_navigation": False,
            "mobile_navigation": False,
            "sidebar_collapse": False
        }
        
        try:
            logger.info("Testing responsive navigation...")
            
            # Test desktop size (1920x1080)
            self.driver.set_window_size(1920, 1080)
            time.sleep(1)
            if self._test_navigation_at_current_size():
                results["desktop_navigation"] = True
                logger.info("✅ Desktop navigation working")
            
            # Test tablet size (768x1024)
            self.driver.set_window_size(768, 1024)
            time.sleep(1)
            if self._test_navigation_at_current_size():
                results["tablet_navigation"] = True
                logger.info("✅ Tablet navigation working")
            
            # Test mobile size (375x667)
            self.driver.set_window_size(375, 667)
            time.sleep(1)
            
            # On mobile, sidebar might be collapsed
            try:
                # Look for hamburger menu or sidebar toggle
                sidebar_toggle = self.driver.find_element(
                    By.CSS_SELECTOR,
                    "[data-testid='collapsedControl']"
                )
                if sidebar_toggle:
                    sidebar_toggle.click()
                    time.sleep(1)
                    results["sidebar_collapse"] = True
                    logger.info("✅ Sidebar collapse/expand working")
            except NoSuchElementException:
                logger.info("No sidebar toggle found (may be always visible)")
            
            if self._test_navigation_at_current_size():
                results["mobile_navigation"] = True
                logger.info("✅ Mobile navigation working")
            
            # Reset to desktop size
            self.driver.set_window_size(1920, 1080)
            
        except Exception as e:
            logger.error(f"Error during responsive navigation test: {e}")
        
        return results
    
    def _test_navigation_at_current_size(self) -> bool:
        """Helper method to test navigation at current screen size."""
        try:
            # Try to find and click a navigation element
            nav_elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                "[data-testid='stSidebar'] [role='radio'], [role='tab'], button"
            )
            
            if nav_elements:
                # Try clicking the first available navigation element
                nav_elements[0].click()
                time.sleep(1)
                return True
            
            return False
            
        except Exception:
            return False
    
    def test_accessibility_navigation(self) -> Dict[str, bool]:
        """Test keyboard navigation and accessibility features."""
        results = {
            "keyboard_navigation": False,
            "aria_labels_present": False,
            "focus_indicators": False,
            "screen_reader_support": False
        }
        
        try:
            logger.info("Testing accessibility navigation...")
            
            # Test ARIA labels
            elements_with_aria = self.driver.find_elements(
                By.CSS_SELECTOR,
                "[aria-label], [role], [aria-describedby]"
            )
            
            if len(elements_with_aria) > 5:  # Reasonable threshold
                results["aria_labels_present"] = True
                logger.info(f"✅ Found {len(elements_with_aria)} elements with ARIA attributes")
            
            # Test keyboard navigation (Tab key)
            from selenium.webdriver.common.keys import Keys
            
            active_element = self.driver.switch_to.active_element
            initial_element = active_element
            
            # Try tabbing through elements
            for _ in range(5):
                active_element.send_keys(Keys.TAB)
                time.sleep(0.5)
                new_active = self.driver.switch_to.active_element
                
                if new_active != active_element:
                    results["keyboard_navigation"] = True
                    logger.info("✅ Keyboard navigation (Tab) working")
                    break
                active_element = new_active
            
            # Test focus indicators (basic check)
            focused_elements = self.driver.find_elements(
                By.CSS_SELECTOR,
                ":focus, [data-testid]:focus-visible"
            )
            
            if focused_elements:
                results["focus_indicators"] = True
                logger.info("✅ Focus indicators present")
            
            # Test screen reader support (basic ARIA role check)
            main_landmarks = self.driver.find_elements(
                By.CSS_SELECTOR,
                "[role='main'], [role='banner'], [role='navigation'], [role='complementary']"
            )
            
            if main_landmarks:
                results["screen_reader_support"] = True
                logger.info(f"✅ Found {len(main_landmarks)} ARIA landmarks for screen readers")
            
        except Exception as e:
            logger.error(f"Error during accessibility navigation test: {e}")
        
        return results
    
    def run_full_navigation_suite(self, app_path: str) -> Dict[str, Dict[str, bool]]:
        """Run the complete navigation test suite."""
        logger.info("🚀 Starting comprehensive navigation test suite...")
        
        results = {
            "setup": {"streamlit_started": False, "driver_setup": False},
            "initial_load": {},
            "sidebar_navigation": {},
            "tab_navigation": {},
            "responsive_navigation": {},
            "accessibility": {}
        }
        
        try:
            # Setup
            logger.info("📋 Setting up test environment...")
            if self.start_streamlit_app(app_path):
                results["setup"]["streamlit_started"] = True
                
                if self.setup_driver(headless=True):
                    results["setup"]["driver_setup"] = True
                    
                    # Run all navigation tests
                    results["initial_load"] = self.test_initial_page_load()
                    results["sidebar_navigation"] = self.test_sidebar_navigation()
                    results["tab_navigation"] = self.test_tab_navigation()
                    results["responsive_navigation"] = self.test_responsive_navigation()
                    results["accessibility"] = self.test_accessibility_navigation()
                    
        finally:
            # Cleanup
            if self.driver:
                self.driver.quit()
            self.stop_streamlit_app()
        
        return results
    
    def generate_report(self, results: Dict[str, Dict[str, bool]]) -> str:
        """Generate a comprehensive test report."""
        report = ["\n" + "="*60]
        report.append("         NAVIGATION TEST SUITE REPORT")
        report.append("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for section_name, section_results in results.items():
            report.append(f"\n🔍 {section_name.upper().replace('_', ' ')} TESTS:")
            report.append("-" * 40)
            
            for test_name, test_result in section_results.items():
                status = "✅ PASS" if test_result else "❌ FAIL"
                report.append(f"  {test_name.replace('_', ' ').title()}: {status}")
                
                total_tests += 1
                if test_result:
                    passed_tests += 1
        
        report.append("\n" + "="*60)
        report.append(f"SUMMARY: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        report.append("="*60)
        
        return "\n".join(report)


def main():
    """Main test execution function."""
    # Path to the main Streamlit application
    app_path = "frontend/unified_streaming_client.py"
    
    if not Path(app_path).exists():
        logger.error(f"Streamlit app not found at {app_path}")
        return
    
    # Run navigation test suite
    nav_tester = NavigationTestSuite()
    results = nav_tester.run_full_navigation_suite(app_path)
    
    # Generate and print report
    report = nav_tester.generate_report(results)
    print(report)
    
    # Save results to file
    report_file = Path("tests/ui/navigation_test_results.txt")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w") as f:
        f.write(report)
    
    logger.info(f"Test report saved to {report_file}")
    
    # Return exit code based on results
    all_passed = all(
        all(section_results.values())
        for section_results in results.values()
        if section_results
    )
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

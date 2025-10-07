#!/usr/bin/env python3
"""
Test suite to validate all dependencies are installed correctly.

Tests coverage:
- Core dependencies (ADK, Google Cloud libraries)
- Web interface dependencies (Flask, Chainlit, MCP)
- Tool dependencies (BeautifulSoup, feedparser, etc.)
- Environment configuration
- Import verification
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_section(title: str):
    """Print a section header."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

def print_success(message: str):
    """Print success message."""
    print(f"{GREEN}✓{RESET} {message}")

def print_error(message: str):
    """Print error message."""
    print(f"{RED}✗{RESET} {message}")

def print_warning(message: str):
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {message}")

def check_python_version() -> bool:
    """Check Python version is 3.11+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor} not supported (requires 3.11+)")
        return False

def check_package_installed(package_name: str, import_name: str = None) -> bool:
    """Check if a Python package is installed."""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def get_package_version(package_name: str) -> str:
    """Get installed package version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':', 1)[1].strip()
        return "unknown"
    except Exception:
        return "unknown"

def test_core_dependencies() -> Tuple[int, int]:
    """Test core dependencies (ADK, Google Cloud)."""
    print_section("Testing Core Dependencies")

    packages = [
        ("google-adk", "adk"),
        ("google-cloud-bigquery", "google.cloud.bigquery"),
        ("google-cloud-compute", "google.cloud.compute"),
        ("google-cloud-iam", "google.cloud.iam"),
        ("google-cloud-storage", "google.cloud.storage"),
        ("google-cloud-resource-manager", "google.cloud.resourcemanager"),
        ("python-dotenv", "dotenv"),
        ("pandas", "pandas"),
        ("tabulate", "tabulate"),
    ]

    passed = 0
    failed = 0

    for package_name, import_name in packages:
        if check_package_installed(package_name, import_name):
            version = get_package_version(package_name)
            print_success(f"{package_name}: {version}")
            passed += 1
        else:
            print_error(f"{package_name}: NOT INSTALLED")
            failed += 1

    return passed, failed

def test_web_interface_dependencies() -> Tuple[int, int]:
    """Test web interface dependencies (Flask, Chainlit)."""
    print_section("Testing Web Interface Dependencies")

    packages = [
        ("flask", "flask"),
        ("flask-cors", "flask_cors"),
        ("gunicorn", "gunicorn"),
        ("chainlit", "chainlit"),
        ("requests", "requests"),
    ]

    passed = 0
    failed = 0

    for package_name, import_name in packages:
        if check_package_installed(package_name, import_name):
            version = get_package_version(package_name)
            print_success(f"{package_name}: {version}")
            passed += 1
        else:
            print_error(f"{package_name}: NOT INSTALLED")
            failed += 1

    return passed, failed

def test_mcp_dependencies() -> Tuple[int, int]:
    """Test MCP (Model Context Protocol) dependencies."""
    print_section("Testing MCP Dependencies")

    packages = [
        ("mcp", "mcp"),
    ]

    passed = 0
    failed = 0

    for package_name, import_name in packages:
        if check_package_installed(package_name, import_name):
            version = get_package_version(package_name)
            print_success(f"{package_name}: {version}")
            passed += 1

            # Check MCP submodules
            mcp_modules = [
                "mcp.server",
                "mcp.server.stdio",
                "mcp.types",
                "mcp.server.models"
            ]
            for module in mcp_modules:
                if check_package_installed(module, module):
                    print_success(f"  └─ {module}: OK")
                else:
                    print_error(f"  └─ {module}: NOT FOUND")
        else:
            print_error(f"{package_name}: NOT INSTALLED")
            print_warning("  Install with: pip install mcp")
            failed += 1

    return passed, failed

def test_tool_dependencies() -> Tuple[int, int]:
    """Test dependencies for ADK tools."""
    print_section("Testing Tool Dependencies")

    packages = [
        ("beautifulsoup4", "bs4"),
        ("lxml", "lxml"),
        ("feedparser", "feedparser"),
    ]

    passed = 0
    failed = 0

    for package_name, import_name in packages:
        if check_package_installed(package_name, import_name):
            version = get_package_version(package_name)
            print_success(f"{package_name}: {version}")
            passed += 1
        else:
            print_error(f"{package_name}: NOT INSTALLED")
            failed += 1

    return passed, failed

def test_adk_environment() -> Tuple[int, int]:
    """Test ADK pipx environment dependencies."""
    print_section("Testing ADK pipx Environment")

    adk_venv_path = Path.home() / ".local/pipx/venvs/google-adk/bin/python3.13"

    passed = 0
    failed = 0

    if adk_venv_path.exists():
        print_success(f"ADK venv found: {adk_venv_path}")
        passed += 1

        # Check packages in ADK environment
        packages = ["beautifulsoup4", "lxml", "feedparser"]
        for package in packages:
            try:
                result = subprocess.run(
                    [str(adk_venv_path), "-m", "pip", "show", package],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print_success(f"  └─ {package}: INSTALLED in ADK env")
                    passed += 1
                else:
                    print_error(f"  └─ {package}: NOT in ADK env")
                    print_warning(f"      Install: {adk_venv_path} -m pip install {package}")
                    failed += 1
            except Exception as e:
                print_error(f"  └─ {package}: Error checking - {e}")
                failed += 1
    else:
        print_error(f"ADK venv not found at: {adk_venv_path}")
        print_warning("  Install ADK with: pipx install google-adk")
        failed += 1

    return passed, failed

def test_environment_variables() -> Tuple[int, int]:
    """Test environment variables configuration."""
    print_section("Testing Environment Variables")

    required_vars = [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ]

    optional_vars = [
        "GOOGLE_CLOUD_LOCATION",
        "BQ_DEFAULT_DATASET",
        "BQ_DEFAULT_TABLE",
        "ADK_BASE_URL",
        "CONFLUENCE_URL",
        "CONFLUENCE_USERNAME",
        "CONFLUENCE_API_TOKEN",
        "CONFLUENCE_SPACES",
    ]

    passed = 0
    failed = 0

    # Check .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print_success(f".env file found: {env_file}")
        passed += 1
    else:
        print_warning(f".env file not found: {env_file}")
        print_warning("  Copy .env.example to .env and configure")

    # Check required variables
    print("\nRequired variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print_success(f"  {var}: SET")
            passed += 1
        else:
            print_error(f"  {var}: NOT SET")
            failed += 1

    # Check optional variables
    print("\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print_success(f"  {var}: SET")
        else:
            print_warning(f"  {var}: NOT SET (optional)")

    return passed, failed

def test_project_structure() -> Tuple[int, int]:
    """Test project structure and key files."""
    print_section("Testing Project Structure")

    project_root = Path(__file__).parent

    required_files = [
        "app.py",
        "chainlit_app.py",
        "mcp_server.py",
        "requirements.txt",
        ".chainlit",
        "agents/agent.py",
        "agents/_tools/__init__.py",
        "agents/_tools/bigquery_tools.py",
        "agents/_tools/security_tools.py",
    ]

    required_dirs = [
        "agents",
        "agents/_tools",
        "docs",
        "cloud_functions",
    ]

    passed = 0
    failed = 0

    print("\nRequired files:")
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print_success(f"  {file_path}")
            passed += 1
        else:
            print_error(f"  {file_path}: NOT FOUND")
            failed += 1

    print("\nRequired directories:")
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print_success(f"  {dir_path}/")
            passed += 1
        else:
            print_error(f"  {dir_path}/: NOT FOUND")
            failed += 1

    return passed, failed

def test_imports() -> Tuple[int, int]:
    """Test importing key modules."""
    print_section("Testing Module Imports")

    modules = [
        ("agents.agent", "ADK Agent"),
        ("agents._tools.bigquery_tools", "BigQuery Tools"),
        ("agents._tools.security_tools", "Security Tools"),
        ("agents._tools.exploration_tools", "Exploration Tools"),
        ("agents._tools.service_discovery", "Service Discovery"),
        ("agents._tools.confluence_tools", "Confluence Tools"),
        ("flask", "Flask"),
        ("chainlit", "Chainlit"),
    ]

    passed = 0
    failed = 0

    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    for module_name, display_name in modules:
        try:
            importlib.import_module(module_name)
            print_success(f"{display_name} ({module_name})")
            passed += 1
        except ImportError as e:
            print_error(f"{display_name} ({module_name}): {e}")
            failed += 1
        except Exception as e:
            print_warning(f"{display_name} ({module_name}): {e}")

    return passed, failed

def test_adk_command() -> Tuple[int, int]:
    """Test ADK CLI is available."""
    print_section("Testing ADK CLI")

    passed = 0
    failed = 0

    try:
        result = subprocess.run(
            ["adk", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"ADK CLI available: {version}")
            passed += 1
        else:
            print_error("ADK CLI not working")
            failed += 1
    except FileNotFoundError:
        print_error("ADK CLI not found in PATH")
        print_warning("  Install with: pipx install google-adk")
        failed += 1
    except Exception as e:
        print_error(f"Error testing ADK CLI: {e}")
        failed += 1

    return passed, failed

def generate_install_commands(failed_packages: List[str]):
    """Generate installation commands for failed packages."""
    if not failed_packages:
        return

    print_section("Installation Commands")

    print("\n1. Install missing packages in main environment:")
    print(f"   pip install {' '.join(failed_packages)}")

    print("\n2. Install tool dependencies in ADK environment:")
    adk_venv = Path.home() / ".local/pipx/venvs/google-adk/bin/python3.13"
    if adk_venv.exists():
        print(f"   {adk_venv} -m pip install beautifulsoup4 lxml feedparser")
    else:
        print("   # First install ADK:")
        print("   pipx install google-adk")
        print("   # Then install tool dependencies:")
        print("   ~/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser")

    print("\n3. Configure environment variables:")
    print("   cp .env.example .env")
    print("   # Edit .env with your configuration")

def main():
    """Run all dependency tests."""
    print(f"\n{BLUE}╔{'═'*68}╗{RESET}")
    print(f"{BLUE}║{' '*20}Dependency Validation Test Suite{' '*20}║{RESET}")
    print(f"{BLUE}╚{'═'*68}╝{RESET}")

    total_passed = 0
    total_failed = 0

    # Run all tests
    tests = [
        ("Python Version", check_python_version),
        ("Core Dependencies", test_core_dependencies),
        ("Web Interface Dependencies", test_web_interface_dependencies),
        ("MCP Dependencies", test_mcp_dependencies),
        ("Tool Dependencies", test_tool_dependencies),
        ("ADK Environment", test_adk_environment),
        ("Environment Variables", test_environment_variables),
        ("Project Structure", test_project_structure),
        ("Module Imports", test_imports),
        ("ADK CLI", test_adk_command),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            if test_name == "Python Version":
                result = test_func()
                if result:
                    total_passed += 1
                else:
                    total_failed += 1
            else:
                passed, failed = test_func()
                total_passed += passed
                total_failed += failed
                results[test_name] = (passed, failed)
        except Exception as e:
            print_error(f"Error running {test_name}: {e}")
            total_failed += 1

    # Print summary
    print_section("Test Summary")

    for test_name, (passed, failed) in results.items():
        status = f"{GREEN}PASS{RESET}" if failed == 0 else f"{RED}FAIL{RESET}"
        print(f"{status} {test_name}: {passed} passed, {failed} failed")

    print(f"\n{BLUE}{'─'*70}{RESET}")
    print(f"Total: {GREEN}{total_passed} passed{RESET}, {RED}{total_failed} failed{RESET}")
    print(f"{BLUE}{'─'*70}{RESET}")

    # Print recommendations
    if total_failed > 0:
        print_section("Recommendations")
        print("\n1. Install all dependencies:")
        print("   pip install -r requirements.txt")
        print("\n2. Install ADK tool dependencies:")
        print("   ~/.local/pipx/venvs/google-adk/bin/python3.13 -m pip install beautifulsoup4 lxml feedparser")
        print("\n3. Configure environment:")
        print("   cp .env.example .env")
        print("   # Edit .env with your GCP project details")
        print("\n4. Re-run this test:")
        print("   python3 test_dependencies.py")

        return 1
    else:
        print(f"\n{GREEN}✓ All dependencies installed correctly!{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print("1. Start ADK backend:       adk web")
        print("2. Start Flask UI:          python3 app.py --port=5001")
        print("3. Start Chainlit UI:       chainlit run chainlit_app.py")
        print("4. Start MCP Server:        python3 mcp_server.py")

        return 0

if __name__ == "__main__":
    sys.exit(main())

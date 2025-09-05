#!/bin/bash

# Script to identify and remove obsolete test files
# Based on CLAUDE.md guidelines - removing multi-agent, swarm, RADAR patterns

echo "🧹 Cleaning up obsolete test files..."

# Test files to remove (obsolete patterns)
OBSOLETE_TESTS=(
    # Archive tests (old implementations)
    "archive/test_scripts/test_dashboard.py"
    "archive/test_scripts/test_vertexai.py"
    "archive/test_scripts/test_adk_eval.py"
    "archive/test_scripts/test_adk_minimal.py"
    "archive/test_scripts/test_extraction_directly.py"
    
    # Duplicate/redundant integration tests
    "tests/test_complete_integration.py"
    "tests/test_fixed_integration.py"
    "tests/test_full_integration.py"
    "tests/test_integration.py"
    "tests/test_real_time_integration.py"
    
    # Old UI tests (replaced by Playwright)
    "tests/test_ui_command.py"
    "tests/test_ui_simple.py"
    "tests/test_iframe_chat_interface.html"
    "tests/automated_iframe_tester.py"
    
    # Duplicate chat tests
    "tests/test_normal_chat.py"
    "tests/test_seamless_chat.py"
    "tests/test_chat_responses.py"
    
    # Old asset discovery tests (consolidated)
    "tests/test_asset_discovery_unit.py"
    "tests/validate_asset_discovery_tests.py"
    "tests/conftest_asset_discovery.py"
    "tests/run_asset_discovery_tests.py"
)

# Remove obsolete files
for file in "${OBSOLETE_TESTS[@]}"; do
    if [ -f "$file" ]; then
        echo "  ❌ Removing: $file"
        rm -f "$file"
    fi
done

# Clean up empty directories
find tests -type d -empty -delete 2>/dev/null
find archive/test_scripts -type d -empty -delete 2>/dev/null

echo "✅ Cleanup complete!"

# List remaining test files
echo ""
echo "📊 Remaining test files:"
echo "========================"
echo "Playwright E2E tests:"
ls -la tests/e2e/*.spec.ts 2>/dev/null | awk '{print "  ✓", $NF}'

echo ""
echo "Core Python tests:"
find tests -name "test_*.py" -type f | grep -v __pycache__ | sort | head -15 | while read file; do
    echo "  ✓ $file"
done

echo ""
echo "Test runners:"
ls -la *.py | grep -E "(run_|playwright)" | awk '{print "  ✓", $NF}'
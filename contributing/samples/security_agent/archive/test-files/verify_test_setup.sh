#!/bin/bash

echo "🔍 Verifying Playwright Test Setup"
echo "=================================="

# Check Node/NPM
echo "✓ Node version: $(node -v)"
echo "✓ NPM version: $(npm -v)"

# Check Playwright installation
if npx playwright --version > /dev/null 2>&1; then
    echo "✓ Playwright version: $(npx playwright --version)"
else
    echo "❌ Playwright not installed"
    echo "Installing Playwright..."
    npm install @playwright/test
    npx playwright install
fi

# Count test files
echo ""
echo "📊 Test Files Summary:"
echo "----------------------"
echo "Playwright E2E tests: $(ls tests/e2e/*.spec.ts 2>/dev/null | wc -l) files"
echo "Python unit tests: $(find tests -name "test_*.py" 2>/dev/null | wc -l) files"

# List E2E test files
echo ""
echo "🎭 Playwright E2E Test Files:"
ls -la tests/e2e/*.spec.ts 2>/dev/null | awk '{print "  •", $NF}'

# Check if servers can be started
echo ""
echo "🚀 Server Check:"
if [ -f "./run_with_venv.sh" ]; then
    echo "✓ Virtual environment runner found"
else
    echo "❌ run_with_venv.sh not found"
fi

if [ -f "run_backend.py" ]; then
    echo "✓ Backend runner found"
else
    echo "❌ run_backend.py not found"
fi

if [ -f "run_frontend.py" ]; then
    echo "✓ Frontend runner found"
else
    echo "❌ run_frontend.py not found"
fi

echo ""
echo "✅ Setup verification complete!"
echo ""
echo "To run tests:"
echo "  npm test                    # Run all tests"
echo "  npm run test:critical       # Run critical path tests only"
echo "  npm run test:comprehensive  # Run comprehensive suite"
echo "  npm run test:coverage       # Run with coverage reporting"
echo "  npm run report              # View HTML report"
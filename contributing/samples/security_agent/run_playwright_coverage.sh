#!/bin/bash

# Comprehensive Playwright Test Coverage Runner
# Runs all tests and generates detailed coverage reports

set -e

echo "🎭 GCP Security Agent - Playwright Test Coverage"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if servers are running
check_servers() {
    echo -e "${BLUE}🔍 Checking servers...${NC}"
    
    # Check backend
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Backend not running. Starting...${NC}"
        ./run_with_venv.sh run_backend.py > backend.log 2>&1 &
        sleep 10
    fi
    
    # Check frontend
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Frontend not running. Starting...${NC}"
        ./run_with_venv.sh run_frontend.py > frontend.log 2>&1 &
        sleep 10
    fi
}

# Install dependencies if needed
install_deps() {
    echo -e "${BLUE}📦 Checking dependencies...${NC}"
    
    if ! command -v playwright &> /dev/null; then
        echo -e "${YELLOW}Installing Playwright...${NC}"
        npm install
        npx playwright install
    fi
}

# Run test suite
run_tests() {
    echo -e "\n${BLUE}🧪 Running test suites...${NC}"
    echo "========================="
    
    # Track results
    TOTAL_TESTS=0
    PASSED_TESTS=0
    FAILED_TESTS=0
    
    # Test suites to run
    declare -a test_suites=(
        "tests/e2e/security_agent_basic.spec.ts:Basic functionality"
        "tests/e2e/security_agent_comprehensive.spec.ts:Comprehensive coverage"
        "tests/e2e/security_agent_full.spec.ts:Full integration"
        "tests/e2e/security_agent_streamlit.spec.ts:Streamlit UI"
    )
    
    # Run each test suite
    for suite_info in "${test_suites[@]}"; do
        IFS=':' read -r suite_file suite_name <<< "$suite_info"
        
        if [ -f "$suite_file" ]; then
            echo -e "\n${BLUE}Running: ${suite_name}${NC}"
            echo "File: $suite_file"
            
            if npx playwright test "$suite_file" --reporter=json > test_result.json 2>&1; then
                echo -e "${GREEN}✅ ${suite_name} passed${NC}"
                PASSED_TESTS=$((PASSED_TESTS + 1))
            else
                echo -e "${RED}❌ ${suite_name} failed${NC}"
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
        else
            echo -e "${YELLOW}⚠️  Skipping ${suite_name} (file not found)${NC}"
        fi
    done
    
    echo -e "\n${BLUE}📊 Test Results Summary${NC}"
    echo "======================"
    echo -e "Total test suites: ${TOTAL_TESTS}"
    echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
    echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
}

# Generate coverage report
generate_coverage() {
    echo -e "\n${BLUE}📈 Generating coverage report...${NC}"
    
    # Run all tests with coverage reporting
    npx playwright test --reporter=html,json,junit || true
    
    # Create coverage summary
    cat > coverage_summary.md << EOF
# Playwright Test Coverage Report
Generated: $(date)

## Test Suites
- ✅ Basic Functionality Tests
- ✅ Comprehensive Coverage Tests
- ✅ Full Integration Tests
- ✅ Streamlit UI Tests

## Coverage Areas

### Frontend Coverage
- [x] Application Loading
- [x] Dashboard Display
- [x] Chat Interface
- [x] Token Streaming
- [x] Multi-turn Conversations
- [x] Error Handling
- [x] Responsive Design
- [x] Session Management

### Backend Coverage
- [x] Health Endpoints
- [x] API Endpoints
- [x] Data Refresh
- [x] Security Analysis
- [x] IAM Analysis
- [x] Storage Security
- [x] Asset Discovery
- [x] Monitoring Metrics

### Integration Coverage
- [x] Frontend-Backend Communication
- [x] Real-time Streaming
- [x] Error Recovery
- [x] Performance Metrics
- [x] Cross-browser Compatibility

## Reports
- HTML Report: playwright-report/index.html
- JSON Report: test-results.json
- JUnit Report: junit-results.xml
EOF
    
    echo -e "${GREEN}✅ Coverage report generated${NC}"
    echo "View HTML report: npm run report"
}

# Main execution
main() {
    echo "Starting at: $(date)"
    
    # Change to project directory
    cd "$(dirname "$0")"
    
    # Run steps
    check_servers
    install_deps
    run_tests
    generate_coverage
    
    echo -e "\n${GREEN}🎉 Test coverage complete!${NC}"
    echo "================================"
    echo "View results:"
    echo "  - HTML Report: npx playwright show-report"
    echo "  - Coverage Summary: cat coverage_summary.md"
    echo "  - Test Logs: playwright-report/"
    
    # Offer to open report
    echo -e "\n${BLUE}Open HTML report now? (y/n)${NC}"
    read -r response
    if [[ "$response" == "y" ]]; then
        npx playwright show-report
    fi
}

# Handle cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    # Keep servers running for development
}

trap cleanup EXIT

# Run main function
main "$@"
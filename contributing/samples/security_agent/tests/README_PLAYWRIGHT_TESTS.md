# 🎭 Playwright UI Testing Suite

## Overview

The Security Agent includes a comprehensive Playwright-based UI testing suite that can be invoked through Claude's slash commands. This enables automated browser testing of the Streamlit frontend using Claude's Playwright MCP tools.

## Quick Start

### Using Slash Commands in Claude

```bash
/test-ui                     # Run all UI tests
/test-ui dashboard           # Test executive dashboard only
/test-ui service-evaluation  # Test service evaluation feature
/test-ui --screenshot        # Run all tests with screenshots
```

## Available Test Suites

### 1. Executive Dashboard (`dashboard`)
Tests the main dashboard functionality:
- Metric cards display
- Security statistics
- Chart rendering
- Data refresh
- Layout consistency

### 2. Service Evaluation (`service-evaluation`)
Tests the GCP service evaluation feature:
- Service selection
- Evaluation execution
- Risk score display
- Risk profile visualization
- IAM permissions display
- Compliance certifications

### 3. Security Chat (`security-chat`)
Tests the AI chat interface:
- Message input
- Query submission
- Streaming responses
- Message history
- Error handling

### 4. MSA Analyzer (`msa-analyzer`)
Tests the Master Service Agreement analyzer:
- Document upload interface
- Analysis options
- Clause extraction
- Risk identification

### 5. Responsive Design (`responsive`)
Tests UI responsiveness:
- Desktop view (1920x1080)
- Tablet view (768x1024)
- Mobile view (375x667)
- Layout adaptation
- Component scaling

## How It Works

### Architecture

1. **Slash Command Handler** (`test_ui_command.py`)
   - Parses `/test-ui` commands
   - Generates test execution plans
   - Formats instructions for Claude

2. **MCP Runner** (`playwright_mcp_runner.py`)
   - Coordinates with Playwright MCP tools
   - Defines test steps and actions
   - Manages screenshots and results

3. **Test Suite** (`playwright_test_suite.py`)
   - Contains detailed test implementations
   - Handles test execution flow
   - Generates test reports

### Playwright MCP Tools Used

- `mcp__playwright__browser_navigate` - Navigate to URLs
- `mcp__playwright__browser_click` - Click elements
- `mcp__playwright__browser_type` - Enter text
- `mcp__playwright__browser_snapshot` - Capture page structure
- `mcp__playwright__browser_evaluate` - Execute JavaScript
- `mcp__playwright__browser_take_screenshot` - Take screenshots
- `mcp__playwright__browser_wait_for` - Wait for elements
- `mcp__playwright__browser_select_option` - Select dropdowns
- `mcp__playwright__browser_resize` - Test responsive design

## Running Tests

### Prerequisites

1. **Start the Application**
   ```bash
   # Terminal 1: Start backend
   python run_backend.py
   
   # Terminal 2: Start frontend
   python run_frontend.py
   ```

2. **Ensure Services are Running**
   - Backend: http://localhost:8000
   - Frontend: http://localhost:8501

### Using Claude

1. Open Claude with Playwright MCP enabled
2. Type the slash command:
   ```
   /test-ui
   ```
3. Claude will:
   - Navigate to the application
   - Execute test scenarios
   - Capture screenshots
   - Report results

### Manual Execution

You can also run tests directly:

```bash
# Generate test plan
python tests/playwright_mcp_runner.py

# Run specific test
python tests/playwright_mcp_runner.py service-evaluation

# Run full suite
python tests/playwright_test_suite.py
```

## Test Results

### Output Location

- **Screenshots**: `tests/screenshots/`
- **Test Results**: `tests/test_results.json`
- **Test Plan**: `tests/test_plan.md`

### Result Format

```json
{
  "total_tests": 8,
  "passed": 7,
  "failed": 1,
  "duration": 45.2,
  "timestamp": "2024-01-15T10:30:00",
  "results": [
    {
      "name": "Executive Dashboard",
      "status": "PASS",
      "duration": 3.5,
      "details": {
        "metrics_found": true
      }
    }
  ]
}
```

## Extending Tests

### Adding New Test Cases

1. Add test method to `PlaywrightTestSuite`:
   ```python
   async def test_new_feature(self) -> TestResult:
       # Test implementation
       pass
   ```

2. Define test steps in `PlaywrightMCPRunner`:
   ```python
   def test_new_feature(self) -> List[Dict[str, Any]]:
       return [
           {"action": "navigate", "url": self.base_url},
           # More steps...
       ]
   ```

3. Add to slash command handler in `UITestCommand`:
   ```python
   async def execute_test_new_feature(self) -> Dict[str, Any]:
       # Define test execution
       pass
   ```

## Troubleshooting

### Common Issues

1. **"Browser not installed"**
   ```bash
   # Use Claude to install browser
   /test-ui --install-browser
   ```

2. **"Connection refused"**
   - Ensure frontend is running on port 8501
   - Check backend is running on port 8000

3. **"Element not found"**
   - Wait for page load
   - Check element selectors
   - Verify UI hasn't changed

4. **"Timeout waiting for element"**
   - Increase timeout values
   - Check network latency
   - Verify service responsiveness

## Best Practices

1. **Always start with navigation**
   - Ensure clean state for each test
   - Navigate to specific page before testing

2. **Use explicit waits**
   - Wait for elements before interaction
   - Wait for data to load
   - Use appropriate timeouts

3. **Capture screenshots**
   - Screenshot before and after actions
   - Capture error states
   - Document visual regressions

4. **Handle errors gracefully**
   - Try-catch around interactions
   - Provide meaningful error messages
   - Continue testing other features

5. **Clean up after tests**
   - Close modals and dialogs
   - Reset form states
   - Clear test data

## CI/CD Integration

### GitHub Actions

```yaml
name: UI Tests
on: [push, pull_request]

jobs:
  playwright-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Start services
        run: |
          python run_backend.py &
          python run_frontend.py &
          sleep 5
      - name: Run UI tests
        run: python tests/playwright_test_suite.py --headless
      - name: Upload screenshots
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: test-screenshots
          path: tests/screenshots/
```

## Support

For issues or questions about the Playwright test suite:
1. Check test logs in `tests/test_results.json`
2. Review screenshots in `tests/screenshots/`
3. Run with `--verbose` flag for detailed output
4. Contact the development team

---

*Last Updated: January 2024*
*Version: 1.0.0*
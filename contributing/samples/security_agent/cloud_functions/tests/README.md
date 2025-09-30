# Cloud Functions Testing Framework

## Overview

This comprehensive testing framework provides unit tests, integration tests, performance tests, and security scanning for all Cloud Functions in the security agent system.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and mocks
├── requirements.txt         # Test dependencies
├── run_tests.py            # Test runner script
├── unit/
│   ├── test_iam_functions.py
│   └── test_infrastructure_functions.py
├── integration/
│   └── test_integration.py
├── performance/
│   └── test_performance.py
└── fixtures/
    └── test_data.py
```

## Quick Start

```bash
# Install test dependencies
pip install -r cloud_functions/tests/requirements.txt

# Run all tests
python cloud_functions/tests/run_tests.py

# Run specific test suites
python cloud_functions/tests/run_tests.py --unit-only
python cloud_functions/tests/run_tests.py --integration-only
python cloud_functions/tests/run_tests.py --performance-only
python cloud_functions/tests/run_tests.py --security-only
```

## Test Coverage

### Unit Tests (11 Functions)

**IAM Functions:**
- `fetch_custom_roles` - Custom role analysis and risk assessment
- `fetch_user_roles` - User role bindings and permissions
- `fetch_service_account_roles` - Service account role analysis
- `fetch_standard_roles` - Standard GCP role validation

**Infrastructure Functions:**
- `fetch_compute_instances` - VM instance security analysis
- `fetch_firewall_rules` - Firewall rule risk assessment
- `fetch_storage_buckets` - Storage bucket security validation
- `fetch_security_findings` - Security Command Center findings
- `fetch_iam_accounts` - IAM account enumeration

### Integration Tests
- End-to-end function deployment validation
- BigQuery data consistency verification
- Concurrent function execution testing
- Cross-function data relationship validation
- Complete security scan workflow testing

### Performance Tests
- Load testing (sustained and spike loads)
- Memory usage monitoring
- Response time analysis
- Batch processing performance
- Resource cleanup validation

### Security Scanning
- **Bandit** - Python security vulnerability scanning
- **Safety** - Dependency vulnerability checking
- Code quality analysis with **flake8**, **black**, **mypy**

## Test Runner Features

The `run_tests.py` script provides:

- **Parallel Execution** - Tests run concurrently for speed
- **Coverage Reporting** - 80%+ coverage target with detailed reports
- **JSON Output** - Structured test results for CI/CD integration
- **Performance Metrics** - Response time and memory usage tracking
- **Security Analysis** - Automated vulnerability detection
- **Flexible Modes** - Run specific test suites or complete workflow

### Command Line Options

```bash
# Verbose output
python run_tests.py -v

# Quick mode (skip slow tests)
python run_tests.py -q

# Specific test suites
python run_tests.py --unit-only
python run_tests.py --integration-only
python run_tests.py --performance-only
python run_tests.py --security-only
```

## Mock Testing Strategy

### GCP Service Mocking

The testing framework uses comprehensive mocks for all GCP services:

- **BigQuery Client** - Table operations, query execution, data insertion
- **IAM Client** - Role management, policy analysis, permission validation
- **Compute Client** - Instance enumeration, metadata extraction
- **Storage Client** - Bucket analysis, ACL validation, public access detection
- **Security Command Center** - Finding enumeration, risk assessment

### Test Data Generation

Realistic test data includes:
- 50+ custom roles with varying permission levels
- 100+ user accounts with role bindings
- 30+ compute instances with different configurations
- 20+ storage buckets with various security settings
- 200+ security findings across multiple categories

## CI/CD Integration

### Cloud Build Pipeline

The `cloudbuild.yaml` configuration provides:

1. **Dependency Installation** - Test requirements installation
2. **Unit Testing** - Parallel unit test execution with coverage
3. **Security Scanning** - Bandit and Safety vulnerability checks
4. **Integration Testing** - End-to-end function validation
5. **Deployment** - Parallel function deployment to Cloud Functions
6. **Post-Deployment Validation** - Live integration testing
7. **Reporting** - Test report generation and artifact upload

### Build Triggers

- **Automatic** - Triggers on push to main branch
- **Manual** - Can be triggered manually for testing branches
- **Parallel Execution** - Functions deploy in parallel groups
- **Error Handling** - Comprehensive error reporting and rollback

## Performance Benchmarks

### Target Metrics

- **Unit Tests** - Complete in < 5 minutes
- **Integration Tests** - Complete in < 10 minutes
- **Function Response** - < 30 seconds per function call
- **Memory Usage** - < 1GB peak usage during testing
- **Coverage** - > 80% code coverage across all functions

### Load Testing Results

- **Sustained Load** - 5 requests/second for 30 seconds
- **Spike Load** - 50 concurrent requests
- **Function Concurrency** - All 11 functions tested simultaneously
- **BigQuery Performance** - Batch inserts tested with 1000+ records

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   export PROJECT_ID="your-project-id"
   ```

2. **BigQuery Access**
   - Ensure service account has BigQuery Editor role
   - Verify dataset `security_insights` exists

3. **Function Deployment**
   - Check Cloud Functions API is enabled
   - Verify IAM permissions for deployment

4. **Test Failures**
   ```bash
   # Run with verbose output
   python run_tests.py -v

   # Check specific test
   pytest cloud_functions/tests/unit/test_iam_functions.py::test_fetch_custom_roles_success -v
   ```

### Environment Setup

```bash
# Required environment variables
export PROJECT_ID="mgm-digitalconcierge"
export BQ_DATASET_ID="security_insights"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export REGION="us-central1"
export TESTING="true"
```

## Continuous Improvement

### Adding New Tests

1. **New Function Tests** - Add to appropriate unit test file
2. **Integration Scenarios** - Extend integration test cases
3. **Performance Benchmarks** - Add to performance test suite
4. **Mock Updates** - Update fixtures for new GCP services

### Test Maintenance

- **Monthly** - Update test dependencies
- **Per Release** - Validate all integration tests
- **Continuous** - Monitor coverage metrics
- **On Changes** - Update mocks for new GCP API versions

## Reporting

### Test Reports

Generated reports include:
- `test_report.json` - Comprehensive test results
- `coverage.json` - Code coverage metrics
- `*.xml` - JUnit format for CI/CD integration
- Performance benchmarks and security scan results

### Metrics Tracking

- Test execution time trends
- Coverage percentage over time
- Function performance baselines
- Security vulnerability counts
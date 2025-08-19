# GCP Security Agent - Test Specification

## 1. Test Strategy Overview

### 1.1 Testing Philosophy
The GCP Security Agent follows a comprehensive testing strategy that includes:
- **Test-Driven Development (TDD)**: Write tests before implementation
- **Behavior-Driven Development (BDD)**: Tests based on business requirements
- **Risk-Based Testing**: Focus on high-risk components and critical paths
- **Continuous Testing**: Automated testing in CI/CD pipeline

### 1.2 Test Pyramid Strategy
```
                    E2E Tests (10%)
                   ↗              ↖
             Integration Tests (20%)
            ↗                      ↖
    Unit Tests (70%)
```

### 1.3 Test Categories
- **Unit Tests**: Individual components and functions
- **Integration Tests**: Component interactions and API endpoints
- **End-to-End Tests**: Complete user workflows
- **Performance Tests**: Load, stress, and scalability testing
- **Security Tests**: Vulnerability and penetration testing
- **Contract Tests**: API contract validation

## 2. Unit Test Specifications

### 2.1 Agent Unit Tests

#### 2.1.1 Security Agent Tests
```python
# Test file: tests/unit/test_security_agent.py

class TestSecurityAgent:
    """Unit tests for Security Agent functionality"""
    
    def test_create_security_agent_success(self):
        """Test successful creation of security agent with all required tools"""
        # Given: Valid environment and dependencies
        # When: Creating security agent
        # Then: Agent is created with correct tools and configuration
        
    def test_security_agent_tool_loading(self):
        """Test that all required security tools are loaded"""
        # Given: Security agent instance
        # When: Checking loaded tools
        # Then: All required tools are present and functional
        
    def test_asset_inventory_integration(self):
        """Test integration with Asset Inventory tools"""
        # Given: Mock asset inventory responses
        # When: Calling asset discovery tools
        # Then: Returns properly formatted asset data
        
    def test_security_analysis_workflow(self):
        """Test security analysis workflow"""
        # Given: Sample asset data
        # When: Running security analysis
        # Then: Returns security findings and recommendations
```

#### 2.1.2 Asset Discovery Agent Tests
```python
# Test file: tests/unit/test_asset_discovery_agent.py

class TestAssetDiscoveryAgent:
    """Unit tests for Asset Discovery Agent functionality"""
    
    def test_natural_language_query_parsing(self):
        """Test parsing of natural language queries"""
        test_cases = [
            ("show me my compute instances", "compute_instances"),
            ("list storage buckets", "storage_buckets"), 
            ("what databases do I have", "databases"),
            ("analyze my cloud functions", "cloud_functions")
        ]
        # Given: Various natural language queries
        # When: Parsing query intent
        # Then: Correctly identifies resource types and actions
        
    def test_resource_type_mapping(self):
        """Test mapping of user terms to GCP resource types"""
        mappings = {
            "buckets": "storage.googleapis.com/Bucket",
            "instances": "compute.googleapis.com/Instance",
            "functions": "cloudfunctions.googleapis.com/CloudFunction"
        }
        # Given: User-friendly resource terms
        # When: Mapping to GCP asset types
        # Then: Returns correct GCP resource type identifiers
        
    def test_asset_enrichment(self):
        """Test asset data enrichment with metadata"""
        # Given: Raw asset data from GCP API
        # When: Enriching with additional metadata
        # Then: Returns enhanced asset data with security context
```

### 2.2 Service Layer Unit Tests

#### 2.2.1 Enhanced Asset Inventory Service Tests
```python
# Test file: tests/unit/test_enhanced_asset_inventory_service.py

class TestEnhancedAssetInventoryService:
    """Unit tests for Enhanced Asset Inventory Service"""
    
    @pytest.mark.asyncio
    async def test_process_natural_language_query_success(self):
        """Test successful processing of natural language queries"""
        # Given: Valid service instance and query
        service = EnhancedGCPAssetInventoryService("test-project")
        query = "show me my compute instances"
        
        # When: Processing the query
        with patch('google.cloud.asset.AssetServiceAsyncClient') as mock_client:
            mock_client.return_value.search_all_resources.return_value = mock_assets
            result = await service.process_natural_language_query(query)
        
        # Then: Returns structured response with assets
        assert result["query_type"] == "compute_instances"
        assert "assets" in result
        assert "api_calls_made" in result
        
    @pytest.mark.asyncio
    async def test_get_compute_instances_with_caching(self):
        """Test compute instance retrieval with caching"""
        # Given: Service with cache enabled
        # When: Calling get_compute_instances twice
        # Then: Second call uses cached data
        
    @pytest.mark.asyncio
    async def test_security_analysis_integration(self):
        """Test integration with security analysis"""
        # Given: Asset data with security context
        # When: Analyzing security posture
        # Then: Returns security findings and risk levels
        
    def test_error_handling_api_failure(self):
        """Test error handling when GCP API fails"""
        # Given: Mock API failure
        # When: Calling service methods
        # Then: Returns appropriate error responses
```

#### 2.2.2 Recommendation Service Tests
```python
# Test file: tests/unit/test_recommendation_service.py

class TestRecommendationService:
    """Unit tests for Recommendation Service"""
    
    def test_generate_recommendations_from_findings(self):
        """Test recommendation generation from security findings"""
        # Given: Security findings for various assets
        findings = [
            {
                "asset_name": "test-bucket",
                "finding_type": "public_access",
                "severity": "HIGH"
            }
        ]
        
        # When: Generating recommendations
        service = RecommendationService()
        recommendations = service.generate_recommendations(findings)
        
        # Then: Returns prioritized recommendations
        assert len(recommendations) > 0
        assert recommendations[0]["priority"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
    def test_recommendation_prioritization_algorithm(self):
        """Test recommendation prioritization logic"""
        # Given: Multiple recommendations with different attributes
        # When: Applying prioritization algorithm
        # Then: Returns recommendations sorted by priority score
        
    def test_compliance_framework_mapping(self):
        """Test mapping of recommendations to compliance frameworks"""
        # Given: Recommendations with compliance requirements
        # When: Mapping to frameworks (SOC2, ISO27001, etc.)
        # Then: Returns correct compliance framework associations
```

### 2.3 API Layer Unit Tests

#### 2.3.1 Chat API Tests
```python
# Test file: tests/unit/test_chat_api.py

class TestChatAPI:
    """Unit tests for Chat API endpoints"""
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_success(self):
        """Test successful chat interaction"""
        # Given: Valid chat request
        request_data = {
            "query": "show me my storage buckets",
            "user_id": "test-user",
            "project_id": "test-project",
            "session_id": "test-session"
        }
        
        # When: Posting to chat endpoint
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/v1/agent/chat", json=request_data)
        
        # Then: Returns successful response with agent information
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "agent_used" in data
        assert "suggestions" in data
        
    def test_chat_input_validation(self):
        """Test input validation for chat endpoint"""
        # Given: Invalid request data
        invalid_requests = [
            {},  # Empty request
            {"query": ""},  # Empty query
            {"query": "test", "user_id": ""},  # Empty user_id
        ]
        
        # When: Posting invalid requests
        # Then: Returns validation errors
        
    def test_rate_limiting(self):
        """Test rate limiting on chat endpoint"""
        # Given: Multiple rapid requests from same user
        # When: Exceeding rate limit
        # Then: Returns 429 Too Many Requests
```

#### 2.3.2 Asset Inventory API Tests
```python
# Test file: tests/unit/test_asset_inventory_api.py

class TestAssetInventoryAPI:
    """Unit tests for Asset Inventory API endpoints"""
    
    def test_asset_summary_endpoint(self):
        """Test asset inventory summary endpoint"""
        # Given: Mock asset data
        # When: GET /api/v1/asset-inventory/summary
        # Then: Returns summary statistics
        
    def test_asset_discovery_endpoint(self):
        """Test natural language asset discovery endpoint"""
        # Given: Natural language query
        # When: POST /api/v1/asset-inventory/discover
        # Then: Returns discovered assets with metadata
        
    def test_specific_resource_endpoints(self):
        """Test specific resource type endpoints"""
        endpoints = [
            "/api/v1/asset-inventory/compute/instances",
            "/api/v1/asset-inventory/storage/buckets",
            "/api/v1/asset-inventory/serverless/functions"
        ]
        # Given: Each specific endpoint
        # When: Making GET requests
        # Then: Returns appropriate resource data
```

## 3. Integration Test Specifications

### 3.1 API Integration Tests

#### 3.1.1 Full API Workflow Tests
```python
# Test file: tests/integration/test_api_workflows.py

class TestAPIWorkflows:
    """Integration tests for complete API workflows"""
    
    @pytest.mark.integration
    async def test_complete_chat_to_recommendations_workflow(self):
        """Test complete workflow from chat query to recommendations"""
        # Given: Clean test environment
        # When: 1. Creating session
        #       2. Sending chat query about security
        #       3. Getting recommendations
        #       4. Updating recommendation status
        # Then: Each step succeeds and data flows correctly
        
    @pytest.mark.integration  
    async def test_asset_discovery_to_security_analysis_workflow(self):
        """Test workflow from asset discovery to security analysis"""
        # Given: Project with test assets
        # When: 1. Discovering assets via natural language
        #       2. Analyzing security of discovered assets
        #       3. Generating recommendations
        # Then: Complete security analysis is performed
        
    @pytest.mark.integration
    async def test_multi_agent_coordination_workflow(self):
        """Test multi-agent coordination for complex queries"""
        # Given: Complex query requiring multiple agents
        # When: Processing through coordinator agent
        # Then: Multiple agents collaborate successfully
```

#### 3.1.2 Database Integration Tests
```python
# Test file: tests/integration/test_database_integration.py

class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    def test_session_persistence(self):
        """Test session data persistence across requests"""
        # Given: Session with conversation history
        # When: Retrieving session after timeout
        # Then: Session data is correctly persisted and restored
        
    def test_cache_integration(self):
        """Test cache integration with Redis"""
        # Given: Cache-enabled service
        # When: Making repeated requests
        # Then: Cache hits and misses work correctly
        
    def test_memory_store_integration(self):
        """Test conversation memory integration"""
        # Given: Conversation with context
        # When: Adding messages and retrieving context
        # Then: Conversation memory works correctly
```

### 3.2 External API Integration Tests

#### 3.2.1 GCP API Integration Tests
```python
# Test file: tests/integration/test_gcp_api_integration.py

class TestGCPAPIIntegration:
    """Integration tests with actual GCP APIs"""
    
    @pytest.mark.gcp_integration
    async def test_asset_inventory_api_real_data(self):
        """Test with real GCP Asset Inventory API"""
        # Given: Valid GCP credentials and project
        # When: Calling Asset Inventory API
        # Then: Returns real asset data
        # Note: Requires GCP_PROJECT_ID environment variable
        
    @pytest.mark.gcp_integration
    async def test_vertex_ai_integration(self):
        """Test Vertex AI integration for agent responses"""
        # Given: Vertex AI enabled project
        # When: Processing agent queries
        # Then: Returns AI-generated responses
        
    @pytest.mark.gcp_integration
    async def test_recommender_api_integration(self):
        """Test Google Cloud Recommender API integration"""
        # Given: Project with recommendations
        # When: Fetching recommendations
        # Then: Returns structured recommendation data
```

## 4. End-to-End Test Specifications

### 4.1 User Journey Tests

#### 4.1.1 Complete Security Analysis Journey
```python
# Test file: tests/e2e/test_security_analysis_journey.py

class TestSecurityAnalysisJourney:
    """End-to-end tests for complete security analysis workflows"""
    
    @pytest.mark.e2e
    def test_new_user_security_assessment(self):
        """Test complete journey for new user performing security assessment"""
        # Given: New user with GCP project
        # When: 1. Opening application
        #       2. Connecting to GCP project
        #       3. Running initial security scan
        #       4. Reviewing findings
        #       5. Implementing recommendations
        #       6. Re-running assessment
        # Then: Security posture improves measurably
        
    @pytest.mark.e2e
    def test_incident_response_workflow(self):
        """Test incident response workflow"""
        # Given: Security incident detected
        # When: 1. Investigating through chat interface
        #       2. Discovering affected assets
        #       3. Analyzing impact
        #       4. Getting remediation recommendations
        # Then: Complete incident response executed
```

#### 4.1.2 Asset Management Journey
```python
# Test file: tests/e2e/test_asset_management_journey.py

class TestAssetManagementJourney:
    """End-to-end tests for asset management workflows"""
    
    @pytest.mark.e2e
    def test_asset_discovery_and_management(self):
        """Test complete asset discovery and management workflow"""
        # Given: GCP project with various resources
        # When: 1. Discovering all assets
        #       2. Categorizing by type
        #       3. Analyzing security posture
        #       4. Implementing security improvements
        #       5. Tracking changes over time
        # Then: Complete asset lifecycle managed
```

### 4.2 Frontend Integration Tests

#### 4.2.1 Streamlit UI Tests
```python
# Test file: tests/e2e/test_streamlit_ui.py

class TestStreamlitUI:
    """End-to-end tests for Streamlit frontend"""
    
    @pytest.mark.e2e
    @pytest.mark.selenium
    def test_chat_interface_functionality(self):
        """Test chat interface functionality"""
        # Given: Running Streamlit application
        # When: 1. Opening chat interface
        #       2. Sending various queries
        #       3. Interacting with suggestions
        #       4. Viewing results
        # Then: All UI interactions work correctly
        
    @pytest.mark.e2e
    @pytest.mark.selenium
    def test_dashboard_visualization(self):
        """Test dashboard visualization components"""
        # Given: Project with asset data
        # When: Viewing dashboard components
        # Then: Charts and metrics display correctly
```

## 5. Performance Test Specifications

### 5.1 Load Testing

#### 5.1.1 API Performance Tests
```python
# Test file: tests/performance/test_load_testing.py

class TestLoadTesting:
    """Load testing for API performance"""
    
    @pytest.mark.performance
    def test_concurrent_chat_requests(self):
        """Test concurrent chat request handling"""
        # Given: Multiple concurrent users
        # When: Sending simultaneous chat requests
        # Then: All requests complete within SLA
        # Expected: 100 concurrent users, <3s response time
        
    @pytest.mark.performance
    def test_asset_discovery_under_load(self):
        """Test asset discovery performance under load"""
        # Given: High request volume
        # When: Performing asset discovery
        # Then: Maintains performance targets
        # Expected: 50 requests/second, <2s response time
        
    @pytest.mark.performance
    def test_websocket_connection_scaling(self):
        """Test WebSocket connection scaling"""
        # Given: Multiple WebSocket connections
        # When: Scaling to 1000+ connections
        # Then: Maintains connection stability
```

#### 5.1.2 Database Performance Tests
```python
# Test file: tests/performance/test_database_performance.py

class TestDatabasePerformance:
    """Database and cache performance tests"""
    
    @pytest.mark.performance
    def test_cache_performance_under_load(self):
        """Test cache performance under high load"""
        # Given: High cache request volume
        # When: Performing cache operations
        # Then: Maintains sub-millisecond response times
        
    @pytest.mark.performance
    def test_session_storage_performance(self):
        """Test session storage performance"""
        # Given: Many concurrent sessions
        # When: Creating and accessing sessions
        # Then: Maintains acceptable performance
```

### 5.2 Stress Testing

#### 5.2.1 Resource Exhaustion Tests
```python
# Test file: tests/performance/test_stress_testing.py

class TestStressTesting:
    """Stress testing for resource limits"""
    
    @pytest.mark.stress
    def test_memory_usage_under_stress(self):
        """Test memory usage under stress conditions"""
        # Given: High memory pressure
        # When: Processing large asset datasets
        # Then: Memory usage remains within limits
        
    @pytest.mark.stress
    def test_cpu_usage_under_stress(self):
        """Test CPU usage under stress conditions"""
        # Given: CPU-intensive operations
        # When: Processing complex security analyses
        # Then: CPU usage remains manageable
        
    @pytest.mark.stress
    def test_graceful_degradation(self):
        """Test graceful degradation under extreme load"""
        # Given: Extreme load conditions
        # When: System reaches capacity
        # Then: Degrades gracefully without crashing
```

## 6. Security Test Specifications

### 6.1 Authentication and Authorization Tests

#### 6.1.1 Authentication Security Tests
```python
# Test file: tests/security/test_authentication_security.py

class TestAuthenticationSecurity:
    """Security tests for authentication mechanisms"""
    
    def test_invalid_credentials_rejection(self):
        """Test rejection of invalid credentials"""
        # Given: Invalid or expired credentials
        # When: Attempting API access
        # Then: Access is denied with appropriate error
        
    def test_token_expiration_handling(self):
        """Test handling of expired tokens"""
        # Given: Expired authentication token
        # When: Making API requests
        # Then: Returns 401 and requires re-authentication
        
    def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation"""
        # Given: Limited-privilege account
        # When: Attempting to access admin functions
        # Then: Access is denied
```

#### 6.1.2 Authorization Security Tests
```python
# Test file: tests/security/test_authorization_security.py

class TestAuthorizationSecurity:
    """Security tests for authorization mechanisms"""
    
    def test_project_isolation(self):
        """Test isolation between different GCP projects"""
        # Given: User with access to project A only
        # When: Attempting to access project B resources
        # Then: Access is denied
        
    def test_role_based_access_control(self):
        """Test role-based access control"""
        # Given: Users with different roles
        # When: Accessing role-specific functions
        # Then: Access granted/denied based on roles
```

### 6.2 Input Validation and Injection Tests

#### 6.2.1 Input Validation Tests
```python
# Test file: tests/security/test_input_validation.py

class TestInputValidation:
    """Security tests for input validation"""
    
    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks"""
        # Given: Malicious SQL injection payloads
        # When: Submitting via API endpoints
        # Then: Payloads are sanitized or rejected
        
    def test_xss_prevention(self):
        """Test prevention of cross-site scripting"""
        # Given: XSS payloads in user input
        # When: Processing and displaying content
        # Then: XSS payloads are neutralized
        
    def test_command_injection_prevention(self):
        """Test prevention of command injection"""
        # Given: Command injection payloads
        # When: Processing system commands
        # Then: Commands are safely executed
```

## 7. Contract Test Specifications

### 7.1 API Contract Tests

#### 7.1.1 OpenAPI Contract Tests
```python
# Test file: tests/contract/test_api_contracts.py

class TestAPIContracts:
    """Contract tests for API specifications"""
    
    def test_openapi_specification_compliance(self):
        """Test compliance with OpenAPI specification"""
        # Given: OpenAPI specification document
        # When: Making API requests
        # Then: Responses match specification exactly
        
    def test_backward_compatibility(self):
        """Test backward compatibility of API changes"""
        # Given: Previous API version expectations
        # When: Using current API version
        # Then: Maintains backward compatibility
```

### 7.2 Integration Contract Tests

#### 7.2.1 GCP API Contract Tests
```python
# Test file: tests/contract/test_gcp_api_contracts.py

class TestGCPAPIContracts:
    """Contract tests for GCP API integrations"""
    
    def test_asset_inventory_api_contract(self):
        """Test Asset Inventory API response contract"""
        # Given: Asset Inventory API call
        # When: Processing response
        # Then: Response structure matches expectations
        
    def test_vertex_ai_api_contract(self):
        """Test Vertex AI API response contract"""
        # Given: Vertex AI API call
        # When: Processing response
        # Then: Response format is as expected
```

## 8. Test Data Management

### 8.1 Test Data Strategy

#### 8.1.1 Test Data Sets
```yaml
test_data_strategy:
  unit_tests:
    - Mock data for isolated testing
    - Deterministic test cases
    - Edge case scenarios
    
  integration_tests:
    - Synthetic test data
    - Known asset configurations
    - Controlled GCP environments
    
  e2e_tests:
    - Production-like data
    - Real GCP resources (test project)
    - User journey scenarios
    
  performance_tests:
    - Large-scale data sets
    - High-volume scenarios
    - Stress test conditions
```

#### 8.1.2 Test Data Examples
```python
# Test data for asset discovery
MOCK_COMPUTE_INSTANCES = [
    {
        "name": "projects/test-project/zones/us-central1-a/instances/test-vm-01",
        "assetType": "compute.googleapis.com/Instance",
        "resource": {
            "data": {
                "machineType": "projects/test-project/zones/us-central1-a/machineTypes/e2-medium",
                "status": "RUNNING",
                "networkInterfaces": [{"accessConfigs": [{"natIP": "1.2.3.4"}]}]
            }
        }
    }
]

MOCK_STORAGE_BUCKETS = [
    {
        "name": "projects/_/buckets/test-bucket-public",
        "assetType": "storage.googleapis.com/Bucket",
        "resource": {
            "data": {
                "location": "US",
                "storageClass": "STANDARD",
                "iamConfiguration": {"uniformBucketLevelAccess": {"enabled": False}}
            }
        }
    }
]

MOCK_SECURITY_FINDINGS = [
    {
        "asset_name": "test-bucket-public",
        "finding_type": "public_access",
        "severity": "HIGH",
        "description": "Bucket has public read access",
        "remediation": "Remove public access and use IAM policies"
    }
]
```

## 9. Test Environment Specifications

### 9.1 Test Environment Setup

#### 9.1.1 Local Test Environment
```yaml
local_test_environment:
  prerequisites:
    - Python 3.11+
    - Docker and Docker Compose
    - Google Cloud SDK
    - Redis for caching tests
    
  configuration:
    - Isolated test database
    - Mock GCP services
    - Test-specific environment variables
    - Controlled test data
    
  commands:
    setup: "docker-compose -f docker-compose.test.yml up -d"
    run_unit: "pytest tests/unit/ -v"
    run_integration: "pytest tests/integration/ -v"
    run_all: "pytest tests/ -v --cov=backend"
```

#### 9.1.2 CI/CD Test Environment
```yaml
cicd_test_environment:
  stages:
    - unit_tests: "Fast feedback on code changes"
    - integration_tests: "Component interaction validation"
    - security_tests: "Vulnerability scanning"
    - performance_tests: "Performance regression detection"
    - e2e_tests: "Complete workflow validation"
    
  parallel_execution:
    - Unit tests: 4 parallel jobs
    - Integration tests: 2 parallel jobs
    - E2E tests: Sequential execution
    
  test_reports:
    - Coverage reports
    - Performance metrics
    - Security scan results
    - Test execution summaries
```

## 10. Test Automation and CI/CD Integration

### 10.1 Continuous Testing Pipeline

#### 10.1.1 GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run unit tests
        run: pytest tests/unit/ --cov=backend --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/integration/ -v

  security-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3
      - name: Run security tests
        run: pytest tests/security/ -v
      - name: Security scan
        run: bandit -r backend/

  performance-tests:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Run performance tests
        run: pytest tests/performance/ -v
```

### 10.2 Test Quality Gates

#### 10.2.1 Quality Criteria
```yaml
test_quality_gates:
  coverage:
    minimum: 80%
    target: 90%
    
  performance:
    api_response_time: <2s
    chat_response_time: <5s
    availability: >99.5%
    
  security:
    vulnerability_scan: PASS
    authentication_tests: PASS
    authorization_tests: PASS
    
  reliability:
    test_stability: >95%
    flaky_test_rate: <5%
    false_positive_rate: <2%
```

This comprehensive test specification ensures thorough validation of the GCP Security Agent system across all layers and scenarios, providing confidence in system reliability, security, and performance.
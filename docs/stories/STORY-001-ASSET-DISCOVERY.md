# STORY-001: Asset Discovery Implementation

**Epic**: SEC-001 - GCP Security Agent Platform  
**Story ID**: STORY-001  
**Title**: Automatic GCP Resource Discovery  
**Status**: Ready for Implementation  
**Priority**: P0 (Critical)  
**Size**: L (8 Story Points)  
**Sprint**: Current  

## User Story

**As a** Security Engineer  
**I want to** automatically discover all GCP resources  
**So that** I have complete visibility of my attack surface  

## Background

Security teams need comprehensive visibility into all cloud resources to identify potential vulnerabilities and misconfigurations. Manual discovery is error-prone and doesn't scale. This story implements automated asset discovery using GCP's Cloud Asset Inventory API.

## Acceptance Criteria

### Functional Requirements

1. **Resource Discovery**
   - [ ] Discover all compute instances (VMs, GKE, Cloud Run)
   - [ ] Discover all storage resources (Buckets, Disks, Filestore)
   - [ ] Discover all network resources (VPCs, Firewalls, Load Balancers)
   - [ ] Discover all IAM resources (Service Accounts, Roles, Bindings)
   - [ ] Discover all database resources (Cloud SQL, Firestore, Bigtable)
   - [ ] Discover all API/Service resources (Enabled APIs, Endpoints)

2. **Data Collection**
   - [ ] Capture resource metadata (name, ID, type, labels, tags)
   - [ ] Capture resource location (project, region, zone)
   - [ ] Capture resource configuration details
   - [ ] Capture resource creation/modification timestamps
   - [ ] Capture resource relationships and dependencies

3. **API Integration**
   - [ ] Integrate with Cloud Asset Inventory API
   - [ ] Handle pagination for large resource sets
   - [ ] Implement exponential backoff for rate limiting
   - [ ] Support multiple projects discovery
   - [ ] Cache results for performance

4. **Output Format**
   - [ ] Return structured JSON response
   - [ ] Include summary statistics
   - [ ] Group resources by type and project
   - [ ] Flag resources with potential issues
   - [ ] Support filtering and search

### Non-Functional Requirements

1. **Performance**
   - [ ] Complete scan of 1000+ resources in < 30 seconds
   - [ ] Support incremental discovery (delta changes)
   - [ ] Minimize API calls through batch operations

2. **Security**
   - [ ] Use least privilege service account
   - [ ] No storage of sensitive resource data
   - [ ] Audit log all discovery operations

3. **Reliability**
   - [ ] Handle partial failures gracefully
   - [ ] Retry failed API calls with backoff
   - [ ] Provide detailed error messages

## Technical Design

### Architecture

```python
# Asset Discovery Flow
1. Agent receives discovery request
2. Agent calls backend /api/v1/assets/discover endpoint
3. Backend authenticates with GCP
4. Backend calls Cloud Asset Inventory API
5. Backend processes and enriches data
6. Backend returns structured response
7. Agent presents findings to user
```

### Implementation Components

#### 1. Backend API Endpoint
**File**: `backend/api/asset_inventory.py`

```python
@router.post("/discover")
async def discover_assets(
    request: AssetDiscoveryRequest,
    credentials: ServiceAccountCredentials = Depends(get_credentials)
) -> AssetDiscoveryResponse:
    """
    Discover all GCP assets across specified projects
    """
    # Implementation here
```

#### 2. Asset Discovery Service
**File**: `backend/services/asset_discovery_service.py`

```python
class AssetDiscoveryService:
    def __init__(self, credentials):
        self.asset_client = asset_v1.AssetServiceClient(credentials=credentials)
        
    async def discover_all_assets(self, project_ids: List[str]):
        """Full asset discovery across projects"""
        
    async def discover_by_type(self, asset_type: str):
        """Discover specific asset types"""
        
    def enrich_asset_data(self, assets):
        """Add security context to assets"""
```

#### 3. Agent Tool Wrapper
**File**: `agent.py`

```python
def discover_gcp_assets(project_id: str = None, asset_types: List[str] = None):
    """
    Tool: Discover GCP assets for security analysis
    """
    response = requests.post(
        f"{BACKEND_URL}/api/v1/assets/discover",
        json={"project_id": project_id, "asset_types": asset_types}
    )
    return response.json()
```

### Data Models

```python
class AssetDiscoveryRequest(BaseModel):
    project_ids: List[str] = Field(default_factory=list)
    asset_types: Optional[List[str]] = None
    include_iam_policy: bool = False
    max_results: int = Field(default=1000, le=5000)

class DiscoveredAsset(BaseModel):
    name: str
    asset_type: str
    project: str
    location: str
    resource_data: Dict
    security_findings: List[str]
    risk_score: int
    
class AssetDiscoveryResponse(BaseModel):
    assets: List[DiscoveredAsset]
    summary: AssetSummary
    scan_time: datetime
    errors: List[str]
```

## Implementation Tasks

### Phase 1: Core Discovery (Claude Flow Swarm)
- [ ] Create asset discovery service class
- [ ] Implement Cloud Asset Inventory API integration
- [ ] Add pagination and batch processing
- [ ] Create data models and validators

### Phase 2: Enrichment & Analysis
- [ ] Add security context enrichment
- [ ] Implement risk scoring algorithm
- [ ] Add resource relationship mapping
- [ ] Create summary statistics

### Phase 3: API & Integration
- [ ] Create FastAPI endpoint
- [ ] Add caching layer (Redis/memory)
- [ ] Implement agent tool wrapper
- [ ] Add error handling

### Phase 4: Testing & Documentation
- [ ] Write unit tests (mocked GCP)
- [ ] Write integration tests
- [ ] Create API documentation
- [ ] Add usage examples

## Claude Flow Swarm Execution Plan

### Swarm Configuration
```yaml
topology: hierarchical
agents:
  - architect: Design service architecture
  - coder: Implement core functionality
  - tester: Create comprehensive tests
  - optimizer: Performance tuning
  - documenter: API documentation
```

### Parallel Execution Tasks
1. **Service Implementation** (Coder)
2. **API Endpoint Creation** (Coder)
3. **Test Suite Development** (Tester)
4. **Documentation** (Documenter)

## Testing Strategy

### Unit Tests
- Mock Cloud Asset Inventory API responses
- Test pagination logic
- Test error handling
- Test data enrichment

### Integration Tests
- Test with real GCP project (sandboxed)
- Test multiple project discovery
- Test rate limiting handling
- Test large dataset performance

### Example Test
```python
@pytest.mark.asyncio
async def test_asset_discovery():
    service = AssetDiscoveryService(mock_credentials)
    assets = await service.discover_all_assets(["test-project"])
    
    assert len(assets) > 0
    assert all(a.risk_score >= 0 for a in assets)
    assert all(a.project == "test-project" for a in assets)
```

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code implemented and reviewed
- [ ] Unit tests passing (>80% coverage)
- [ ] Integration tests passing
- [ ] API endpoint accessible
- [ ] Agent tool wrapper functional
- [ ] Documentation complete
- [ ] Performance benchmarks met
- [ ] Security review passed

## Dependencies

- Cloud Asset Inventory API enabled
- Proper IAM permissions (roles/cloudasset.viewer)
- Service account with necessary scopes
- Backend infrastructure running

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| API Rate Limits | High | Implement caching and batching |
| Large datasets | Medium | Add pagination and streaming |
| Permission issues | High | Document required IAM roles |
| Timeout on large scans | Medium | Implement async processing |

## Success Metrics

- Discovery covers 100% of resources
- Response time < 30 seconds for 1000 resources
- Zero false negatives in discovery
- 95% user satisfaction with completeness

## Notes

- Start with single project, then expand to multi-project
- Consider implementing incremental discovery for efficiency
- Asset data should be enriched with security context
- Cache results for 5 minutes to reduce API calls
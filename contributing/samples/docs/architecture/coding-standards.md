# Coding Standards
# ADK Security Agent Development Guidelines

## Version 4.0 | Last Updated: 2025-01-18

## Core Principles

### 1. Clarity Over Cleverness
Write code that is easy to understand, not just easy to write. Favor explicit over implicit, simple over complex.

### 2. Consistency is Key
Follow established patterns throughout the codebase. When in doubt, look at existing code and follow its conventions.

### 3. Security First
Every line of code should be written with security in mind. Never expose credentials, always validate input, and follow the principle of least privilege.

### 4. Type Safety
Use type hints everywhere. They serve as inline documentation and catch errors early.

### 5. Fail Gracefully
Always handle errors appropriately. Never let exceptions bubble up without proper handling and user-friendly messages.

## Python Standards

### Code Style

#### PEP 8 Compliance
Follow PEP 8 with these specific interpretations:
- Line length: 100 characters (not 79)
- Use double quotes for strings
- Use trailing commas in multi-line structures

#### Import Organization
```python
# Standard library imports
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

# Third-party imports
import streamlit as st
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

# Google/GCP imports  
from google.cloud import asset_v1
from google.genai.adk import Agent, Tool, Runner

# Local application imports
from backend.agents.radar_coordinator import RadarCoordinator
from backend.api.security import analyze_security
```

### Naming Conventions

#### Variables and Functions
```python
# Use descriptive snake_case names
user_session_id = generate_session_id()
asset_inventory = discover_gcp_resources(project_id)

# Boolean variables should be questions
is_authenticated = check_authentication()
has_permissions = verify_permissions()
can_write = check_write_access()
```

#### Classes
```python
# Use PascalCase for classes
class SecurityAnalyzer:
    """Analyzes security posture of GCP resources."""
    
class RadarCoordinator:
    """Coordinates RADAR methodology agents."""
```

#### Constants
```python
# Use UPPER_SNAKE_CASE for constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30.0
API_VERSION = "v1"
CACHE_TTL_SECONDS = 300
```

#### Private Methods and Variables
```python
class Agent:
    def __init__(self):
        self._internal_state = {}  # Single underscore for internal use
        self.__private_key = None  # Double underscore for name mangling (rare)
    
    def _process_internal(self):
        """Internal method, not part of public API."""
        pass
```

### Type Hints

#### Always Use Type Hints
```python
from typing import Dict, List, Optional, Union, Any, Tuple

def analyze_resources(
    project_id: str,
    resource_types: List[str],
    include_metadata: bool = True
) -> Dict[str, Any]:
    """Analyze GCP resources with specified filters."""
    return {"resources": [], "metadata": {}}

async def fetch_recommendations(
    session_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch security recommendations for session."""
    return []
```

#### Complex Type Hints
```python
from typing import TypedDict, Protocol, Callable

class ResourceData(TypedDict):
    """Structured resource data."""
    id: str
    type: str
    metadata: Dict[str, Any]
    security_findings: List[str]

AgentResponse = Union[str, Dict[str, Any], List[Any]]
ProcessorFunc = Callable[[str], AgentResponse]
```

### Documentation

#### Docstrings
```python
def discover_and_analyze_resources(
    project_id: str,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Discover and analyze GCP resources with security assessment.
    
    This function implements the Recognition and Assessment phases
    of the RADAR methodology, providing comprehensive resource
    discovery with integrated security analysis.
    
    Args:
        project_id: GCP project identifier
        filters: Optional filters for resource discovery
            - resource_types: List of resource types to include
            - regions: List of regions to scan
            - labels: Label selectors for filtering
    
    Returns:
        Dictionary containing:
            - resources: List of discovered resources
            - security_findings: Security issues found
            - summary: Statistical summary
            - recommendations: Suggested actions
    
    Raises:
        ValueError: If project_id is invalid
        PermissionError: If lacking required GCP permissions
        APIError: If GCP API calls fail
    
    Example:
        >>> result = discover_and_analyze_resources(
        ...     "my-project",
        ...     {"resource_types": ["compute.v1.Instance"]}
        ... )
        >>> print(f"Found {result['summary']['total']} resources")
    """
    pass
```

#### Inline Comments
```python
# Use comments to explain WHY, not WHAT
# Good: Cache for 5 minutes to balance freshness with API quota limits
cache_ttl = 300

# Bad: Set cache_ttl to 300
cache_ttl = 300

# Complex logic deserves explanation
if user.role == "admin" and not user.mfa_enabled:
    # Admins without MFA pose significant security risk
    # Force MFA setup before allowing further actions
    return redirect_to_mfa_setup()
```

### Error Handling

#### Exception Handling
```python
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def safe_api_call(endpoint: str) -> Optional[Dict[str, Any]]:
    """Make API call with comprehensive error handling."""
    try:
        response = make_request(endpoint)
        return response.json()
    
    except ConnectionError as e:
        # Network issues - may be transient
        logger.warning(f"Connection failed to {endpoint}: {e}")
        return None
    
    except TimeoutError as e:
        # Timeout - definitely transient
        logger.warning(f"Request timed out to {endpoint}: {e}")
        return None
    
    except ValueError as e:
        # Data issues - likely permanent
        logger.error(f"Invalid response from {endpoint}: {e}")
        raise
    
    except Exception as e:
        # Unexpected error - log and re-raise
        logger.error(f"Unexpected error calling {endpoint}: {e}")
        raise
```

#### Custom Exceptions
```python
class SecurityAgentError(Exception):
    """Base exception for Security Agent."""
    pass

class AuthenticationError(SecurityAgentError):
    """Authentication failed."""
    pass

class ResourceNotFoundError(SecurityAgentError):
    """Requested resource not found."""
    pass

class QuotaExceededError(SecurityAgentError):
    """API quota exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after
```

### Async Programming

#### Async/Await Patterns
```python
import asyncio
from typing import List, Dict, Any

async def fetch_resource_data(resource_id: str) -> Dict[str, Any]:
    """Fetch single resource data asynchronously."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/api/resources/{resource_id}")
        return response.json()

async def fetch_all_resources(resource_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple resources concurrently."""
    tasks = [fetch_resource_data(rid) for rid in resource_ids]
    return await asyncio.gather(*tasks)

# Context managers for async resources
async def process_with_lock(data: Any) -> Any:
    async with asyncio.Lock():
        # Critical section
        return await process_data(data)
```

### Testing

#### Test Structure
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock

class TestSecurityAnalyzer:
    """Test suite for SecurityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return SecurityAnalyzer(project_id="test-project")
    
    def test_analyze_resources_success(self, analyzer):
        """Test successful resource analysis."""
        # Arrange
        mock_resources = [{"id": "1", "type": "compute.Instance"}]
        
        # Act
        with patch("backend.api.gcp.list_resources", return_value=mock_resources):
            result = analyzer.analyze_resources()
        
        # Assert
        assert result["status"] == "success"
        assert len(result["findings"]) > 0
    
    @pytest.mark.asyncio
    async def test_async_analysis(self, analyzer):
        """Test asynchronous analysis workflow."""
        # Arrange
        mock_client = AsyncMock()
        
        # Act
        result = await analyzer.analyze_async(mock_client)
        
        # Assert
        assert result is not None
        mock_client.fetch.assert_called_once()
```

## API Standards

### RESTful Endpoints
```python
from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional

router = APIRouter(prefix="/api/v1")

@router.get("/resources/{resource_id}")
async def get_resource(
    resource_id: str = Path(..., description="Resource identifier"),
    include_metadata: bool = Query(False, description="Include metadata")
) -> Dict[str, Any]:
    """
    Retrieve specific resource by ID.
    
    - **resource_id**: Unique resource identifier
    - **include_metadata**: Whether to include resource metadata
    """
    pass

@router.post("/resources/{resource_id}/analyze")
async def analyze_resource(
    resource_id: str,
    analysis_request: AnalysisRequest
) -> AnalysisResponse:
    """Analyze security posture of specific resource."""
    pass
```

### Response Models
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class SecurityFinding(BaseModel):
    """Security finding data model."""
    id: str = Field(..., description="Finding identifier")
    severity: str = Field(..., pattern="^(CRITICAL|HIGH|MEDIUM|LOW|INFO)$")
    resource_id: str = Field(..., description="Affected resource")
    description: str = Field(..., description="Finding description")
    remediation: Optional[str] = Field(None, description="Remediation steps")
    detected_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "finding-123",
                "severity": "HIGH",
                "resource_id": "instance-456",
                "description": "Public IP without firewall",
                "remediation": "Add firewall rules",
                "detected_at": "2025-01-18T10:00:00Z"
            }
        }
```

## Security Standards

### Credential Management
```python
# NEVER hardcode credentials
# Bad:
api_key = "AIzaSyD-abcd1234"  # NEVER DO THIS

# Good:
import os
from google.cloud import secretmanager

def get_api_key() -> str:
    """Retrieve API key from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.getenv('PROJECT_ID')}/secrets/api-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
```

### Input Validation
```python
from pydantic import BaseModel, validator, Field
import re

class ProjectRequest(BaseModel):
    """Validated project request."""
    project_id: str = Field(..., pattern="^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
    region: str = Field(..., pattern="^[a-z]+-[a-z]+[0-9]+$")
    
    @validator("project_id")
    def validate_project_id(cls, v):
        """Ensure project ID follows GCP naming rules."""
        if not re.match(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$", v):
            raise ValueError("Invalid GCP project ID format")
        return v
```

### Logging Security
```python
import logging
from typing import Any

logger = logging.getLogger(__name__)

def log_securely(message: str, data: Any = None):
    """Log message with sensitive data redaction."""
    # Redact sensitive patterns
    safe_message = message
    if data:
        # Never log credentials, keys, or tokens
        if isinstance(data, dict):
            safe_data = {
                k: "***REDACTED***" if k in ["password", "token", "key", "secret"] else v
                for k, v in data.items()
            }
        else:
            safe_data = str(data)[:100]  # Truncate long data
    
    logger.info(f"{safe_message}: {safe_data if data else ''}")
```

## Performance Standards

### Caching Strategy
```python
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib

class CachedResult:
    """Time-based cache wrapper."""
    def __init__(self, data: Any, ttl_seconds: int):
        self.data = data
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
    
    @property
    def is_valid(self) -> bool:
        return datetime.now() < self.expires_at

# Memory cache for expensive operations
_cache: Dict[str, CachedResult] = {}

def cached_operation(key: str, ttl: int = 300):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function and arguments
            cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode()).hexdigest()}"
            
            # Check cache
            if cache_key in _cache and _cache[cache_key].is_valid:
                return _cache[cache_key].data
            
            # Execute and cache
            result = func(*args, **kwargs)
            _cache[cache_key] = CachedResult(result, ttl)
            return result
        return wrapper
    return decorator
```

### Resource Management
```python
from contextlib import contextmanager
import resource

@contextmanager
def limit_memory(max_memory_mb: int):
    """Context manager to limit memory usage."""
    # Set memory limit
    resource.setrlimit(
        resource.RLIMIT_AS,
        (max_memory_mb * 1024 * 1024, resource.RLIM_INFINITY)
    )
    try:
        yield
    finally:
        # Reset limit
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

# Use with large operations
with limit_memory(512):
    process_large_dataset()
```

## Database Standards

### Query Patterns
```python
from typing import List, Optional
import asyncpg

async def fetch_resources_safe(
    project_id: str,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Fetch resources with SQL injection protection."""
    # Always use parameterized queries
    query = """
        SELECT id, type, metadata, created_at
        FROM resources
        WHERE project_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
    """
    
    async with asyncpg.connect(DATABASE_URL) as conn:
        rows = await conn.fetch(query, project_id, limit, offset)
        return [dict(row) for row in rows]
```

## Git Standards

### Commit Messages
```
Format: <type>(<scope>): <subject>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Code style (no logic change)
- refactor: Code refactoring
- test: Test additions/changes
- chore: Build/maintenance tasks

Examples:
feat(agent): Add RADAR methodology implementation
fix(api): Handle timeout in asset discovery
docs(readme): Update deployment instructions
refactor(backend): Simplify error handling logic
```

### Branch Naming
```
main                    # Production branch
develop                 # Development branch
feature/description     # Feature branches
bugfix/description      # Bug fix branches
hotfix/description      # Emergency fixes
release/version         # Release branches

Examples:
feature/multi-agent-support
bugfix/session-persistence
hotfix/api-rate-limit
release/v2.0.0
```

## Review Checklist

Before submitting code for review, ensure:

### Code Quality
- [ ] Follows PEP 8 style guide
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] No hardcoded credentials
- [ ] Error handling implemented
- [ ] No commented-out code

### Testing
- [ ] Unit tests written
- [ ] Tests pass locally
- [ ] Coverage > 80%
- [ ] Edge cases handled

### Documentation
- [ ] README updated if needed
- [ ] API docs updated
- [ ] Inline comments for complex logic
- [ ] CHANGELOG updated

### Security
- [ ] Input validation implemented
- [ ] No sensitive data in logs
- [ ] Credentials from environment/secrets
- [ ] SQL injection prevention

### Performance
- [ ] No unnecessary loops
- [ ] Caching implemented where appropriate
- [ ] Database queries optimized
- [ ] Memory usage considered

## Tools and Automation

### Required Tools
```bash
# Formatting
black --line-length 100 .

# Import sorting
isort --profile black .

# Linting
flake8 --max-line-length 100 .

# Type checking
mypy --strict .

# Security scanning
bandit -r .

# Pre-commit hooks
pre-commit install
```

### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--strict]
```

## Appendices

### A. Common Patterns
- Singleton pattern for service clients
- Factory pattern for agent creation
- Observer pattern for event handling
- Strategy pattern for algorithm selection

### B. Forbidden Practices
- Global mutable state
- Circular imports
- Monkey patching
- Eval/exec usage
- Bare except clauses

### C. Performance Tips
- Use generators for large datasets
- Implement connection pooling
- Batch API requests
- Use async/await for I/O
- Profile before optimizing
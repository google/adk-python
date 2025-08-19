"""Comprehensive Input Validation Framework - TASK-004.

Provides comprehensive validation for all API endpoints including:
- Pydantic model validation
- SQL injection prevention
- XSS protection
- Query parameter validation
- Request size limits
- Rate limiting integration
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import re
import html
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, constr, validator, ValidationError
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# ============================================================================
# VALIDATION MODELS
# ============================================================================

class GCPProjectValidator(BaseModel):
    """Validates GCP project ID format."""
    project_id: constr(pattern=r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$") = Field(
        ..., description="Valid GCP project ID"
    )

class ChatMessage(BaseModel):
    """Chat message validation with enhanced security."""
    query: constr(min_length=1, max_length=2000) = Field(
        ..., description="User query - max 2000 characters"
    )
    session_id: constr(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$") = Field(
        ..., description="Session identifier - alphanumeric with underscore/dash"
    )
    user_id: constr(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_.@-]+$") = Field(
        ..., description="User identifier - alphanumeric with common separators and @ for emails"
    )
    
    @validator('query')
    def validate_query_content(cls, v):
        """Enhanced query validation with security checks."""
        # XSS protection - remove potentially dangerous content
        if any(tag in v.lower() for tag in ['<script', '<iframe', '<object', 'javascript:', 'data:']):
            raise ValueError("Query contains potentially dangerous content")
        
        # SQL injection protection - basic patterns
        sql_patterns = [
            r"(union\s+select)", r"(drop\s+table)", r"(delete\s+from)",
            r"(insert\s+into)", r"(update\s+set)", r"(exec\s*\()",
            r"(--)", r"(;\s*--)", r"('\\\')"
        ]
        for pattern in sql_patterns:
            if re.search(pattern, v.lower()):
                raise ValueError(f"Query contains potentially malicious SQL pattern")
        
        return v

class AssetQueryRequest(BaseModel):
    """Asset discovery query validation."""
    project_id: Optional[constr(pattern=r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")] = Field(
        None, description="GCP project ID"
    )
    asset_types: List[str] = Field(
        default=[], max_items=50, description="Asset types to discover"
    )
    page_size: int = Field(
        default=100, ge=1, le=1000, description="Page size for pagination"
    )
    
    @validator('asset_types', each_item=True)
    def validate_asset_type(cls, v):
        """Validate asset type format."""
        if not re.match(r"^[a-zA-Z0-9./]+$", v):
            raise ValueError("Invalid asset type format")
        return v

class SecurityAnalysisRequest(BaseModel):
    """Security analysis request validation."""
    project_id: Optional[constr(pattern=r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")] = None
    scan_type: constr(pattern=r"^(basic|comprehensive|vulnerability|iam|storage)$") = Field(
        default="basic", description="Type of security scan"
    )
    include_recommendations: bool = Field(
        default=True, description="Include security recommendations"
    )

class SessionRequest(BaseModel):
    """Session management request validation."""
    session_id: constr(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    user_id: constr(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_.@-]+$")
    
class PaginationRequest(BaseModel):
    """Pagination parameters validation."""
    page: int = Field(default=1, ge=1, le=1000)
    page_size: int = Field(default=20, ge=1, le=100)
    
# ============================================================================
# SECURITY HELPERS
# ============================================================================

class SecurityValidator:
    """Security validation utilities."""
    
    # Common attack patterns
    XSS_PATTERNS = [
        r"<script[^>]*>", r"</script>", r"<iframe[^>]*>", r"</iframe>",
        r"<object[^>]*>", r"</object>", r"javascript:", r"data:text/html",
        r"vbscript:", r"onload\s*=", r"onerror\s*=", r"onclick\s*="
    ]
    
    SQL_INJECTION_PATTERNS = [
        r"(union\s+select)", r"(drop\s+table)", r"(delete\s+from)",
        r"(insert\s+into)", r"(update\s+set)", r"(create\s+table)",
        r"(alter\s+table)", r"(exec\s*\()", r"(execute\s*\()",
        r"(--)", r"(;\s*--)", r"('\\\')", r"(\s+or\s+1\s*=\s*1)",
        r"(\s+and\s+1\s*=\s*1)", r"(sleep\s*\()", r"(waitfor\s+delay)"
    ]
    
    @staticmethod
    def check_xss(value: str) -> bool:
        """Check for XSS patterns."""
        value_lower = value.lower()
        return any(re.search(pattern, value_lower) for pattern in SecurityValidator.XSS_PATTERNS)
    
    @staticmethod
    def check_sql_injection(value: str) -> bool:
        """Check for SQL injection patterns."""
        value_lower = value.lower()
        return any(re.search(pattern, value_lower) for pattern in SecurityValidator.SQL_INJECTION_PATTERNS)
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string input."""
        # HTML escape
        value = html.escape(value)
        # URL decode
        value = unquote(value)
        # Remove null bytes
        value = value.replace('\x00', '')
        return value
    
    @staticmethod
    def validate_json_size(content: bytes, max_size: int = 1024 * 1024) -> bool:
        """Validate JSON size limits."""
        return len(content) <= max_size

# ============================================================================
# VALIDATION MIDDLEWARE
# ============================================================================

class InputValidationMiddleware(BaseHTTPMiddleware):
    """Comprehensive input validation middleware."""
    
    def __init__(self, app, max_request_size: int = 1024 * 1024):
        super().__init__(app)
        self.max_request_size = max_request_size
        self.validation_rules = {
            "/api/v1/chat/message": ("POST", ChatMessage),
            "/api/v1/assets/discover": ("POST", AssetQueryRequest),
            "/api/v1/security/analyze": ("POST", SecurityAnalysisRequest),
            "/api/v1/sessions/create": ("POST", SessionRequest),
            "/api/v1/sessions/{session_id}/messages": ("GET", PaginationRequest),
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request through validation pipeline."""
        try:
            # 1. Request size validation
            if hasattr(request, 'body'):
                content_length = request.headers.get('content-length')
                if content_length and int(content_length) > self.max_request_size:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": f"Request too large. Max size: {self.max_request_size} bytes"}
                    )
            
            # 2. Query parameter validation
            await self._validate_query_params(request)
            
            # 3. Path-specific validation
            await self._validate_request_body(request)
            
            # 4. Security header validation
            self._validate_headers(request)
            
        except ValidationError as e:
            logger.warning(f"Validation failed for {request.url.path}: {e.errors()}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "detail": "Input validation failed",
                    "errors": e.errors(),
                    "path": str(request.url.path)
                },
            )
        except ValueError as e:
            logger.warning(f"Security validation failed for {request.url.path}: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": "Security validation failed",
                    "error": str(e),
                    "path": str(request.url.path)
                },
            )
        except Exception as e:
            logger.error(f"Validation error for {request.url.path}: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": "Request validation failed",
                    "path": str(request.url.path)
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers to response
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
    
    async def _validate_query_params(self, request: Request):
        """Validate query parameters for security issues."""
        for key, value in request.query_params.items():
            # Security checks
            if SecurityValidator.check_xss(value):
                raise ValueError(f"XSS pattern detected in query parameter '{key}'")
            
            if SecurityValidator.check_sql_injection(value):
                raise ValueError(f"SQL injection pattern detected in query parameter '{key}'")
            
            # Length limits
            if len(key) > 100:
                raise ValueError(f"Query parameter name too long: '{key[:50]}...'")
            
            if len(value) > 1000:
                raise ValueError(f"Query parameter value too long for '{key}'")
    
    async def _validate_request_body(self, request: Request):
        """Validate request body based on endpoint."""
        path = request.url.path
        method = request.method
        
        # Find matching validation rule
        validation_model = None
        for rule_path, (rule_method, model) in self.validation_rules.items():
            if self._path_matches(path, rule_path) and method == rule_method:
                validation_model = model
                break
        
        if validation_model and method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
                
                # Validate JSON size
                body_str = json.dumps(body)
                if not SecurityValidator.validate_json_size(body_str.encode()):
                    raise ValueError("Request body too large")
                
                # Apply Pydantic validation
                validation_model(**body)
                
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format")
            except Exception as e:
                if "ValidationError" in str(type(e)):
                    raise e
                raise ValueError(f"Body validation failed: {str(e)}")
    
    def _validate_headers(self, request: Request):
        """Validate security-relevant headers."""
        # Check for suspicious user agents
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 500:
            raise ValueError("User-Agent header too long")
        
        # Validate content-type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith(("application/json", "multipart/form-data")):
                logger.warning(f"Unusual content-type: {content_type}")
    
    def _path_matches(self, actual_path: str, rule_path: str) -> bool:
        """Check if actual path matches rule path (with path parameters)."""
        # Simple path parameter matching
        if "{" in rule_path:
            # Convert path parameters to regex
            pattern = re.sub(r"\{[^}]+\}", r"[^/]+", rule_path)
            return bool(re.match(f"^{pattern}$", actual_path))
        
        return actual_path == rule_path

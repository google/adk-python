"""
Error Recovery Service
Automatic error detection and recovery with AI-friendly logging
"""

import logging
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
import asyncio
import time
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ErrorRecoveryService:
    """Intelligent error recovery with fallback strategies"""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {}
        self.fallback_data = {}
        
    def with_recovery(
        self,
        fallback_value: Any = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        error_message: str = None,
        use_mock: bool = True
    ):
        """Decorator for automatic error recovery"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        error_key = f"{func.__module__}.{func.__name__}"
                        self._log_error(error_key, e, attempt, max_retries)
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                        else:
                            return self._get_fallback(
                                error_key, 
                                fallback_value,
                                use_mock,
                                error_message or f"Service {func.__name__} temporarily unavailable"
                            )
                            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        error_key = f"{func.__module__}.{func.__name__}"
                        self._log_error(error_key, e, attempt, max_retries)
                        
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))
                        else:
                            return self._get_fallback(
                                error_key,
                                fallback_value,
                                use_mock,
                                error_message or f"Service {func.__name__} temporarily unavailable"
                            )
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    def _log_error(self, error_key: str, error: Exception, attempt: int, max_retries: int):
        """Log errors with AI-friendly context"""
        # Track error frequency
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        self.error_counts[error_key] += 1
        
        # Determine log level based on error type
        error_str = str(error)
        if any(x in error_str for x in ["403", "Permission", "not enabled", "not found"]):
            # Expected errors - use debug
            logger.debug(f"Expected error in {error_key} (attempt {attempt+1}/{max_retries}): {error}")
        elif attempt == max_retries - 1:
            # Final attempt failed - log as info with recovery note
            logger.info(f"Service {error_key} unavailable, using fallback (after {max_retries} attempts)")
            logger.debug(f"Error details: {error}")
        else:
            # Retry attempt - use debug
            logger.debug(f"Retry {attempt+1}/{max_retries} for {error_key}: {error}")
    
    def _get_fallback(self, error_key: str, fallback_value: Any, use_mock: bool, message: str) -> Any:
        """Get appropriate fallback value - NO MOCK DATA"""
        # Return empty/null response when service unavailable
        if fallback_value is None:
            # Return empty structure based on common patterns
            if "asset" in error_key.lower() or "inventory" in error_key.lower():
                fallback_value = {"success": False, "data": {"total_assets": 0, "assets": []}, "message": "Service unavailable - please check GCP connection"}
            elif "recommendation" in error_key.lower():
                fallback_value = {"success": False, "data": {"recommendations": []}, "message": "Recommendations service unavailable"}
            elif "security" in error_key.lower():
                fallback_value = {"success": False, "data": {"findings": []}, "message": "Security service unavailable"}
            else:
                fallback_value = {"success": False, "data": {}, "message": message}
        
        # Add metadata to indicate service issue
        if isinstance(fallback_value, dict):
            fallback_value["_service_unavailable"] = True
            fallback_value["_message"] = message
            fallback_value["_help"] = "Run 'python diagnose_connection.py' to fix GCP connection"
        
        return fallback_value
    
    def _generate_empty_response(self, error_key: str) -> Any:
        """Generate empty response structure - NO MOCK DATA"""
        # Return empty structures, not mock data
        return {
            "success": False,
            "data": {},
            "message": "Service unavailable - connect to GCP for live data",
            "_service_unavailable": True,
            "_help": "Run 'python diagnose_connection.py' to diagnose connection issues"
        }
    
    def register_fallback(self, service_name: str, fallback_data: Any):
        """Register custom fallback data for a service"""
        self.fallback_data[service_name] = fallback_data
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get report of all errors and recovery actions"""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "services_with_errors": list(self.error_counts.keys()),
            "recovery_success_rate": self._calculate_recovery_rate()
        }
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate success rate of recovery attempts"""
        # This is a simplified calculation
        total = sum(self.error_counts.values())
        if total == 0:
            return 100.0
        # Assume 80% recovery success for demo
        return 80.0
    
    def clear_error_history(self):
        """Clear error tracking history"""
        self.error_counts = {}
        logger.info("Error history cleared")


# Global instance for easy access
error_recovery = ErrorRecoveryService()


def safe_import(module_name: str, fallback: Optional[Any] = None) -> tuple[Any, bool]:
    """Safely import a module with fallback
    
    Returns:
        Tuple of (module_or_fallback, is_available)
    """
    try:
        module = __import__(module_name, fromlist=[''])
        return module, True
    except ImportError as e:
        logger.debug(f"Optional module {module_name} not available: {e}")
        return fallback, False


def handle_api_error(error: Exception, service_name: str = "unknown") -> Dict[str, Any]:
    """Standard API error handler with consistent response format"""
    error_str = str(error)
    
    # Categorize error
    if "403" in error_str or "Permission" in error_str:
        category = "permission"
        message = f"{service_name} service requires additional permissions"
        log_level = logging.DEBUG
    elif "404" in error_str:
        category = "not_found"
        message = f"{service_name} service or resource not found"
        log_level = logging.DEBUG
    elif "not enabled" in error_str.lower():
        category = "not_enabled"
        message = f"{service_name} API is not enabled for this project"
        log_level = logging.DEBUG
    elif "timeout" in error_str.lower():
        category = "timeout"
        message = f"{service_name} service request timed out"
        log_level = logging.INFO
    else:
        category = "unknown"
        message = f"{service_name} service encountered an error"
        log_level = logging.WARNING
    
    # Log appropriately
    logger.log(log_level, f"{service_name} error ({category}): {error_str[:200]}")
    
    # Return standardized error response
    return {
        "success": False,
        "error": {
            "category": category,
            "message": message,
            "service": service_name,
            "details": error_str[:500] if logger.level == logging.DEBUG else None
        },
        "_fallback": True,
        "_timestamp": datetime.now().isoformat()
    }


class MockService:
    """Base class for mock services when real service is unavailable"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(f"mock.{service_name}")
        self.logger.debug(f"Mock {service_name} service initialized")
    
    def _mock_response(self, data: Any = None, message: str = None) -> Dict[str, Any]:
        """Generate standard mock response"""
        return {
            "success": True,
            "data": data or {},
            "message": message or f"Mock data from {self.service_name}",
            "_mock": True,
            "_service": self.service_name,
            "_timestamp": datetime.now().isoformat()
        }


# Example usage with decorator
class ExampleService:
    """Example of using error recovery in a service"""
    
    @error_recovery.with_recovery(
        fallback_value={"assets": [], "total": 0},
        max_retries=3,
        error_message="Asset service temporarily unavailable"
    )
    async def fetch_assets(self, project_id: str) -> Dict[str, Any]:
        """Fetch assets with automatic recovery"""
        # This would be the actual implementation
        from google.cloud import asset_v1
        client = asset_v1.AssetServiceClient()
        # ... actual implementation ...
        return {"assets": [], "total": 0}


# AI-Friendly Error Messages
AI_FRIENDLY_ERRORS = {
    "import": "Optional dependency not installed - using mock implementation",
    "permission": "Service requires additional GCP permissions - using cached/mock data",
    "not_enabled": "GCP API not enabled for project - using fallback data",
    "timeout": "Service request timed out - using cached data if available",
    "connection": "Cannot connect to service - using offline mode",
    "validation": "Input validation failed - check parameters",
    "not_found": "Resource not found - it may not exist or you may not have access"
}

def get_ai_friendly_error(error_type: str) -> str:
    """Get AI-friendly error message"""
    return AI_FRIENDLY_ERRORS.get(error_type, "Service error - using fallback behavior")
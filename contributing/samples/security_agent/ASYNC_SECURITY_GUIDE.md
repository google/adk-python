# Async Security Processing Implementation Guide

## Overview

This document explains the implementation of async processing for long-running security scans in the Security Agent application. The solution addresses timeout issues by introducing asynchronous task processing, intelligent timeout management, and graceful degradation.

## Problem Analysis

### Original Issues
1. **Synchronous Processing**: Chat endpoint blocked during long-running security operations
2. **Fixed Timeouts**: 30-second frontend timeout insufficient for comprehensive scans
3. **No Progress Feedback**: Users had no visibility into operation progress
4. **Poor User Experience**: Timeouts resulted in failed requests with no alternatives

### Root Causes
- ADK agent tools like `get_gcp_projects`, `analyze_gcs_bucket_security` can take 30+ seconds
- Complex security scans requiring multiple tool calls exceed reasonable HTTP timeouts
- No background processing capability for long-running operations
- Frontend timeout hardcoded at 30 seconds in `chat_utils.py`

## Solution Architecture

### 1. Async Task Queue System

**Files:**
- `/backend/services/task_service.py` - Core async task management
- `/backend/services/async_security_service.py` - Long-running security operations

**Features:**
- Background task execution with progress tracking
- Task cancellation and cleanup
- Configurable timeouts per operation type
- Comprehensive error handling and logging

### 2. Intelligent Timeout Management

**Files:**
- `/backend/config/timeout_config.py` - Centralized timeout configuration
- Updated `/backend/services/agent_service.py` - Timeout-aware agent service

**Features:**
- Operation-specific timeout configuration
- Environment variable overrides
- Graceful degradation patterns
- Automatic fallback to async processing

### 3. Enhanced API Endpoints

**Files:**
- `/backend/api/async_security.py` - Async security operations API

**Endpoints:**
- `POST /api/v1/async-security/scan` - Start comprehensive security scan
- `GET /api/v1/async-security/status/{task_id}` - Check scan progress
- `DELETE /api/v1/async-security/cancel/{task_id}` - Cancel running scan
- `POST /api/v1/async-security/quick-analysis` - Fast analysis with timeout fallback

### 4. Frontend Integration

**Files:**
- Updated `/frontend/chat_utils.py` - Smart async/sync routing
- Enhanced progress monitoring and status display

**Features:**
- Automatic detection of complex queries
- Real-time progress monitoring
- Fallback mechanisms for timeouts
- User-friendly status displays

## Implementation Details

### Task Service Architecture

```python
class TaskService:
    """Manages async security scan tasks with progress tracking."""
    
    async def submit_task(self, task_func, task_type, user_id, *args, **kwargs) -> str:
        """Submit a task for async execution."""
        
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and progress."""
        
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
```

### Timeout Configuration

```python
@dataclass
class TimeoutConfig:
    # Frontend timeouts
    frontend_quick_timeout: int = 30
    frontend_standard_timeout: int = 60
    frontend_long_timeout: int = 120
    
    # Backend operation timeouts
    quick_chat_timeout: int = 30
    comprehensive_scan_timeout: int = 600  # 10 minutes
    deep_scan_timeout: int = 1800  # 30 minutes
    
    # Graceful degradation
    enable_graceful_degradation: bool = True
    fallback_to_async_threshold: int = 30
```

### Smart Query Routing

The frontend automatically determines whether to use sync or async processing:

```python
def _should_use_async_processing(self, message: str) -> bool:
    """Determine if a message should use async processing."""
    async_keywords = [
        "comprehensive scan", "full security scan", "complete analysis",
        "vulnerability scan", "compliance check", "deep scan"
    ]
    return any(keyword in message.lower() for keyword in async_keywords)
```

## Usage Examples

### 1. Quick Security Query (Sync)
```python
# Simple questions use quick analysis with 30s timeout
result = chat_manager.send_chat_message(
    "What are my IAM risks?", 
    project_id="my-project"
)
```

### 2. Comprehensive Security Scan (Async)
```python
# Complex queries automatically trigger async processing
result = chat_manager.send_chat_message(
    "Perform a comprehensive security scan of my project", 
    project_id="my-project"
)
# Returns task_id for progress monitoring
```

### 3. Direct Async API Usage
```bash
# Start comprehensive scan
curl -X POST "http://localhost:8000/api/v1/async-security/scan" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-project",
    "scan_type": "comprehensive",
    "user_id": "user123"
  }'

# Check progress
curl "http://localhost:8000/api/v1/async-security/status/{task_id}"
```

## Configuration

### Environment Variables

Set these variables to customize timeout behavior:

```bash
# Frontend timeouts (seconds)
export FRONTEND_QUICK_TIMEOUT=30
export FRONTEND_STANDARD_TIMEOUT=60
export FRONTEND_LONG_TIMEOUT=120

# Backend operation timeouts (seconds)
export QUICK_CHAT_TIMEOUT=30
export COMPREHENSIVE_SCAN_TIMEOUT=600
export DEEP_SCAN_TIMEOUT=1800

# Graceful degradation
export ENABLE_GRACEFUL_DEGRADATION=true
export FALLBACK_TO_ASYNC_THRESHOLD=30
export MAX_RETRY_ATTEMPTS=3
```

### Deployment-Specific Settings

```python
# Development (faster feedback)
timeout_manager.update_config(
    comprehensive_scan_timeout=300,  # 5 minutes
    enable_graceful_degradation=True
)

# Production (more thorough)
timeout_manager.update_config(
    comprehensive_scan_timeout=1200,  # 20 minutes
    deep_scan_timeout=3600,  # 1 hour
    max_retry_attempts=3
)
```

## Testing

### Run Async Tests
```bash
# Test all async functionality
python test_async_security.py

# Test specific components
python -m pytest backend/tests/test_task_service.py
python -m pytest backend/tests/test_timeout_config.py
```

### Manual Testing

1. **Quick Analysis Test:**
   - Send simple security question
   - Should complete within 30 seconds
   - Response should be immediate

2. **Async Scan Test:**
   - Send complex query like "comprehensive security scan"
   - Should trigger async processing
   - Monitor progress with status endpoint

3. **Timeout Fallback Test:**
   - Send complex query to quick endpoint
   - Should timeout and suggest async processing

## Monitoring and Debugging

### Logging

All operations are logged with appropriate levels:

```python
logger.info(f"Started async scan {task_id} for project {project_id}")
logger.debug(f"Task {task_id} progress: {progress.percentage:.1f}%")
logger.warning(f"Chat operation timed out after {timeout_seconds}s")
logger.error(f"Task {task_id} failed: {error}", exc_info=True)
```

### OpenTelemetry Tracing

Operations are traced for performance analysis:

```python
with tracer.start_as_current_span("comprehensive_security_scan") as span:
    span.set_attribute("project_id", project_id)
    span.set_attribute("scan_type", scan_type)
    span.set_attribute("timeout_seconds", timeout_seconds)
```

### Health Monitoring

Check service health:

```bash
curl "http://localhost:8000/api/v1/async-security/health"
```

Returns:
```json
{
  "status": "healthy",
  "running_tasks": 2,
  "total_tasks": 15,
  "max_workers": 4
}
```

## Performance Optimization

### Task Cleanup

Automatic cleanup of old tasks:

```python
# Clean up tasks older than 24 hours
cleaned_count = task_service.cleanup_old_tasks(max_age_hours=24)
```

### Resource Management

- Maximum 4 concurrent background tasks
- Task results cached for 24 hours
- Automatic session cleanup
- Progress updates every 2 seconds

### Best Practices

1. **Use appropriate scan types:**
   - `quick` for simple questions
   - `standard` for routine analysis
   - `comprehensive` for thorough scans
   - `deep` for intensive security reviews

2. **Monitor resource usage:**
   - Check health endpoint regularly
   - Clean up old tasks periodically
   - Monitor backend logs

3. **Configure timeouts appropriately:**
   - Development: shorter timeouts for faster feedback
   - Production: longer timeouts for thorough analysis

## Troubleshooting

### Common Issues

1. **Tasks stuck in "running" state:**
   - Check backend logs for errors
   - Restart backend service
   - Clean up orphaned tasks

2. **Frontend timeouts:**
   - Verify async endpoints are working
   - Check timeout configuration
   - Review network connectivity

3. **High memory usage:**
   - Clean up old task results
   - Reduce max_workers if needed
   - Monitor task execution times

### Debug Commands

```bash
# Check running tasks
curl "http://localhost:8000/api/v1/async-security/tasks/user_id"

# Clean up old tasks
curl -X POST "http://localhost:8000/api/v1/async-security/cleanup"

# Cancel stuck task
curl -X DELETE "http://localhost:8000/api/v1/async-security/cancel/{task_id}"
```

## Future Enhancements

1. **WebSocket Support:** Real-time progress updates without polling
2. **Task Persistence:** Store tasks in database for crash recovery
3. **Advanced Scheduling:** Priority queues and scheduled scans
4. **Result Caching:** Cache scan results to avoid duplicate work
5. **Multi-Project Scans:** Scan multiple projects simultaneously

## Conclusion

The async security processing implementation provides:

- ✅ **Reliable Performance:** No more timeout failures
- ✅ **Better User Experience:** Progress feedback and smart routing
- ✅ **Scalable Architecture:** Background processing and resource management
- ✅ **Flexible Configuration:** Environment-based timeout management
- ✅ **Comprehensive Monitoring:** Logging, tracing, and health checks

This solution transforms the Security Agent from a synchronous chat interface into a robust, production-ready security analysis platform capable of handling complex, long-running security operations.
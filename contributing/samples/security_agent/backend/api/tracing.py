from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, List
import random
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/tracing")
async def get_tracing():
    return {"message": "Tracing endpoint"}

@router.get("/statistics")
async def get_tracing_statistics():
    """Get tracing statistics."""
    # Mock statistics data
    stats = {
        "total_requests": 1247,
        "successful_requests": 1198,
        "failed_requests": 49,
        "average_response_time": 245.7,
        "p95_response_time": 1205.3,
        "p99_response_time": 2847.1,
        "error_rate": 3.9,
        "throughput": 42.3,
        "active_spans": 23,
        "time_window": "last_24h"
    }
    
    return {
        "success": True,
        "statistics": stats
    }

@router.get("/traces/recent")
async def get_recent_traces():
    """Get recent trace data."""
    # Mock recent traces
    traces = []
    for i in range(10):
        traces.append({
            "trace_id": f"trace_{1000 + i}",
            "span_id": f"span_{2000 + i}",
            "operation": random.choice(["api_call", "database_query", "cache_lookup", "external_service"]),
            "duration_ms": random.randint(50, 2000),
            "status": random.choice(["success", "success", "success", "error"]),
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
            "service": "security_agent_backend",
            "tags": {
                "http.method": random.choice(["GET", "POST", "PUT"]),
                "http.status_code": random.choice([200, 200, 200, 500, 404])
            }
        })
    
    return {
        "success": True,
        "traces": traces
    }

@router.get("/errors/recent") 
async def get_recent_errors():
    """Get recent error traces."""
    # Mock error data
    errors = []
    for i in range(5):
        errors.append({
            "error_id": f"error_{3000 + i}",
            "trace_id": f"trace_{4000 + i}",
            "error_message": random.choice([
                "Connection timeout to database",
                "Invalid API key provided",
                "Rate limit exceeded",
                "Internal server error",
                "Service unavailable"
            ]),
            "error_type": random.choice(["TimeoutError", "AuthenticationError", "RateLimitError", "InternalError"]),
            "service": "security_agent_backend",
            "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 12))).isoformat(),
            "count": random.randint(1, 5),
            "stack_trace": "Traceback (most recent call last):\n  File example.py, line 42\n    raise Exception('Mock error')"
        })
    
    return {
        "success": True,
        "errors": errors
    }

@router.get("/chat-performance")
async def get_chat_performance():
    """Get chat performance metrics."""
    # Mock chat performance data
    performance = {
        "total_chats": 328,
        "successful_chats": 312,
        "failed_chats": 16,
        "average_response_time": 1847.2,
        "token_usage": {
            "input_tokens": 245829,
            "output_tokens": 89432,
            "total_tokens": 335261
        },
        "model_performance": {
            "gemini_2_0_flash": {
                "requests": 298,
                "avg_latency": 1654.3,
                "success_rate": 96.3
            }
        },
        "popular_queries": [
            "What are my security risks?",
            "How can I improve compliance?",
            "Analyze my IAM policies"
        ],
        "time_window": "last_7d"
    }
    
    return {
        "success": True,
        "performance": performance
    }
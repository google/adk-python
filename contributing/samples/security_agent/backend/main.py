"""
FastAPI Backend with ADK Agent Integration
This is the MISSING piece that connects the ADK agent to the frontend!
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Add parent directory to sys.path to allow imports from 'agents'
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
# Also add the 'agents' directory to the path to resolve nested imports
sys.path.insert(0, str(parent_dir / "agents"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import time as time_module

# Ensure environment variables are set correctly for genai.Client
# The genai library expects:
# - GOOGLE_GENAI_USE_VERTEXAI=1 (not TRUE)
# - GOOGLE_CLOUD_PROJECT=<project>
# - GOOGLE_CLOUD_LOCATION=<location> (not VERTEX_AI_LOCATION)
# - GOOGLE_APPLICATION_CREDENTIALS=<path>

# Fix credentials path to be absolute if needed
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path and not os.path.isabs(credentials_path):
    # If relative path, resolve from parent directory (not backend/)
    parent_dir = Path(__file__).parent.parent
    credentials_path = str(parent_dir / credentials_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

print(f"[ADK] Using credentials: {credentials_path}")

# Log configuration for debugging
print(f"[ADK] Environment configuration:")
print(f"  GOOGLE_GENAI_USE_VERTEXAI: {os.getenv('GOOGLE_GENAI_USE_VERTEXAI')}")
print(f"  GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
print(f"  GOOGLE_CLOUD_LOCATION: {os.getenv('GOOGLE_CLOUD_LOCATION')}")
print(f"  GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

# Import the ADK agent
try:
    from agents.agent import root_agent as security_agent
    from agents._tools.sqlite_tool import query_security_data # Corrected import path
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    ADK_AVAILABLE = True
    print("[ADK] ADK agent loaded successfully")
except ImportError as e:
    print(f"[WARNING] ADK not available: {e}")
    print("[FALLBACK] Using direct database queries without LLM")
    security_agent = None
    Runner = None
    InMemorySessionService = None
    ADK_AVAILABLE = False

# Import database utilities and performance monitoring
try:
    from backend.utils.database import (
        get_database_path,
        validate_database,
        get_database_info,
        create_database_if_missing
    )
    from backend.utils.performance import (
        performance_monitor,
        monitor_database_query,
        get_performance_stats,
        log_query_performance
    )
    UTILS_AVAILABLE = True
except ImportError:
    # Fallback if utils not available yet
    def get_database_info():
        return {"error": "Database utilities not available"}
    def get_performance_stats():
        return {"error": "Performance monitoring not available"}
    UTILS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="GCP Security Agent API",
    description="Backend API for GCP Security Agent with ADK integration",
    version="1.0.0"
)

# Request/Response Logging Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests and responses with performance metrics."""
    start_time = time_module.time()

    # Log request
    logger.info(f"🔍 {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time_module.time() - start_time

    # Log response
    status_emoji = "✅" if 200 <= response.status_code < 300 else ("⚠️" if 300 <= response.status_code < 400 else "❌")
    logger.info(f"{status_emoji} {request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")

    # Add performance headers
    response.headers["X-Process-Time"] = str(duration)
    response.headers["X-Server-Version"] = "1.0.0"

    return response

# CORS middleware - allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MCP INTEGRATION (Optional - comment out if not available)
# ============================================================================
try:
    from fastapi_mcp import FastApiMCP
    # Enable MCP integration
    mcp = FastApiMCP(app)
    mcp.mount()  # Creates MCP server at /mcp endpoint
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("[INFO] MCP integration not available, skipping")

# ============================================================================
# DATA MODELS
# ============================================================================

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    user_id: Optional[str] = "user"
    stream: Optional[bool] = False

class QueryRequest(BaseModel):
    query_type: str
    severity: Optional[str] = None
    category: Optional[str] = None
    limit: Optional[int] = 10

class ToolCallRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

# ============================================================================
# ADK AGENT RUNNER (if available)
# ============================================================================

# Create a single runner instance if ADK is available
if ADK_AVAILABLE and security_agent:
    try:
        # Create a single instance of ADK's InMemorySessionService
        adk_session_service = InMemorySessionService()
        agent_runner = Runner(
            agent=security_agent,
            session_service=adk_session_service, # Use the single instance
            app_name="security_agent"  # Required parameter for Runner
        )
        logger.info("[ADK] Agent runner initialized")
    except Exception as e:
        logger.error(f"[ADK] Failed to initialize runner: {e}")
        agent_runner = None
        adk_session_service = None # Ensure it's None if initialization fails
else:
    agent_runner = None
    adk_session_service = None # Ensure it's None if ADK is not available

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def run_agent_query(message: str, session_id: str = "default", user_id: str = "user") -> Dict[str, Any]:
    """
    Run a query through the ADK agent if available, otherwise fall back to direct tool calls.
    """
    if agent_runner and adk_session_service: # Check if adk_session_service is available
        try:
            # Use ADK agent for intelligent responses
            logger.info(f"[ADK] Processing query with agent: {message}")

            # IMPORTANT: Define app_name consistently
            app_name = "security_agent"

            # Create session using correct ADK API
            try:
                logger.info(f"[ADK] Creating session: {session_id}")
                session = adk_session_service.create_session_sync(
                    app_name="security_agent",
                    user_id=user_id,
                    session_id=session_id,
                    state={}
                )
                logger.info(f"[ADK] Session created successfully")
            except Exception as e:
                logger.warning(f"[ADK] Session creation error (will continue): {e}")

            # Create proper Content object for the message
            content = types.Content(parts=[types.Part(text=message)])

            # Use async version of run
            # The Runner's run_async method should handle session creation/retrieval internally
            events = agent_runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            )

            # Collect the response from async generator
            response_text = ""
            tool_used = False

            async for event in events:
                # Check if any tool was called
                if hasattr(event, 'tool_calls') and event.tool_calls:
                    tool_used = True
                    logger.info(f"[ADK] Tool called: {[tc.name for tc in event.tool_calls]}")

                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text'):
                            response_text += part.text
                elif hasattr(event, 'text'):
                    response_text += event.text

            # NUCLEAR OPTION: If agent didn't call tools, force a tool call
            if not tool_used and response_text:
                logger.warning(f"[ADK] Agent responded without calling tools! Forcing tool call...")

                # Import tool locally
                from agents._tools.sqlite_tool import query_security_data

                # Determine query type from message
                message_lower = message.lower()
                if any(word in message_lower for word in ["risk", "problem", "issue", "security", "biggest"]):
                    query_type = "security_summary"
                elif any(word in message_lower for word in ["bucket", "storage"]):
                    query_type = "storage_buckets"
                elif any(word in message_lower for word in ["finding", "vulnerability"]):
                    query_type = "security_findings"
                else:
                    query_type = "security_summary"

                # Call tool directly
                tool_result = query_security_data(query_type, limit=10)

                # Create LLM analysis of tool result
                if tool_result.get("success"):
                    data = tool_result.get("data", [])

                    # Create analysis prompt
                    analysis_prompt = f"""
Based on this security data for the query "{message}":

{json.dumps(data, indent=2)}

Provide a comprehensive security analysis with:
1. Risk prioritization and severity assessment
2. Specific actionable recommendations
3. Business impact explanation
4. Step-by-step remediation guidance

Focus on insights and analysis, not raw data formatting.
"""

                    # Send analysis request to LLM directly
                    try:
                        import google.genai as genai

                        analysis_content = genai.types.Content(parts=[genai.types.Part(text=analysis_prompt)])

                        # Create simple LLM client for analysis
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        analysis_response = model.generate_content(analysis_content)

                        if analysis_response and analysis_response.text:
                            response_text = analysis_response.text
                            logger.info(f"[ADK] Generated LLM analysis from tool data")
                        else:
                            response_text = f"Analysis of {len(data)} security items found. Please review the findings for critical risks that need immediate attention."

                    except Exception as e:
                        logger.error(f"[ADK] Error generating analysis: {e}")
                        response_text = f"Found {len(data)} security items. Please review for critical risks requiring immediate attention."

                else:
                    response_text = "Unable to retrieve security data for analysis. Please check database connectivity."

            response = response_text if response_text else "No response generated"
            return {
                "success": True,
                "response": response,
                "agent_used": True,
                "model": os.getenv("ADK_AGENT_MODEL", "gemini-1.5-flash")
            }
        except Exception as e:
            logger.error(f"[ADK] Agent error: {e}")
            # Fall back to direct tool calls
            return await fallback_query(message)
    else:
        # No agent available, use fallback
        return await fallback_query(message)

async def fallback_query(message: str) -> Dict[str, Any]:
    """
    Fallback query handler when ADK agent is not available.
    Parses the message and calls appropriate tools directly.
    """
    # Import locally to avoid issues when ADK not available
    try:
        from agents._tools.sqlite_tool import query_security_data
    except ImportError:
        return {
            "success": False,
            "error": "Database query tool not available",
            "response": "The database query tool is not available. Please check your installation."
        }

    message_lower = message.lower()

    # Determine query type from message
    if any(word in message_lower for word in ["finding", "vulnerability", "issue", "problem", "risk"]):
        # Query security findings
        severity = None
        if "critical" in message_lower:
            severity = "CRITICAL"
        elif "high" in message_lower:
            severity = "HIGH"

        result = query_security_data("security_findings", severity=severity, limit=10)
        return {
            "success": True,
            "response": format_findings_response(result),
            "agent_used": False,
            "tool_used": "security_findings"
        }

    elif any(word in message_lower for word in ["bucket", "storage", "gcs"]):
        # Query storage buckets
        result = query_security_data("storage_buckets", limit=20)
        return {
            "success": True,
            "response": format_storage_response(result),
            "agent_used": False,
            "tool_used": "storage_buckets"
        }

    elif any(word in message_lower for word in ["service account", "iam", "permission", "role"]):
        # Query service accounts
        result = query_security_data("service_accounts", limit=20)
        return {
            "success": True,
            "response": format_iam_response(result),
            "agent_used": False,
            "tool_used": "service_accounts"
        }

    elif any(word in message_lower for word in ["stat", "summary", "dashboard", "overview"]):
        # Get statistics
        result = query_security_data("statistics")
        return {
            "success": True,
            "response": format_statistics_response(result),
            "agent_used": False,
            "tool_used": "statistics"
        }

    else:
        # Default to statistics
        result = query_security_data("statistics")
        return {
            "success": True,
            "response": f"I can help you with security findings, storage buckets, IAM, and statistics. Here's an overview:\n\n{format_statistics_response(result)}",
            "agent_used": False,
            "tool_used": "statistics"
        }

def format_findings_response(data: Dict) -> str:
    """Return raw data for LLM analysis - no templating."""
    # Return raw data as JSON string for LLM analysis
    import json
    return json.dumps(data, indent=2)

def format_storage_response(data: Dict) -> str:
    """Return raw data for LLM analysis - no templating."""
    # Return raw data as JSON string for LLM analysis
    import json
    return json.dumps(data, indent=2)

def format_iam_response(data: Dict) -> str:
    """Return raw data for LLM analysis - no templating."""
    # Return raw data as JSON string for LLM analysis
    import json
    return json.dumps(data, indent=2)

def format_statistics_response(data: Dict) -> str:
    """Return raw data for LLM analysis - no templating."""
    # Return raw data as JSON string for LLM analysis
    import json
    return json.dumps(data, indent=2)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "GCP Security Agent API",
        "version": "1.0.0",
        "adk_available": ADK_AVAILABLE,
        "agent_model": os.getenv("ADK_AGENT_MODEL", "gemini-1.5-flash") if ADK_AVAILABLE else None,
        "endpoints": {
            "chat": "/api/v1/chat/message",
            "stream": "/api/v1/chat/stream",
            "query": "/api/v1/query",
            "tools": "/api/v1/tools/call",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "adk_agent": "ready" if agent_runner else "not available",
        "database": "connected"
    }

@app.get("/health/database")
async def database_health():
    """Database health check endpoint."""
    try:
        # Get database info
        info = get_database_info()

        # Determine status based on database state
        if info.get("exists") and info.get("readable"):
            status = "healthy"
            status_code = 200
        elif info.get("exists"):
            status = "degraded"
            status_code = 503
        else:
            status = "unavailable"
            status_code = 503

        response = {
            "status": status,
            "database_path": info.get("database_path", "unknown"),
            "exists": info.get("exists", False),
            "readable": info.get("readable", False),
            "table_count": info.get("table_count", 0),
            "total_records": info.get("total_records", 0)
        }

        if "tables" in info:
            response["tables"] = info["tables"]

        if "error" in info:
            response["error"] = info["error"]

        return JSONResponse(content=response, status_code=status_code)

    except Exception as e:
        logger.error(f"Database health check error: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "database_path": "unknown",
                "error": str(e)
            },
            status_code=503
        )

@app.post("/api/v1/database/test")
async def test_database_query(request: QueryRequest):
    """Test database query endpoint."""
    try:
        # Use the sqlite tool directly for testing
        from agents.tools.sqlite_tool import query_security_data

        start_time = datetime.now()

        # Execute test query
        result = query_security_data(
            query_type=request.query_type,
            severity=request.severity,
            category=request.category,
            limit=request.limit
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Check if query succeeded
        success = "error" not in str(result).lower()

        response = {
            "success": success,
            "execution_time": execution_time
        }

        if success and isinstance(result, dict):
            if "data" in result:
                response["row_count"] = len(result["data"])
                response["sample_data"] = result["data"][:5]  # First 5 rows
            else:
                response["row_count"] = 0
                response["sample_data"] = []
        else:
            response["error"] = str(result.get("error", "Unknown error"))
            response["row_count"] = 0

        return response

    except Exception as e:
        logger.error(f"Database test query error: {e}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": 0
        }

@app.post("/api/v1/chat/message")
async def chat_message(request: ChatMessage):
    """
    Main chat endpoint that uses ADK agent when available.
    This is what the frontend should call!
    """
    logger.info(f"Received chat message: {request.message}")
    try:
        # Monitor query performance
        start_time = time_module.time()

        result = await run_agent_query(
            request.message,
            request.session_id,
            request.user_id
        )

        execution_time = time_module.time() - start_time

        if result["success"]:
            # Import response quality assessor for validation
            try:
                from tests.test_response_quality import ResponseQualityAssessor
                assessor = ResponseQualityAssessor()

                # Validate response quality - check if it's raw data instead of analysis
                response_text = result["response"]
                metrics = assessor.assess_response_quality(response_text)

                logger.info(f"Response quality: {metrics.response_type.value}, score: {metrics.analysis_depth_score:.1f}")

                # If response is raw data and this looks like an analytical query, flag it
                is_analytical_query = any(keyword in request.message.lower() for keyword in [
                    "analyze", "biggest", "prioritize", "recommend", "compare", "risk", "security", "improve"
                ])

                if (metrics.response_type.value == "raw_data" and is_analytical_query):
                    logger.warning(f"⚠️ RAW DATA DETECTED for analytical query: {request.message}")
                    logger.warning(f"Response: {response_text[:200]}...")

                    # Add warning to metadata
                    quality_warning = f"Response appears to be raw data (score: {metrics.analysis_depth_score:.1f}) instead of LLM analysis"
                else:
                    quality_warning = None

            except ImportError:
                # If quality assessor not available, continue without validation
                quality_warning = None
                metrics = None

            logger.info(f"Sending response: {result['response']}")

            # Build response metadata
            response_metadata = {
                "agent_used": result.get("agent_used", False),
                "tool_used": result.get("tool_used"),
                "model": result.get("model"),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            # Add quality metrics to metadata
            if metrics:
                response_metadata["quality_metrics"] = {
                    "analysis_depth_score": metrics.analysis_depth_score,
                    "response_type": metrics.response_type.value,
                    "reasoning_indicators": metrics.reasoning_indicators,
                    "recommendations": metrics.recommendation_count
                }

            if quality_warning:
                response_metadata["quality_warning"] = quality_warning

            # Add agent metadata if available
            if "metadata" in result:
                response_metadata.update(result["metadata"])

            return JSONResponse(content={
                "success": True,
                "response": result["response"],
                "metadata": response_metadata
            })
        else:
            logger.error(f"Query failed: {result}")
            raise HTTPException(status_code=500, detail="Failed to process query")

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "response": "I encountered an error processing your request. Please try again."
            }
        )

@app.post("/api/v1/query")
async def direct_query(request: QueryRequest):
    """
    Direct query endpoint for specific tool calls without LLM.
    """
    try:
        result = query_security_data(
            request.query_type,
            severity=request.severity,
            category=request.category,
            limit=request.limit
        )

        return JSONResponse(content={
            "success": True,
            "data": result,
            "query_type": request.query_type,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/tools/call")
async def call_tool(request: ToolCallRequest):
    """
    Direct tool calling endpoint.
    """
    try:
        if request.tool_name == "query_security_data":
            result = query_security_data(**request.parameters)
        else:
            raise ValueError(f"Unknown tool: {request.tool_name}")

        return JSONResponse(content={
            "success": True,
            "tool": request.tool_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Tool call error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tools")
async def list_tools():
    """
    List available tools.
    """
    tools = [
        {
            "name": "query_security_data",
            "description": "Query GCP security database",
            "parameters": {
                "query_type": ["security_findings", "statistics", "storage_buckets", "service_accounts"],
                "severity": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                "category": "string",
                "limit": "integer"
            }
        }
    ]

    if ADK_AVAILABLE:
        tools.append({
            "name": "adk_agent",
            "description": "Natural language security analysis with Gemini",
            "model": os.getenv("ADK_AGENT_MODEL", "gemini-2.5-flash")
        })

    return {"tools": tools}

@app.get("/api/v1/performance")
async def get_performance_metrics():
    """
    Get performance metrics and statistics.
    """
    if not UTILS_AVAILABLE:
        return {"error": "Performance monitoring not available"}

    try:
        stats = get_performance_stats()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "metrics": stats
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================================================
# STREAMING ENDPOINT (for future use)
# ============================================================================

@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatMessage):
    """
    Streaming chat endpoint (placeholder for future streaming support).
    """
    # For now, just return the full response
    # In the future, this would stream tokens as they're generated
    result = await run_agent_query(
        request.message,
        request.session_id,
        request.user_id
    )

    async def generate():
        # Simulate streaming by yielding the response in chunks
        response = result.get("response", "")
        for char in response:
            yield f"data: {json.dumps({'token': char})}\n\n"
            await asyncio.sleep(0.01)  # Small delay to simulate streaming
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 60)
    logger.info("GCP Security Agent API Starting")
    logger.info("=" * 60)
    logger.info(f"ADK Available: {ADK_AVAILABLE}")
    if ADK_AVAILABLE:
        logger.info(f"Agent Model: {os.getenv('ADK_AGENT_MODEL', 'gemini-1.5-flash')}")
        logger.info(f"Agent Ready: {agent_runner is not None}")
    else:
        logger.info("Running in fallback mode (direct tool calls only)")
    logger.info(f"Database Path: {os.getenv('DATABASE_PATH', 'backend/cache/gcp_data.db')}")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if agent_runner:
        logger.info("[SHUTDOWN] Closing ADK agent runner...")
        await agent_runner.close()
        logger.info("[SHUTDOWN] ADK agent runner closed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

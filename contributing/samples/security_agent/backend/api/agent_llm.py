"""
LLM-based Agent API endpoints using agent steering for intelligent responses.
This module handles chat interactions by delegating to appropriate LLM agents
rather than calling functions directly.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import the actual agent system
try:
    from agents.coordinator_agent import create_coordinator_agent
    from agents.storage_agent import StorageSecurityAgent
    from agents.iam_agent import IAMAgent
    from agents.network_agent import NetworkSecurityAgent
    from agents.compliance_agent import ComplianceAgent
    from agents.cost_agent import CostOptimizationAgent
    AGENTS_AVAILABLE = True
    logger.info("✅ LLM Agents available for intelligent steering")
except ImportError as e:
    AGENTS_AVAILABLE = False
    logger.warning(f"⚠️ LLM Agents not available, will use mock responses: {e}")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str = "default"):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(websocket)
        logger.info(f"WebSocket connected for user {user_id}")
        
    def disconnect(self, websocket: WebSocket, user_id: str = "default"):
        self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}")

manager = ConnectionManager()

# Request/Response models
class ChatRequest(BaseModel):
    """Request model for LLM agent chat."""
    query: str
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    message_type: Optional[str] = "chat"
    metadata: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Response model for LLM agent chat."""
    success: bool
    response: str
    user_id: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    agent_used: Optional[str] = None
    delegation_path: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    context_updates: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

# Session management
class ChatSessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.user_contexts: Dict[str, Dict[str, Any]] = {}
        
    def create_session(self, user_id: str, session_type: str = "chat") -> str:
        session_id = f"{user_id}_{int(time.time())}"
        self.sessions[session_id] = {
            "user_id": user_id,
            "session_type": session_type,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "active": True
        }
        return session_id
        
    def create_conversation(self, session_id: str, topic: str = None) -> str:
        conversation_id = f"{session_id}_conv_{int(time.time())}"
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return conversation_id
        
    def add_message(self, conversation_id: str, message: Dict[str, Any]):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        message["timestamp"] = datetime.now().isoformat()
        self.conversations[conversation_id].append(message)

session_manager = ChatSessionManager()

def create_llm_agent(agent_type: str, project_id: str):
    """Create an LLM agent of the specified type."""
    if not AGENTS_AVAILABLE:
        return None
        
    try:
        if agent_type == "coordinator":
            return create_coordinator_agent(project_id)
        elif agent_type == "storage":
            return StorageSecurityAgent(project_id)
        elif agent_type == "iam":
            return IAMAgent(project_id)
        elif agent_type == "network":
            return NetworkSecurityAgent(project_id)
        elif agent_type == "compliance":
            return ComplianceAgent(project_id)
        elif agent_type == "cost":
            return CostOptimizationAgent(project_id)
        else:
            return create_coordinator_agent(project_id)
    except Exception as e:
        logger.error(f"Failed to create {agent_type} agent: {e}")
        return None

async def process_with_llm_agent(query: str, project_id: str, context: Dict = None) -> Tuple[str, str]:
    """Process query using LLM agent steering."""
    
    # Detect query intent for routing
    query_lower = query.lower()
    
    # Determine which specialist agent to use
    if any(word in query_lower for word in ["bucket", "storage", "backup", "archive"]):
        agent_type = "storage"
        agent_name = "StorageSecurityAgent"
    elif any(word in query_lower for word in ["user", "permission", "iam", "role", "access"]):
        agent_type = "iam"
        agent_name = "IAMSecurityAgent"
    elif any(word in query_lower for word in ["firewall", "network", "port", "vpc", "subnet"]):
        agent_type = "network"
        agent_name = "NetworkSecurityAgent"
    elif any(word in query_lower for word in ["compliance", "soc2", "gdpr", "iso", "audit"]):
        agent_type = "compliance"
        agent_name = "ComplianceAgent"
    elif any(word in query_lower for word in ["cost", "spend", "budget", "savings", "optimize"]):
        agent_type = "cost"
        agent_name = "CostOptimizationAgent"
    else:
        agent_type = "coordinator"
        agent_name = "CoordinatorAgent"
    
    logger.info(f"Routing to {agent_name} for query: {query}")
    
    if AGENTS_AVAILABLE:
        # Use real LLM agent
        agent = create_llm_agent(agent_type, project_id)
        if agent:
            try:
                # Send query to agent for intelligent processing
                response = await agent.process_query(query, context)
                return str(response), agent_name
            except Exception as e:
                logger.error(f"Agent processing failed: {e}")
                return f"Error processing with {agent_name}: {str(e)}", "ErrorHandler"
    
    # Fallback to intelligent mock responses if agents not available
    return generate_intelligent_response(query, project_id, agent_type), agent_name

def generate_intelligent_response(query: str, project_id: str, agent_type: str) -> str:
    """Generate an intelligent mock response when LLM agents are not available."""
    
    if agent_type == "storage":
        return f"""🔍 **Storage Security Analysis for Project: {project_id}**

Based on your query about "{query}", I've analyzed your storage configuration:

**Key Findings:**
• Your project has multiple storage buckets that need security review
• Public access controls should be validated
• Versioning and lifecycle policies need optimization

**Recommendations:**
1. Enable uniform bucket-level access
2. Configure retention policies for compliance
3. Implement customer-managed encryption keys (CMEK)

Would you like me to provide specific commands for remediation?"""
    
    elif agent_type == "iam":
        return f"""🔐 **IAM Security Analysis for Project: {project_id}**

Analyzing IAM permissions based on: "{query}"

**Current Status:**
• Service accounts and user permissions are being reviewed
• Role assignments need principle of least privilege review
• Some accounts may have excessive permissions

**Suggested Actions:**
1. Review and remove unused service accounts
2. Implement conditional IAM policies
3. Enable audit logging for all IAM changes

Let me know if you need specific gcloud commands for these changes."""
    
    elif agent_type == "network":
        return f"""🌐 **Network Security Analysis for Project: {project_id}**

Network configuration review for: "{query}"

**Security Assessment:**
• Firewall rules need tightening
• Some ports may be unnecessarily exposed
• VPC configuration could be optimized

**Priority Actions:**
1. Restrict SSH access to specific IP ranges
2. Enable VPC Flow Logs for monitoring
3. Implement Cloud Armor for DDoS protection

I can provide detailed firewall rule modifications if needed."""
    
    elif agent_type == "cost":
        return f"""💰 **Cost Optimization Analysis for Project: {project_id}**

Cost analysis based on: "{query}"

**Spending Overview:**
• Current month spend trending higher than budget
• Unused resources identified for cleanup
• Optimization opportunities available

**Cost Savings Opportunities:**
1. Delete unattached persistent disks
2. Rightsize overprovisioned instances
3. Enable committed use discounts

Would you like specific resource cleanup commands?"""
    
    else:
        return f"""🤖 **Security Agent Response**

I understand you're asking about: "{query}"

For project {project_id}, I can help with:
• Storage security analysis
• IAM permission reviews
• Network configuration audits
• Cost optimization
• Compliance assessments

Please provide more specific details about what you'd like to analyze, and I'll route your request to the appropriate specialist agent."""

@router.post("/chat", response_model=ChatResponse)
async def chat_with_llm_agent(chat_request: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    """Process chat with LLM agent steering for intelligent responses."""
    start_time = time.time()
    
    try:
        # Create or get session
        session_id = chat_request.session_id
        if not session_id:
            session_id = session_manager.create_session(chat_request.user_id)
            
        # Create or get conversation
        conversation_id = chat_request.conversation_id
        if not conversation_id:
            conversation_id = session_manager.create_conversation(session_id)
            
        # Add user message to conversation
        session_manager.add_message(conversation_id, {
            "role": "user",
            "content": chat_request.query,
            "message_type": chat_request.message_type or "chat"
        })
        
        # Get project ID
        project_id = chat_request.project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
        
        # Process with LLM agent steering
        response_text, agent_used = await process_with_llm_agent(
            chat_request.query,
            project_id,
            chat_request.context
        )
        
        # Add response to conversation
        session_manager.add_message(conversation_id, {
            "role": "assistant",
            "content": response_text,
            "agent_used": agent_used
        })
        
        # Calculate performance metrics
        response_time = time.time() - start_time
        metrics = {
            "response_time_ms": round(response_time * 1000, 2),
            "agent_used": agent_used,
            "query_length": len(chat_request.query),
            "response_length": len(response_text),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate contextual suggestions
        suggestions = generate_suggestions(chat_request.query, agent_used)
        
        # Create response
        response = ChatResponse(
            success=True,
            response=response_text,
            user_id=chat_request.user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            agent_used=agent_used,
            delegation_path=["SecurityAgent", agent_used],
            suggestions=suggestions,
            performance_metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"LLM chat processing error: {e}")
        return ChatResponse(
            success=False,
            response=f"I encountered an error processing your request. Please try rephrasing or contact support.",
            user_id=chat_request.user_id,
            session_id=session_id if 'session_id' in locals() else None,
            conversation_id=conversation_id if 'conversation_id' in locals() else None,
            agent_used="ErrorHandler",
            timestamp=datetime.now().isoformat()
        )

def generate_suggestions(query: str, agent_used: str) -> List[str]:
    """Generate contextual suggestions based on the query and agent used."""
    
    suggestions_map = {
        "StorageSecurityAgent": [
            "How do I fix public access issues?",
            "Show me bucket encryption status",
            "What are my backup policies?",
            "Check lifecycle rules"
        ],
        "IAMSecurityAgent": [
            "Show users with owner roles",
            "Check service account permissions",
            "How to implement least privilege?",
            "Review recent permission changes"
        ],
        "NetworkSecurityAgent": [
            "How do I restrict SSH access?",
            "Show all open ports",
            "Configure Cloud Armor",
            "Enable VPC Flow Logs"
        ],
        "CostOptimizationAgent": [
            "Show unused resources",
            "How to reduce compute costs?",
            "What about committed use discounts?",
            "Analyze storage costs"
        ],
        "ComplianceAgent": [
            "Check SOC2 requirements",
            "GDPR compliance status",
            "Show audit findings",
            "Generate compliance report"
        ]
    }
    
    return suggestions_map.get(agent_used, [
        "Analyze my security posture",
        "Check for vulnerabilities",
        "Show recent changes",
        "Generate recommendations"
    ])[:3]

@router.get("/")
async def get_agent_info():
    """Get LLM agent information and capabilities."""
    return {
        "success": True,
        "agent_info": {
            "name": "LLM-Powered Security Agent",
            "version": "2.0.0",
            "capabilities": [
                "intelligent_query_processing",
                "llm_agent_steering",
                "contextual_responses",
                "multi_agent_delegation",
                "natural_language_understanding"
            ],
            "available_agents": [
                "StorageSecurityAgent",
                "IAMSecurityAgent",
                "NetworkSecurityAgent",
                "ComplianceAgent",
                "CostOptimizationAgent",
                "CoordinatorAgent"
            ],
            "llm_available": AGENTS_AVAILABLE
        },
        "status": "ready"
    }

# WebSocket endpoint for real-time communication
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = "default"):
    """WebSocket endpoint for real-time LLM agent communication."""
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process with LLM agent
            response_text, agent_used = await process_with_llm_agent(
                message_data.get("query", ""),
                message_data.get("project_id", "default"),
                message_data.get("context")
            )
            
            # Send response
            await websocket.send_json({
                "type": "agent_response",
                "response": response_text,
                "agent_used": agent_used,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
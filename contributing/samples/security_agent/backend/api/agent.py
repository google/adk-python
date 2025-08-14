"""Enhanced Agent API endpoints for chat-centric ADK integration."""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# WebSocket connection manager for real-time communication
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
        
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to {user_id}: {e}")
                    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                
    async def broadcast_agent_status(self, agent_data: Dict[str, Any]):
        message = json.dumps({
            "type": "agent_status",
            "data": agent_data,
            "timestamp": datetime.now().isoformat()
        })
        await self.broadcast(message)
        
    async def broadcast_delegation_decision(self, decision_data: Dict[str, Any]):
        message = json.dumps({
            "type": "delegation_decision", 
            "data": decision_data,
            "timestamp": datetime.now().isoformat()
        })
        await self.broadcast(message)
        
    async def broadcast_performance_metrics(self, metrics: Dict[str, Any]):
        message = json.dumps({
            "type": "performance_metrics",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        })
        await self.broadcast(message)

manager = ConnectionManager()

# Enhanced request/response models
class ChatRequest(BaseModel):
    """Enhanced request model for agent chat."""
    query: str
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    message_type: Optional[str] = "chat"  # chat, follow_up, clarification
    metadata: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None  # GCP project ID for routing

class ChatResponse(BaseModel):
    """Enhanced response model for agent chat."""
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

class SessionRequest(BaseModel):
    """Request model for session management."""
    user_id: str
    session_type: Optional[str] = "chat"
    metadata: Optional[Dict[str, Any]] = None

class ConversationRequest(BaseModel):
    """Request model for conversation management."""
    user_id: str
    session_id: str
    topic: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# In-memory stores for demonstration (in production, use Redis/database)
class ChatSessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.user_contexts: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
    def create_session(self, user_id: str, session_type: str = "chat") -> str:
        session_id = f"{user_id}_{int(time.time())}"
        self.sessions[session_id] = {
            "user_id": user_id,
            "session_type": session_type,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "conversation_count": 0,
            "active": True
        }
        return session_id
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)
        
    def update_session_activity(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = datetime.now()
            
    def create_conversation(self, session_id: str, topic: str = None) -> str:
        conversation_id = f"{session_id}_conv_{int(time.time())}"
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # Update session conversation count
        if session_id in self.sessions:
            self.sessions[session_id]["conversation_count"] += 1
            
        return conversation_id
        
    def add_message(self, conversation_id: str, message: Dict[str, Any]):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        message["timestamp"] = datetime.now().isoformat()
        self.conversations[conversation_id].append(message)
        
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        return self.conversations.get(conversation_id, [])
        
    def update_user_context(self, user_id: str, context_updates: Dict[str, Any]):
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {}
        self.user_contexts[user_id].update(context_updates)
        
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        return self.user_contexts.get(user_id, {})
        
    def record_performance_metrics(self, session_id: str, metrics: Dict[str, Any]):
        if session_id not in self.performance_metrics:
            self.performance_metrics[session_id] = []
        metrics["timestamp"] = datetime.now().isoformat()
        self.performance_metrics[session_id].append(metrics)
        
    def get_performance_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        return self.performance_metrics.get(session_id, [])

session_manager = ChatSessionManager()

@router.get("/")
async def get_agent_info(request: Request):
    """Get enhanced agent information and capabilities."""
    try:
        # Enhanced agent info with real-time capabilities
        agent_info = {
            "name": "Enhanced ADK Security Agent",
            "version": "2.0.0",
            "capabilities": [
                "real_time_chat",
                "multi_session_support", 
                "context_awareness",
                "delegation_tracking",
                "performance_monitoring",
                "websocket_support"
            ],
            "specialists": [
                "security_specialist",
                "iam_specialist", 
                "storage_specialist",
                "compliance_specialist",
                "recommendations_specialist",
                "incidents_specialist",
                "monitoring_specialist",
                "assets_specialist"
            ],
            "features": {
                "real_time_updates": True,
                "conversation_persistence": True,
                "context_tracking": True,
                "delegation_visibility": True,
                "performance_analytics": True,
                "multi_user_support": True
            }
        }
        
        return {
            "success": True,
            "agent_info": agent_info,
            "status": "ready",
            "websocket_endpoint": "/api/v1/agent/ws",
            "active_sessions": len(session_manager.sessions),
            "total_conversations": len(session_manager.conversations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent info: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    """Enhanced chat with ADK security agent with real-time capabilities."""
    start_time = time.time()
    
    try:
        # Create or get session
        session_id = chat_request.session_id
        if not session_id:
            session_id = session_manager.create_session(chat_request.user_id)
        else:
            session_manager.update_session_activity(session_id)
            
        # Create or get conversation
        conversation_id = chat_request.conversation_id
        if not conversation_id:
            conversation_id = session_manager.create_conversation(session_id)
            
        # Get user context
        user_context = session_manager.get_user_context(chat_request.user_id)
        if chat_request.context:
            user_context.update(chat_request.context)
            
        # Add message to conversation history
        session_manager.add_message(conversation_id, {
            "role": "user",
            "content": chat_request.query,
            "message_type": chat_request.message_type or "chat"
        })
        
        # Get conversation history for context
        conversation_history = session_manager.get_conversation_history(conversation_id)
        enhanced_context = {
            "chat_history": conversation_history,
            "user_context": user_context,
            "session_id": session_id,
            "conversation_id": conversation_id
        }
        
        # Process with smart routing based on query context
        try:
            from backend.api.query_router import query_router, Specialist
            
            project_id = chat_request.project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
            
            # Use smart router to detect specialist
            routing_decision = query_router.route_query(chat_request.query)
            specialist = routing_decision["specialist"]
            confidence = routing_decision["confidence"]
            routing_context = routing_decision["context"]
            
            logger.info(f"Query routing decision: {routing_decision['explanation']}")
            
            # Route to appropriate specialist
            if specialist == Specialist.STORAGE:
                # Route to storage analysis
                logger.info(f"Routing to Storage Specialist for: {chat_request.query}")
                
                try:
                    # Import and call storage analysis directly
                    from backend.api.storage import analyze_buckets
                    storage_data = await analyze_buckets(project_id, detailed=True)
                except Exception as e:
                    logger.error(f"Failed to get storage analysis: {e}")
                    storage_data = {"success": False, "error": str(e)}
                
                if storage_data.get("success"):
                    # Format the response with specific bucket findings
                    findings = storage_data.get("security_findings", {})
                    recommendations = storage_data.get("specific_recommendations", [])
                    immediate_actions = storage_data.get("immediate_actions", [])
                    
                    response_text = f"🔍 **Storage Security Analysis for Project: {project_id}**\n\n"
                    response_text += f"I analyzed {storage_data['summary']['total_buckets']} buckets in your project. Here are the specific findings:\n\n"
                    
                    # Critical issues
                    if findings.get("critical"):
                        response_text += "🚨 **CRITICAL ISSUES FOUND:**\n"
                        for issue in findings["critical"]:
                            response_text += f"• **{issue['bucket']}**: {issue['issue']}\n"
                            response_text += f"  → {issue['description']}\n"
                            response_text += f"  **Fix:** `{issue['remediation']}`\n\n"
                    
                    # High priority issues
                    if findings.get("high"):
                        response_text += "⚠️ **HIGH PRIORITY ISSUES:**\n"
                        for issue in findings["high"][:3]:  # Show top 3
                            response_text += f"• **{issue['bucket']}**: {issue['issue']}\n"
                            response_text += f"  → {issue['description']}\n\n"
                    
                    # Immediate actions
                    response_text += "📋 **IMMEDIATE ACTIONS REQUIRED:**\n"
                    for idx, action in enumerate(immediate_actions[:3], 1):
                        response_text += f"\n{idx}. {action['action']}\n"
                        response_text += f"   ```bash\n   {action['command']}\n   ```\n"
                        response_text += f"   Impact: {action['impact']}\n"
                    
                    result = {
                        "success": True,
                        "response": response_text,
                        "agent_used": "StorageSecuritySpecialist",
                        "delegation_path": ["SecurityAgent", "StorageSpecialist"],
                        "suggestions": [
                            "How do I fix the public access issue?",
                            "Show me the lifecycle policies I should implement",
                            "What's the impact of enabling versioning?",
                            "Help me set up CMEK encryption"
                        ],
                        "gcp_api_calls": ["storage.buckets.list", "storage.buckets.getIamPolicy"],
                        "findings_summary": storage_data.get("summary", {})
                    }
                else:
                    result = storage_data
            
            elif specialist == Specialist.IAM:
                # Route to IAM analysis
                logger.info(f"Routing to IAM Specialist")
                
                try:
                    # Import and call IAM analysis directly
                    from backend.api.iam import analyze_all_users
                    iam_data = await analyze_all_users(project_id)
                    
                    response_text = f"🔐 **IAM Security Analysis for Project: {project_id}**\n\n"
                    response_text += f"Analyzed {iam_data.get('total_users', 0)} users:\n"
                    response_text += f"• 🔴 High Risk Users: {iam_data.get('high_risk_users', 0)}\n"
                    response_text += f"• 🟡 Medium Risk Users: {iam_data.get('medium_risk_users', 0)}\n"
                    response_text += f"• 🟢 Low Risk Users: {iam_data.get('low_risk_users', 0)}\n\n"
                    
                    if iam_data.get('users'):
                        response_text += "**Top Findings:**\n"
                        for user in iam_data['users'][:3]:
                            response_text += f"• {user['email']}: {', '.join(user['roles'])} ({user['risk_level']} risk)\n"
                    
                    result = {
                        "success": True,
                        "response": response_text,
                        "agent_used": "IAMSecuritySpecialist",
                        "suggestions": [
                            "Show me users with owner roles",
                            "How do I implement least privilege?",
                            "Check service account permissions"
                        ]
                    }
                except Exception as e:
                    logger.error(f"IAM analysis failed: {e}")
                    result = {"success": False, "error": str(e)}
            
            elif specialist == Specialist.NETWORK:
                # Route to Network analysis
                logger.info(f"Routing to Network Specialist")
                
                try:
                    # Import and call network analysis directly
                    from backend.api.network import analyze_network_security
                    network_data = await analyze_network_security(project_id, detailed=True)
                    
                    if network_data.get("success"):
                        findings = network_data.get("security_findings", {})
                        
                        response_text = f"🌐 **Network Security Analysis for Project: {project_id}**\n\n"
                        
                        # Critical network issues
                        if findings.get("critical"):
                            response_text += "🚨 **CRITICAL NETWORK ISSUES:**\n"
                            for issue in findings["critical"]:
                                response_text += f"• {issue['resource']}: {issue['issue']}\n"
                                response_text += f"  Fix: `{issue['remediation']}`\n\n"
                        
                        # High priority network issues
                        if findings.get("high"):
                            response_text += "⚠️ **HIGH PRIORITY ISSUES:**\n"
                            for issue in findings["high"][:2]:
                                response_text += f"• {issue['resource']}: {issue['issue']}\n\n"
                        
                        # Immediate actions
                        actions = network_data.get("immediate_actions", [])
                        if actions:
                            response_text += "📋 **IMMEDIATE ACTIONS:**\n"
                            for action in actions[:3]:
                                response_text += f"• {action['action']}\n"
                                response_text += f"  ```bash\n  {action['command']}\n  ```\n"
                        
                        result = {
                            "success": True,
                            "response": response_text,
                            "agent_used": "NetworkSecuritySpecialist",
                            "suggestions": [
                                "How do I restrict SSH access?",
                                "Show me all open ports",
                                "Help me configure Cloud Armor",
                                "Review my VPC configuration"
                            ]
                        }
                    else:
                        result = network_data
                except Exception as e:
                    logger.error(f"Network analysis failed: {e}")
                    result = {"success": False, "error": str(e)}
            
            elif specialist == Specialist.COMPLIANCE:
                # Route to Compliance analysis
                logger.info(f"Routing to Compliance Specialist")
                
                try:
                    # Import and call compliance analysis directly
                    from backend.api.compliance import evaluate_compliance
                    from backend.api.compliance import ComplianceRequest
                    compliance_req = ComplianceRequest(
                        project_id=project_id,
                        frameworks=["SOC2", "ISO27001", "GDPR"]
                    )
                    compliance_data = await evaluate_compliance(compliance_req)
                    
                    response_text = f"📋 **Compliance Analysis for Project: {project_id}**\n\n"
                    response_text += f"Overall Compliance Score: **{compliance_data.get('overall_score', 0)}%**\n\n"
                    
                    frameworks = compliance_data.get('frameworks', {})
                    for fw_name, fw_data in frameworks.items():
                        emoji = "✅" if fw_data['score'] > 80 else "⚠️" if fw_data['score'] > 60 else "❌"
                        response_text += f"{emoji} **{fw_name}**: {fw_data['score']}% ({fw_data['status']})\n"
                        response_text += f"   • Findings: {fw_data['findings']} (Critical: {fw_data['critical']})\n"
                    
                    result = {
                        "success": True,
                        "response": response_text,
                        "agent_used": "ComplianceSpecialist",
                        "suggestions": [
                            "What do I need for SOC2 compliance?",
                            "Show me GDPR requirements",
                            "How do I improve my compliance score?"
                        ]
                    }
                except Exception as e:
                    logger.error(f"Compliance analysis failed: {e}")
                    result = {"success": False, "error": str(e)}
            
            elif specialist == Specialist.FINOPS:
                # Route to Cost/FinOps analysis
                logger.info(f"Routing to FinOps Specialist")
                
                try:
                    # Import and call cost analysis directly
                    from backend.api.cost import analyze_costs
                    cost_data = await analyze_costs(project_id, detailed=False, include_security=True)
                    
                    if cost_data.get("success"):
                        summary = cost_data.get("summary", {})
                        
                        response_text = f"💰 **Cost Analysis for Project: {project_id}**\n\n"
                        response_text += f"📊 **Current Month Spend**: {summary.get('current_month_spend', 'N/A')}\n"
                        response_text += f"📈 **Projected Spend**: {summary.get('projected_month_spend', 'N/A')}\n"
                        response_text += f"🎯 **Budget**: {summary.get('budget', 'N/A')}\n"
                        response_text += f"⚠️ **Status**: {summary.get('budget_status', 'UNKNOWN')}\n\n"
                        
                        response_text += f"💡 **Potential Monthly Savings**: {summary.get('total_potential_savings', '$0')}\n\n"
                        
                        # Top spending services
                        response_text += "**Top Spending Services:**\n"
                        for service in cost_data.get("top_spending_services", [])[:3]:
                            waste = f" (Waste: {service['waste']})" if service.get('waste') else ""
                            response_text += f"• {service['service']}: {service['cost']} ({service['percentage']}){waste}\n"
                        
                        # Immediate cost actions
                        response_text += "\n**🚀 Quick Wins to Save Money:**\n"
                        for action in cost_data.get("immediate_actions", [])[:3]:
                            response_text += f"\n• {action['action']}\n"
                            response_text += f"  Savings: {action['monthly_savings']}\n"
                            response_text += f"  ```bash\n  {action['command']}\n  ```\n"
                        
                        result = {
                            "success": True,
                            "response": response_text,
                            "agent_used": "FinOpsSpecialist",
                            "suggestions": [
                                "Show me unused resources",
                                "How can I reduce my compute costs?",
                                "What about committed use discounts?",
                                "Analyze my storage costs"
                            ]
                        }
                    else:
                        result = cost_data
                except Exception as e:
                    logger.error(f"Cost analysis failed: {e}")
                    result = {"success": False, "error": str(e)}
                    
            else:
                # Default routing for unmatched queries
                result = {
                    "success": True,
                    "response": f"I understand you're asking about: {chat_request.query}. Based on my analysis, this seems to be a {routing_context.get('query_type', 'general')} question.\n\nLet me help you with that. You can ask me about:\n• Storage security\n• IAM and permissions\n• Network configuration\n• Compliance status\n• Cost optimization",
                    "suggestions": [
                        "Analyze my security posture",
                        "Check my bucket security", 
                        "Review IAM permissions",
                        "Show network vulnerabilities",
                        "What's my compliance score?",
                        "How can I reduce costs?"
                    ],
                "agent_used": "fallback_agent"
            }
        except Exception as e:
            logger.error(f"Error in smart routing: {e}")
            # Fallback response if routing fails
            result = {
                "success": True,
                "response": f"I'll help you with: {chat_request.query}\n\nI can assist with:\n• Storage security analysis\n• IAM and permissions review\n• Network configuration checks\n• Compliance evaluation\n• Cost optimization",
                "suggestions": [
                    "Analyze my security posture",
                    "Check bucket security",
                    "Review IAM permissions"
                ],
                "agent_used": "error_fallback"
            }
        
        # Add response to conversation history
        session_manager.add_message(conversation_id, {
            "role": "assistant",
            "content": result.get("response", ""),
            "agent_used": result.get("agent_used"),
            "suggestions": result.get("suggestions", [])
        })
        
        # Calculate performance metrics
        response_time = time.time() - start_time
        metrics = {
            "response_time_ms": round(response_time * 1000, 2),
            "agent_used": result.get("agent_used"),
            "query_length": len(chat_request.query),
            "response_length": len(result.get("response", "")),
            "context_items": len(enhanced_context.get("chat_history", [])),
            "session_id": session_id
        }
        
        # Record metrics
        session_manager.record_performance_metrics(session_id, metrics)
        
        # Update user context with new information
        if result.get("context_updates"):
            session_manager.update_user_context(chat_request.user_id, result["context_updates"])
            
        # Broadcast real-time updates
        background_tasks.add_task(
            manager.broadcast_delegation_decision,
            {
                "user_id": chat_request.user_id,
                "session_id": session_id,
                "agent_used": result.get("agent_used"),
                "delegation_path": result.get("delegation_path", []),
                "response_time_ms": metrics["response_time_ms"]
            }
        )
        
        # Create enhanced response
        response = ChatResponse(
            success=result.get("success", True),
            response=result.get("response", ""),
            user_id=chat_request.user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            agent_used=result.get("agent_used"),
            delegation_path=result.get("delegation_path"),
            suggestions=result.get("suggestions"),
            context_updates=result.get("context_updates"),
            performance_metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process enhanced chat: {str(e)}"
        )


@router.post("/query")
async def query_agent(chat_request: ChatRequest, request: Request):
    """Send a query to the agent (alternative endpoint)."""
    try:
        agent_service = request.app.state.agent_service
        
        # Get response from agent service
        response = await agent_service.query_agent(
            query=chat_request.query,
            user_id=chat_request.user_id
        )
        
        return {
            "success": True,
            "response": response,
            "user_id": chat_request.user_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to query agent: {str(e)}"
        )


# WebSocket endpoint for real-time communication
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = "default"):
    """WebSocket endpoint for real-time agent communication."""
    await manager.connect(websocket, user_id)
    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_established",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }),
            user_id
        )
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    user_id
                )
            elif message_data.get("type") == "chat":
                # Process chat message in real-time
                response = f"Real-time processing: {message_data.get('message', '')}"
                await manager.send_personal_message(
                    json.dumps({
                        "type": "chat_response",
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }),
                    user_id
                )
            elif message_data.get("type") == "status_request":
                # Send current agent status
                await manager.send_personal_message(
                    json.dumps({
                        "type": "agent_status",
                        "status": "active",
                        "active_sessions": len(session_manager.sessions),
                        "timestamp": datetime.now().isoformat()
                    }),
                    user_id
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket, user_id)

# Enhanced session management endpoints
@router.post("/sessions")
async def create_session(session_request: SessionRequest):
    """Create a new chat session."""
    try:
        session_id = session_manager.create_session(
            session_request.user_id,
            session_request.session_type
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "user_id": session_request.user_id,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@router.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user."""
    try:
        user_sessions = [
            {**session_data, "session_id": session_id, "created_at": session_data["created_at"].isoformat()}
            for session_id, session_data in session_manager.sessions.items()
            if session_data["user_id"] == user_id
        ]
        
        return {
            "success": True,
            "user_id": user_id,
            "sessions": user_sessions,
            "total_sessions": len(user_sessions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user sessions: {str(e)}")

@router.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close a specific session."""
    try:
        if session_id in session_manager.sessions:
            session_manager.sessions[session_id]["active"] = False
            session_manager.sessions[session_id]["closed_at"] = datetime.now()
            
            return {
                "success": True,
                "message": f"Session {session_id} closed successfully",
                "session_id": session_id
            }
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close session: {str(e)}")

# Conversation management endpoints
@router.post("/conversations")
async def create_conversation(conversation_request: ConversationRequest):
    """Create a new conversation within a session."""
    try:
        conversation_id = session_manager.create_conversation(
            conversation_request.session_id,
            conversation_request.topic
        )
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "session_id": conversation_request.session_id,
            "topic": conversation_request.topic,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """Get conversation history."""
    try:
        history = session_manager.get_conversation_history(conversation_id)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "history": history,
            "message_count": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")

# Context management endpoints
@router.get("/context/{user_id}")
async def get_user_context(user_id: str):
    """Get user context information."""
    try:
        context = session_manager.get_user_context(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "context": context,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user context: {str(e)}")

@router.put("/context/{user_id}")
async def update_user_context(user_id: str, context_updates: Dict[str, Any]):
    """Update user context."""
    try:
        session_manager.update_user_context(user_id, context_updates)
        
        return {
            "success": True,
            "user_id": user_id,
            "message": "Context updated successfully",
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user context: {str(e)}")
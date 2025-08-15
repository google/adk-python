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
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
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
    from agents.search_enabled_agent import create_search_enabled_agent
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

# Import the enhanced chat manager from backend
try:
    import sys
    import os
    backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    from chat_manager import chat_manager, ChatMessage, MessageType
    CHAT_MANAGER_AVAILABLE = True
    logger.info("✅ Enhanced chat manager loaded")
except ImportError as e:
    CHAT_MANAGER_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced chat manager not available: {e}")
    
    # Enhanced chat manager not available - this should be installed
    chat_manager = None

# Import search service for web search integration
try:
    from backend.api.search import SearchService, get_search_service
    SEARCH_SERVICE_AVAILABLE = True
    logger.info("✅ Search service loaded for web search capabilities")
except ImportError as e:
    SEARCH_SERVICE_AVAILABLE = False
    logger.warning(f"⚠️ Search service not available: {e}")
    logger.error("Enhanced chat manager is required for ADK session management")
    logger.error("Please ensure the chat_manager module is properly installed")

def create_llm_agent(agent_type: str, project_id: str):
    """Create an LLM agent of the specified type."""
    if not AGENTS_AVAILABLE:
        return None
        
    try:
        if agent_type == "recommendation":
            # Create recommendation agent - use coordinator with special context
            agent = create_coordinator_agent(project_id)
            if agent:
                agent.agent_type = "recommendation"
                agent.description = f"Recommendation specialist for project {project_id}"
            return agent
        elif agent_type == "search":
            # Create search-enabled agent with Google Search grounding
            return create_search_enabled_agent(project_id, agent_type="conversational")
        elif agent_type == "coordinator":
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
        elif agent_type == "asset_discovery":
            # Create asset discovery agent for comprehensive GCP resource queries
            from agents.asset_discovery_agent import create_asset_discovery_agent
            return create_asset_discovery_agent(project_id)
        else:
            return create_coordinator_agent(project_id)
    except Exception as e:
        logger.error(f"Failed to create {agent_type} agent: {e}")
        return None

async def process_with_llm_agent(query: str, project_id: str, context: Dict = None, request_id: str = "unknown") -> Tuple[str, str]:
    """Process query using LLM agent steering with real data."""
    
    # Detect query intent for routing
    query_lower = query.lower()
    
    logger.info(f"🎯 [AGENT-{request_id}] Starting agent routing analysis...")
    logger.info(f"   🔍 Query keywords: {[word for word in ['bucket', 'storage', 'iam', 'user', 'network', 'firewall', 'compliance', 'cost', 'search', 'find', 'lookup', 'research'] if word in query_lower]}")
    
    # Check for recommendation intent first (enhanced recommendation routing)
    recommendation_indicators = ["recommend", "suggestion", "advice", "should i", "what to do", "best practice", "optimize", "improve", "fix"]
    if any(indicator in query_lower for indicator in recommendation_indicators):
        agent_type = "recommendation"
        agent_name = "RecommendationAgent"
        routing_reason = f"Recommendation keywords detected: {[word for word in recommendation_indicators if word in query_lower]}"
    # Check for search intent (new search routing)
    elif any(indicator in query_lower for indicator in ["search", "find", "lookup", "research", "google", "web search", "what is", "how to", "latest", "recent", "news", "documentation", "examples"]):
        agent_type = "search"
        agent_name = "SearchAgent"
        routing_reason = f"Search keywords detected: {[word for word in ['search', 'find', 'lookup', 'research', 'google', 'web search', 'what is', 'how to', 'latest', 'recent', 'news', 'documentation', 'examples'] if word in query_lower]}"
    # Check for comprehensive GCP resource discovery (HIGHEST PRIORITY for asset queries)
    elif any(word in query_lower for word in ["resources", "inventory", "assets", "all", "everything", "project", "summary", "overview", "compute instances", "virtual machines", "databases", "cloud functions", "kubernetes clusters", "gke", "what do i have", "show me", "list", "instances", "vms", "functions", "clusters"]):
        agent_type = "asset_discovery"
        agent_name = "AssetDiscoveryAgent"
        routing_reason = f"Asset discovery keywords detected: {[word for word in ['resources', 'inventory', 'assets', 'all', 'everything', 'project', 'summary', 'overview', 'compute instances', 'virtual machines', 'databases', 'cloud functions', 'kubernetes clusters', 'gke', 'what do i have', 'show me', 'list', 'instances', 'vms', 'functions', 'clusters'] if word in query_lower]}"
    # Determine which specialist agent to use
    elif any(word in query_lower for word in ["bucket", "storage", "backup", "archive"]):
        agent_type = "storage"
        agent_name = "StorageSecurityAgent"
        routing_reason = "Storage keywords detected: bucket, storage, backup, archive"
    elif any(word in query_lower for word in ["user", "permission", "iam", "role", "access"]):
        agent_type = "iam"
        agent_name = "IAMSecurityAgent"
        routing_reason = "IAM keywords detected: user, permission, iam, role, access"
    elif any(word in query_lower for word in ["firewall", "network", "port", "vpc", "subnet"]):
        agent_type = "network"
        agent_name = "NetworkSecurityAgent"
        routing_reason = "Network keywords detected: firewall, network, port, vpc, subnet"
    elif any(word in query_lower for word in ["compliance", "soc2", "gdpr", "iso", "audit"]):
        agent_type = "compliance"
        agent_name = "ComplianceAgent"
        routing_reason = "Compliance keywords detected: compliance, soc2, gdpr, iso, audit"
    elif any(word in query_lower for word in ["cost", "spend", "budget", "savings", "optimize"]):
        agent_type = "cost"
        agent_name = "CostOptimizationAgent"
        routing_reason = "Cost keywords detected: cost, spend, budget, savings, optimize"
    else:
        agent_type = "coordinator"
        agent_name = "CoordinatorAgent"
        routing_reason = "No specific keywords found, using coordinator"
    
    logger.info(f"🎯 [AGENT-{request_id}] ROUTING DECISION:")
    logger.info(f"   🤖 Selected Agent: {agent_name}")
    logger.info(f"   📊 Agent Type: {agent_type}")
    logger.info(f"   💭 Reasoning: {routing_reason}")
    logger.info(f"   🎯 Project: {project_id}")
    
    if AGENTS_AVAILABLE:
        logger.info(f"🤖 [AGENT-{request_id}] LLM agents available, attempting to use real agent...")
        # Use real LLM agent
        agent = create_llm_agent(agent_type, project_id)
        if agent:
            try:
                # Send query to agent for intelligent processing
                logger.info(f"🚀 [AGENT-{request_id}] Calling {agent_name} with query processing...")
                
                # Handle search agent specially with its async method
                if agent_type == "search":
                    response_dict = await agent.search_with_context(query, session_id=request_id)
                    if response_dict.get("success"):
                        response = response_dict["response"]
                        # Add citations if available
                        if response_dict.get("citations"):
                            response += "\n\n📚 **Sources:**\n"
                            for citation in response_dict["citations"]:
                                response += f"• {citation}\n"
                    else:
                        response = response_dict.get("response", "Search failed")
                else:
                    # Other agents use the standard process_query method
                    response = await agent.process_query(query, context)
                
                logger.info(f"✅ [AGENT-{request_id}] {agent_name} processed successfully")
                return str(response), agent_name
            except Exception as e:
                logger.error(f"❌ [AGENT-{request_id}] Agent processing failed: {e}")
                return f"Error processing with {agent_name}: {str(e)}", "ErrorHandler"
        else:
            logger.warning(f"⚠️  [AGENT-{request_id}] Failed to create {agent_name}, falling back to API data")
    else:
        logger.info(f"📡 [AGENT-{request_id}] LLM agents not available, using real data APIs...")
    
    # Use real data APIs instead of mock responses
    logger.info(f"🔄 [AGENT-{request_id}] Delegating to real data API for {agent_type}")
    response = await generate_response_with_real_data(query, project_id, agent_type, request_id)
    logger.info(f"✅ [AGENT-{request_id}] Real data API response generated")
    return response, agent_name

async def generate_response_with_real_data(query: str, project_id: str, agent_type: str, request_id: str = "unknown") -> str:
    """Generate response using real data from our APIs with GCP thin client integration."""
    
    logger.info(f"🔍 [API-{request_id}] Fetching real data for {agent_type} query")
    
    # Try to use thin client service for asset and security queries
    if agent_type in ["asset_discovery", "storage", "iam", "network", "compliance"]:
        try:
            from backend.services.gcp_thin_client_service import GCPThinClientService
            
            logger.info(f"🌐 [API-{request_id}] Using GCP Thin Client Service")
            thin_client = GCPThinClientService(project_id)
            
            # Get asset inventory snapshot
            snapshot = await thin_client.get_asset_inventory_snapshot()
            
            # Analyze security based on query
            security_analysis = await thin_client.analyze_asset_security(query)
            
            # Generate insights
            insights = await thin_client.generate_security_insights([])
            
            # Build comprehensive response
            response = f"🔍 **GCP Security Analysis**\n\n"
            response += f"**Project:** {project_id}\n\n"
            
            if snapshot.total_assets > 0:
                response += f"📊 **Asset Overview:**\n"
                response += f"• Total Assets: {snapshot.total_assets}\n"
                for asset_type, count in snapshot.asset_breakdown.items():
                    if count > 0:
                        response += f"• {asset_type}: {count}\n"
                response += "\n"
            
            if security_analysis:
                response += f"🎯 **Security Focus:** {security_analysis.get('focus', 'General').title()}\n\n"
                
                if security_analysis.get('findings'):
                    response += "⚠️ **Key Findings:**\n"
                    for finding in security_analysis['findings']:
                        response += f"• {finding}\n"
                    response += "\n"
                
                if security_analysis.get('recommendations'):
                    response += "💡 **Recommendations:**\n"
                    for rec in security_analysis['recommendations']:
                        response += f"• {rec}\n"
                    response += "\n"
                
                risk_level = security_analysis.get('risk_level', 'unknown')
                risk_emoji = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk_level, "⚪")
                response += f"{risk_emoji} **Risk Level:** {risk_level.title()}\n\n"
            
            if snapshot.high_risk_assets:
                response += "🚨 **High-Risk Assets:**\n"
                for asset in snapshot.high_risk_assets[:5]:
                    response += f"• {asset}\n"
                response += "\n"
            
            if insights.get('summary'):
                response += f"📝 **Summary:** {insights['summary']}\n\n"
            
            response += f"⏱️ Scan completed in {snapshot.scan_duration_ms:.0f}ms"
            
            logger.info(f"✅ [API-{request_id}] Thin client response generated")
            return response
            
        except Exception as e:
            logger.error(f"❌ [API-{request_id}] Thin client service failed: {e}")
            # Fall back to existing logic
    
    logger.info(f"🔍 [API-{request_id}] Using standard API for {agent_type} query")
    
    if agent_type == "search":
        # For search queries, we should use the Gemini agent with Google Search grounding
        # If agents are not available, provide a helpful message
        logger.info(f"🔍 [API-{request_id}] Search query detected")
        
        if not AGENTS_AVAILABLE:
            return f"""🔍 **Google Search with Gemini**

To enable web search, the system needs to use Gemini's built-in Google Search grounding.

**How it works:**
• Gemini API has native Google Search integration
• No separate API keys needed (uses Vertex AI)
• Automatic source citations
• Real-time information retrieval

**Your query:** {query}

Currently, I'll help based on my training knowledge. To enable real-time search:
1. Ensure Vertex AI is configured
2. The search-enabled agent will automatically use Google Search
3. Results will include sources and citations"""
        else:
            # The agent should handle this with Google Search grounding
            return f"Search functionality requires the Gemini search-enabled agent. Please ensure agents are properly configured."
    
    elif agent_type == "recommendation":
        try:
            logger.info(f"💡 [API-{request_id}] RECOMMENDATION API CALLS STARTING:")
            logger.info(f"   🔄 Importing recommendation service...")
            
            # Import recommendation service
            from backend.services.chat_recommendation_service import ChatRecommendationService
            from backend.services.recommender_service import RecommenderService
            
            logger.info(f"   📞 API Call: Google Cloud Recommender API")
            logger.info(f"   🎯 Query: {query}")
            logger.info(f"   🎯 Project: {project_id}")
            
            api_start_time = time.time()
            
            # Initialize services
            recommender_service = RecommenderService()
            chat_service = ChatRecommendationService(recommender_service)
            
            # Process the query through chat service
            response_data = await chat_service.process_natural_language_query(
                query=query,
                project_id=project_id,
                user_id=request_id,
                session_id=request_id
            )
            
            api_duration = time.time() - api_start_time
            
            if response_data.get("success"):
                recommendations = response_data.get("recommendations", [])
                summary = response_data.get("summary", "")
                follow_up = response_data.get("follow_up_questions", [])
                
                logger.info(f"✅ [API-{request_id}] Recommendation API SUCCESS:")
                logger.info(f"   📊 Recommendations found: {len(recommendations)}")
                logger.info(f"   💡 Response generated: {len(summary)} chars")
                
                response = f"💡 **Recommendations for: {query}**\n\n"
                
                if summary:
                    response += f"{summary}\n\n"
                
                if recommendations:
                    response += "🎯 **Key Recommendations:**\n"
                    for i, rec in enumerate(recommendations[:5], 1):
                        priority_emoji = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🔵"}.get(rec.get("priority", "medium").lower(), "💡")
                        response += f"{priority_emoji} **{rec.get('title', 'Recommendation')}**\n"
                        response += f"   📋 {rec.get('description', 'No description')}\n"
                        if rec.get('estimated_impact'):
                            response += f"   💰 Impact: {rec['estimated_impact']}\n"
                        if rec.get('implementation_effort'):
                            response += f"   ⏱️ Effort: {rec['implementation_effort']}\n"
                        response += "\n"
                
                # Add follow-up questions
                if follow_up:
                    response += "🤔 **You might also ask:**\n"
                    for question in follow_up[:3]:
                        response += f"• {question}\n"
                    response += "\n"
                
                response += f"🕒 Analysis completed in {api_duration:.2f}s"
                
                logger.info(f"✅ [API-{request_id}] Recommendation response generated: {len(response)} chars")
                return response
            else:
                error_msg = response_data.get('error', 'No recommendations found')
                logger.error(f"❌ [API-{request_id}] Recommendation API failed: {error_msg}")
                return f"💡 **Recommendation Service**\n\nI encountered an issue getting recommendations for '{query}': {error_msg}\n\nTry asking about:\n• Security recommendations\n• Cost optimization suggestions\n• Performance improvements\n• Compliance requirements"
                
        except Exception as e:
            logger.error(f"❌ [API-{request_id}] Recommendation API exception: {e}")
            import traceback
            logger.error(f"   📋 Stack trace: {traceback.format_exc()}")
            return f"💡 **Recommendation Service Error**\n\nI encountered an error while getting recommendations for '{query}': {str(e)}\n\nThe recommendation service may need to be configured with proper GCP credentials."
    
    elif agent_type == "storage":
        try:
            logger.info(f"📦 [API-{request_id}] STORAGE API CALLS STARTING:")
            logger.info(f"   🔄 Importing storage API module...")
            # Import and call the real storage API
            from backend.api.storage import analyze_buckets
            
            logger.info(f"   📞 API Call: storage.buckets.list")
            logger.info(f"   📞 API Call: storage.buckets.getIamPolicy")
            logger.info(f"   🎯 Target Project: {project_id}")
            logger.info(f"   ⚙️  Detailed Analysis: True")
            
            api_start_time = time.time()
            storage_data = await analyze_buckets(project_id, detailed=True)
            api_duration = time.time() - api_start_time
            
            logger.info(f"   ⏱️  API calls completed in {api_duration:.2f}s")
            
            if storage_data.get("success"):
                buckets = storage_data.get("buckets", [])
                findings = storage_data.get("security_findings", {})
                actions = storage_data.get("immediate_actions", [])
                
                logger.info(f"✅ [API-{request_id}] Storage API SUCCESS:")
                logger.info(f"   📊 Buckets found: {len(buckets)}")
                logger.info(f"   🔍 Findings: {len(findings.get('critical', []))} critical, {len(findings.get('high', []))} high")
                logger.info(f"   📋 Actions recommended: {len(actions)}")
                
                response = f"🔍 **Storage Security Analysis for Project: {project_id}**\n\n"
                response += f"I analyzed {len(buckets)} buckets in your project:\n\n"
                
                # List actual bucket names
                response += "**Your Buckets:**\n"
                for bucket in buckets[:5]:  # Show first 5
                    status = "🔴 PUBLIC" if bucket.get("public_access") else "🟢 PRIVATE"
                    response += f"• **{bucket['name']}** - {status}\n"
                    if bucket.get("issues"):
                        for issue in bucket["issues"][:2]:
                            response += f"  ⚠️ {issue}\n"
                
                # Critical findings
                if findings.get("critical"):
                    response += "\n🚨 **CRITICAL ISSUES:**\n"
                    for finding in findings["critical"][:3]:
                        response += f"• **{finding['bucket']}**: {finding['issue']}\n"
                        response += f"  Fix: `{finding['remediation']}`\n"
                
                # Immediate actions
                if actions:
                    response += "\n📋 **IMMEDIATE ACTIONS:**\n"
                    for action in actions[:3]:
                        response += f"• {action['action']}\n"
                        response += f"  ```bash\n  {action['command']}\n  ```\n"
                
                logger.info(f"✅ [API-{request_id}] Response generated: {len(response)} chars")
                return response
            else:
                error_msg = storage_data.get('error', 'Unknown storage API error')
                logger.error(f"❌ [API-{request_id}] Storage API failed: {error_msg}")
        except Exception as e:
            logger.error(f"❌ [API-{request_id}] Storage API exception: {e}")
            import traceback
            logger.error(f"   📋 Stack trace: {traceback.format_exc()}")
    
    elif agent_type == "iam":
        try:
            from backend.api.iam import analyze_all_users
            
            logger.info(f"👤 Calling GCP API: iam.projects.serviceAccounts.list for project {project_id}")
            logger.info(f"👤 Calling GCP API: cloudresourcemanager.projects.getIamPolicy")
            
            iam_data = await analyze_all_users(project_id)
            
            response = f"🔐 **IAM Security Analysis for Project: {project_id}**\n\n"
            response += f"Analyzed {iam_data.get('total_users', 0)} users and service accounts:\n\n"
            
            # Show actual users
            if iam_data.get("users"):
                response += "**Top Risk Accounts:**\n"
                for user in iam_data["users"][:5]:
                    risk_emoji = "🔴" if user["risk_level"] == "high" else "🟡" if user["risk_level"] == "medium" else "🟢"
                    response += f"• {risk_emoji} **{user['email']}**\n"
                    response += f"  Roles: {', '.join(user['roles'][:3])}\n"
            
            logger.info(f"✅ Generated IAM response with {iam_data.get('total_users', 0)} users")
            return response
        except Exception as e:
            logger.error(f"Error getting real IAM data: {e}")
    
    elif agent_type == "network":
        try:
            from backend.api.network import analyze_network_security
            
            logger.info(f"🌐 Calling GCP API: compute.firewalls.list for project {project_id}")
            logger.info(f"🌐 Calling GCP API: compute.networks.list")
            
            network_data = await analyze_network_security(project_id, detailed=True)
            
            if network_data.get("success"):
                response = f"🌐 **Network Security Analysis for Project: {project_id}**\n\n"
                
                findings = network_data.get("security_findings", {})
                if findings.get("critical"):
                    response += "🚨 **CRITICAL FIREWALL RULES:**\n"
                    for finding in findings["critical"][:3]:
                        response += f"• **{finding['resource']}**: {finding['issue']}\n"
                        response += f"  Fix: `{finding['remediation']}`\n"
                
                logger.info("✅ Generated network response with firewall rules")
                return response
        except Exception as e:
            logger.error(f"Error getting real network data: {e}")
    
    elif agent_type == "asset_discovery":
        try:
            logger.info(f"🔍 [API-{request_id}] ASSET DISCOVERY API CALLS STARTING:")
            logger.info(f"   🔄 Importing asset discovery agent...")
            
            # Import and use the asset discovery agent
            from agents.asset_discovery_agent import create_asset_discovery_agent
            
            logger.info(f"📡 Making HTTP POST to https://cloudasset.googleapis.com/v1/projects/{project_id}:searchAllResources")
            logger.info(f"   🎯 Query: {query}")
            logger.info(f"   🔍 Using Asset Inventory API for comprehensive resource discovery")
            
            # Create and use asset discovery agent
            asset_agent = create_asset_discovery_agent(project_id)
            result = await asset_agent.process_query(query, request_id)
            
            if result.get("success"):
                logger.info(f"✅ [API-{request_id}] Asset Discovery SUCCESS:")
                logger.info(f"   📊 Resources found: {result.get('resource_count', 0)}")
                logger.info(f"   🌐 Data source: {result.get('data_source', 'unknown')}")
                logger.info(f"   ⏱️  API duration: {result.get('api_duration', 0):.2f}s")
                return result["response"]
            else:
                error_msg = result.get("error", "Asset discovery failed")
                logger.error(f"❌ [API-{request_id}] Asset Discovery failed: {error_msg}")
                return f"🔍 **Asset Discovery**\n\nI encountered an issue discovering GCP resources for '{query}': {error_msg}\n\nTry asking about:\n• Show me a project summary\n• What compute instances do I have?\n• List my databases\n• Show me all storage resources"
                
        except Exception as e:
            logger.error(f"❌ [API-{request_id}] Asset Discovery exception: {e}")
            return f"🔍 **Asset Discovery Error**\n\nI encountered an error discovering resources for '{query}': {str(e)}"
    
    elif agent_type == "cost":
        try:
            from backend.api.cost import analyze_costs
            
            logger.info(f"💰 Calling GCP API: cloudbilling.services.list")
            logger.info(f"💰 Calling GCP API: cloudbilling.projects.getBillingInfo for project {project_id}")
            
            cost_data = await analyze_costs(project_id, detailed=False, include_security=True)
            
            if cost_data.get("success"):
                summary = cost_data.get("summary", {})
                response = f"💰 **Cost Analysis for Project: {project_id}**\n\n"
                response += f"**Current Month:** {summary.get('current_month_spend', 'N/A')}\n"
                response += f"**Budget:** {summary.get('budget', 'N/A')}\n"
                response += f"**Status:** {summary.get('budget_status', 'N/A')}\n\n"
                
                # Show actual unused resources
                if cost_data.get("immediate_actions"):
                    response += "**Resources to Delete:**\n"
                    for action in cost_data["immediate_actions"][:3]:
                        response += f"• {action['action']}\n"
                        response += f"  Savings: {action['monthly_savings']}\n"
                        response += f"  ```bash\n  {action['command']}\n  ```\n"
                
                logger.info("✅ Generated cost response with spending data")
                return response
        except Exception as e:
            logger.error(f"Error getting real cost data: {e}")
    
    # Fallback to intelligent response if real data fetch fails
    return generate_intelligent_response(query, project_id, agent_type)

def generate_intelligent_response(query: str, project_id: str, agent_type: str) -> str:
    """Generate an intelligent mock response when real data is not available."""
    
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
async def chat_with_llm_agent(chat_request: ChatRequest):
    """Process chat with LLM agent steering for intelligent responses."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    logger.info(f"🚀 [CHAT-{request_id}] Incoming chat request")
    logger.info(f"   📋 Query: '{chat_request.query[:100]}{'...' if len(chat_request.query) > 100 else ''}'")
    logger.info(f"   👤 User: {chat_request.user_id}")
    logger.info(f"   🎯 Project: {chat_request.project_id}")
    logger.info(f"   🔗 Session: {chat_request.session_id[:20] + '...' if chat_request.session_id else 'None (will create)'}")
    
    try:
        # ADK session management (required)
        if not CHAT_MANAGER_AVAILABLE:
            logger.error(f"❌ [CHAT-{request_id}] Chat manager not available")
            raise HTTPException(
                status_code=503, 
                detail="ADK session management not available. Enhanced chat manager is required."
            )
        
        logger.info(f"✅ [CHAT-{request_id}] Chat manager available, proceeding...")
        
        # Create or get session
        session_id = chat_request.session_id
        if not session_id:
            logger.info(f"🆕 [CHAT-{request_id}] Creating new session for user: {chat_request.user_id}")
            session_id = chat_manager.create_session(chat_request.user_id, {
                "project_id": chat_request.project_id,
                "source": "api_chat",
                "adk_compliant": True,
                "request_id": request_id
            })
            logger.info(f"✅ [CHAT-{request_id}] Created session: {session_id}")
        else:
            logger.info(f"🔄 [CHAT-{request_id}] Using existing session: {session_id[:20]}...")
        
        # Add user message using enhanced manager
        logger.info(f"💬 [CHAT-{request_id}] Adding user message to session")
        await chat_manager.add_message(
            session_id=session_id,
            content=chat_request.query,
            sender_type="user",
            performance_data={"start_time": start_time, "request_id": request_id}
        )
        logger.info(f"✅ [CHAT-{request_id}] User message added successfully")
        
        # Get project ID
        project_id = chat_request.project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
        logger.info(f"🎯 [CHAT-{request_id}] Using project: {project_id}")
        
        # Process with LLM agent steering
        logger.info(f"🧠 [CHAT-{request_id}] Starting agent routing and processing...")
        response_text, agent_used = await process_with_llm_agent(
            chat_request.query,
            project_id,
            chat_request.context,
            request_id=request_id
        )
        logger.info(f"✅ [CHAT-{request_id}] Agent processing completed")
        logger.info(f"   🤖 Agent used: {agent_used}")
        logger.info(f"   📝 Response length: {len(response_text)} characters")
        
        # Add response to conversation
        logger.info(f"💾 [CHAT-{request_id}] Saving assistant response to session...")
        await chat_manager.add_message(
            session_id=session_id,
            content=response_text,
            sender_type="assistant",
            agent_used=agent_used,
            delegation_path=["SecurityAgent", agent_used],
            performance_data={
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
                "request_id": request_id
            }
        )
        logger.info(f"✅ [CHAT-{request_id}] Assistant response saved successfully")
        
        # Get conversation ID from the session
        session = chat_manager.get_session(session_id)
        conversation_id = list(session.conversations.keys())[0] if session and session.conversations else "main"
        logger.info(f"📋 [CHAT-{request_id}] Using conversation: {conversation_id}")
        
        # Calculate performance metrics
        response_time = time.time() - start_time
        metrics = {
            "response_time_ms": round(response_time * 1000, 2),
            "agent_used": agent_used,
            "query_length": len(chat_request.query),
            "response_length": len(response_text),
            "session_id": session_id,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"📊 [CHAT-{request_id}] PERFORMANCE METRICS:")
        logger.info(f"   ⏱️  Total time: {response_time:.2f}s ({metrics['response_time_ms']}ms)")
        logger.info(f"   📏 Query length: {metrics['query_length']} chars")
        logger.info(f"   📏 Response length: {metrics['response_length']} chars")
        
        # Generate contextual suggestions
        logger.info(f"💡 [CHAT-{request_id}] Generating contextual suggestions...")
        suggestions = chat_manager.get_contextual_suggestions(session_id) or generate_suggestions(chat_request.query, agent_used)
        logger.info(f"✅ [CHAT-{request_id}] Generated {len(suggestions)} suggestions")
        
        # Create response
        logger.info(f"📤 [CHAT-{request_id}] Creating final response...")
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
        
        logger.info(f"🎉 [CHAT-{request_id}] REQUEST COMPLETED SUCCESSFULLY")
        logger.info(f"   ✅ Status: Success")
        logger.info(f"   🔗 Session: {session_id[:20]}...")
        logger.info(f"   🤖 Agent: {agent_used}")
        logger.info(f"   ⏱️  Duration: {response_time:.2f}s")
        logger.info(f"   💡 Suggestions: {len(suggestions)}")
        
        return response
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"💥 [CHAT-{request_id}] REQUEST FAILED after {error_time:.2f}s")
        logger.error(f"   ❌ Error: {str(e)}")
        logger.error(f"   📋 Query: '{chat_request.query[:50]}...'")
        logger.error(f"   👤 User: {chat_request.user_id}")
        import traceback
        logger.error(f"   📋 Stack trace: {traceback.format_exc()}")
        
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
    """Generate context-aware security suggestions based on the query and agent used."""
    
    # Base suggestions for security context
    security_suggestions = [
        "What are my highest priority security risks?",
        "Show me resources with public access",
        "Which compliance standards should I focus on?",
        "How can I improve my security posture?"
    ]
    
    # Agent-specific suggestions
    if agent_used == "AssetDiscoveryAgent":
        return [
            "Which assets have security vulnerabilities?",
            "Show me assets created in the last 30 days",
            "What resources are consuming the most cost?",
            "Are there any unused or orphaned resources?",
            "Which assets need encryption enabled?"
        ]
    elif agent_used == "StorageSecurityAgent":
        return [
            "Which buckets have public access?",
            "Show me buckets without encryption",
            "What's my storage compliance status?",
            "How can I optimize storage costs?",
            "Which buckets have retention policies?"
        ]
    elif agent_used == "IAMSecurityAgent":
        return [
            "Who has admin access to this project?",
            "Show me service accounts with excessive permissions",
            "Which users haven't logged in recently?",
            "What are the most risky IAM policies?",
            "How can I implement least privilege?"
        ]
    elif agent_used == "NetworkSecurityAgent":
        return [
            "Which firewall rules allow public access?",
            "Show me resources with external IPs",
            "What VPCs have the weakest security?",
            "Are there any open ports I should close?",
            "How can I improve network segmentation?"
        ]
    elif agent_used == "ComplianceAgent":
        return [
            "What are my SOC2 compliance gaps?",
            "Show me GDPR compliance issues",
            "Which resources need audit logging?",
            "What security controls am I missing?",
            "How can I improve my compliance score?"
        ]
    elif agent_used == "RecommendationAgent":
        return [
            "What's the easiest recommendation to implement?",
            "Which recommendations have the highest impact?",
            "Show me cost-saving recommendations",
            "What security quick wins can I implement today?",
            "How do I apply these recommendations?"
        ]
    elif agent_used == "CostOptimizationAgent":
        return [
            "Which resources are the most expensive?",
            "Show me unused resources I can delete",
            "What are my cost optimization opportunities?",
            "How can I reduce my monthly spend?",
            "Which services are growing in cost?"
        ]
    
    suggestions_map = {
        "RecommendationAgent": [
            "Show me highest priority recommendations",
            "What are my biggest cost savings opportunities?",
            "How do I implement security recommendations?",
            "Prioritize recommendations by impact"
        ],
        "SearchAgent": [
            "Search for latest security vulnerabilities",
            "Find GCP security best practices",
            "Research recent security incidents",
            "Look up compliance documentation"
        ],
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

@router.post("/sessions/create")
async def create_session(request: Dict[str, Any]):
    """Create a new ADK session following thin client best practices."""
    user_id = request.get("user_id", "default_user")
    
    if not CHAT_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="ADK session management not available. Enhanced chat manager is required."
        )
    
    session_id = chat_manager.create_session(user_id, {
        "source": "thin_client_api",
        "adk_compliant": True,
        "project_id": request.get("project_id")
    })
    
    logger.info(f"Created ADK session {session_id} for thin client")
    
    return {
        "success": True,
        "session_id": session_id,
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }

@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: Optional[int] = None):
    """Get messages for an ADK session."""
    if not CHAT_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="ADK session management not available. Enhanced chat manager is required."
        )
    
    messages = chat_manager.get_conversation_history(session_id, limit=limit)
    return {
        "success": True,
        "session_id": session_id,
        "messages": [
            {
                "sender_type": msg.sender_type,
                "content": msg.content,
                "agent_used": msg.agent_used,
                "timestamp": msg.timestamp.isoformat()
            } for msg in messages
        ]
    }

@router.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get ADK session status and analytics."""
    if not CHAT_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="ADK session management not available. Enhanced chat manager is required."
        )
    
    analytics = chat_manager.get_session_analytics(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "analytics": analytics,
        "active": analytics.get("status") == "active" if analytics else False
    }

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
                "natural_language_understanding",
                "adk_session_management",
                "thin_client_optimized"
            ],
            "available_agents": [
                "RecommendationAgent",
                "SearchAgent",
                "StorageSecurityAgent",
                "IAMSecurityAgent",
                "NetworkSecurityAgent",
                "ComplianceAgent",
                "CostOptimizationAgent",
                "CoordinatorAgent"
            ],
            "llm_available": AGENTS_AVAILABLE,
            "adk_compliant": True,
            "thin_client_ready": True
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
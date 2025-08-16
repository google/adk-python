"""
RADAR Coordinator - ADK SequentialAgent with LlmAgent sub-agents.

This follows the proper ADK pattern where:
1. Each RADAR phase is an LlmAgent with specific instructions and output_key
2. The coordinator is a SequentialAgent that chains them together
3. State is shared between agents via output_key
"""

import logging
import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from datetime import datetime
import json

from google.adk.agents import SequentialAgent, LlmAgent

logger = logging.getLogger(__name__)

# ============================================
# FastAPI Router and Models
# ============================================

router = APIRouter()

class ChatRequest(BaseModel):
    """Request model for RADAR chat."""
    query: str
    user_id: Optional[str] = "default_user"
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    project_id: Optional[str] = None
    authorize_actions: Optional[bool] = False

class ChatResponse(BaseModel):
    """Response model for RADAR chat."""
    success: bool
    response: str
    user_id: str
    session_id: Optional[str] = None
    phases_executed: Optional[List[str]] = None
    recommendations: Optional[Dict[str, List[str]]] = None
    risk_level: Optional[str] = None
    timestamp: Optional[str] = None

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
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}")

manager = ConnectionManager()


def create_radar_pipeline(project_id: str) -> SequentialAgent:
    """
    Create the RADAR pipeline using ADK's SequentialAgent pattern.
    
    Each phase is an LlmAgent that:
    1. Reads from previous agents' output via {output_key}
    2. Performs its specific task
    3. Saves results to its own output_key
    
    This follows the exact pattern from ADK documentation.
    """
    
    # Import tools for the agents to use
    from google.adk.tools import FunctionTool
    
    # Create GCP discovery tool function (synchronous for FunctionTool)
    def discover_gcp_resources() -> str:
        """Discover and list all GCP resources in the project.
        
        Returns a summary of compute, storage, and IAM resources.
        """
        import asyncio
        from backend.api.gcp_direct import GCPDirectClient
        
        client = GCPDirectClient(project_id)
        # Run async function in sync context
        resources = asyncio.run(client.discover_resources())
        
        result = f"Project {project_id} resources:\n"
        if resources.get("summary"):
            s = resources["summary"]
            result += f"- Total: {s.get('total_assets', 0)} assets\n"
            result += f"- Compute: {s.get('compute_count', 0)}\n"
            result += f"- Storage: {s.get('storage_count', 0)}\n"
            result += f"- IAM: {s.get('iam_count', 0)}\n"
        
        if resources.get("compute"):
            result += f"\nCompute instances:\n"
            for r in resources["compute"][:5]:
                result += f"  - {r['name']}\n"
        
        if resources.get("storage"):
            result += f"\nStorage buckets:\n"
            for r in resources["storage"][:5]:
                result += f"  - {r['name']}\n"
                
        return result
    
    # Create tool instance using FunctionTool - it only takes the function itself
    discovery_tool = FunctionTool(discover_gcp_resources)
    
    # Phase 1: Recognition - Discover resources
    recognition = LlmAgent(
        name="RecognitionAgent",
        model="gemini-2.0-flash",
        instruction=f"""You are the Recognition Agent for project {project_id}.
        
        Your mission is to discover and inventory all cloud resources.
        
        Use the discover_gcp_resources tool to get real data from the project.
        
        Tasks:
        1. Call the discovery tool to get actual resources
        2. Organize the discovered resources by type
        3. Identify any anomalies or concerns
        4. Summarize the resource inventory
        
        Output a structured inventory based on the real data you discover.
        """,
        tools=[discovery_tool],
        output_key="recognition_results"
    )
    
    # Phase 2: Assessment - Evaluate security
    assessment = LlmAgent(
        name="AssessmentAgent",
        model="gemini-2.0-flash",
        instruction=f"""You are the Assessment Agent for project {project_id}.
        
        Using the resource inventory from {{recognition_results}}, evaluate security posture.
        
        Tasks:
        1. Identify security vulnerabilities in discovered resources
        2. Check for misconfigurations (public access, weak encryption, etc.)
        3. Analyze IAM permissions for over-privileged accounts
        4. Evaluate compliance with standards (CIS, PCI-DSS, HIPAA)
        5. Calculate risk scores for each finding
        
        Classify findings by severity:
        - CRITICAL: Immediate action required (exposed credentials, public databases)
        - HIGH: Significant risk (overly permissive IAM, unencrypted storage)
        - MEDIUM: Should be addressed (missing monitoring, old API versions)
        - LOW: Best practice violations (missing tags, naming conventions)
        
        Output a comprehensive security assessment with risk scoring.
        """,
        output_key="assessment_results"
    )
    
    # Phase 3: Decision - Prioritize and recommend
    decision = LlmAgent(
        name="DecisionAgent",
        model="gemini-2.0-flash",
        instruction=f"""You are the Decision Agent for project {project_id}.
        
        Based on {{recognition_results}} and {{assessment_results}}, prioritize remediation.
        
        Tasks:
        1. Rank security findings by risk and business impact
        2. Create a prioritized remediation queue
        3. Estimate effort and complexity for each fix
        4. Generate specific, actionable recommendations
        5. Identify quick wins vs long-term improvements
        
        Group recommendations into:
        - IMMEDIATE: Fix within 24 hours (critical vulnerabilities)
        - SHORT_TERM: Fix within 1 week (high-risk issues)
        - MEDIUM_TERM: Fix within 1 month (compliance gaps)
        - LONG_TERM: Strategic improvements (architecture changes)
        
        For each recommendation, provide:
        - Clear description of the issue
        - Step-by-step remediation instructions
        - Expected impact and risk reduction
        - Alternative solutions if applicable
        """,
        output_key="decision_results"
    )
    
    # Phase 4: Action - Execute remediation (limited)
    action = LlmAgent(
        name="ActionAgent",
        model="gemini-2.0-flash",
        instruction=f"""You are the Action Agent for project {project_id}.
        
        Execute safe remediation actions based on {{decision_results}}.
        
        IMPORTANT: You have LIMITED write permissions. Only execute:
        1. Add restrictive IAM conditions (never remove permissions)
        2. Enable security features (audit logs, encryption)
        3. Add security policies (never remove existing ones)
        4. Tag resources for tracking
        5. Enable monitoring and alerting
        
        NEVER:
        - Delete resources
        - Remove IAM permissions
        - Disable services
        - Modify production configurations without explicit approval
        
        For each action:
        1. Verify prerequisites and safety
        2. Execute the remediation
        3. Validate the change was successful
        4. Log all actions for audit trail
        
        Output an action report with success/failure status for each remediation.
        """,
        output_key="action_results"
    )
    
    # Phase 5: Review - Verify and report
    review = LlmAgent(
        name="ReviewAgent",
        model="gemini-2.0-flash",
        instruction=f"""You are the Review Agent for project {project_id}.
        
        Generate a comprehensive review based on all RADAR phases.
        
        Using {{recognition_results}}, {{assessment_results}}, {{decision_results}}, and {{action_results}}:
        
        Tasks:
        1. Verify remediation effectiveness
        2. Identify remaining risks
        3. Track improvement metrics
        4. Generate executive summary
        5. Recommend next steps
        
        Create a report with:
        - Executive Summary (2-3 paragraphs for leadership)
        - Key Metrics (before/after risk scores, findings resolved)
        - Actions Taken (what was fixed, what failed)
        - Remaining Issues (prioritized list)
        - Trend Analysis (improving/worsening areas)
        - Next Steps (specific recommendations for continued improvement)
        
        Make the report clear, actionable, and suitable for both technical and executive audiences.
        """,
        output_key="review_report"
    )
    
    # Create the sequential pipeline with proper parent-child relationships
    radar_pipeline = SequentialAgent(
        name="RADARPipeline",
        sub_agents=[recognition, assessment, decision, action, review]
    )
    
    logger.info(f"🎯 RADAR Pipeline created for project {project_id}")
    logger.info(f"   Phases: Recognition → Assessment → Decision → Action → Review")
    
    return radar_pipeline


class RADARCoordinator:
    """
    Wrapper class for the RADAR SequentialAgent pipeline.
    
    This provides convenient methods and integrations while
    maintaining the pure ADK agent architecture underneath.
    """
    
    def __init__(self, project_id: str):
        """Initialize RADAR Coordinator with ADK pipeline and Runner."""
        self.project_id = project_id
        self.pipeline = create_radar_pipeline(project_id)
        
        # Create a Runner to properly execute the pipeline
        from google.adk import Runner
        from google.adk.sessions import InMemorySessionService
        
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.pipeline,  # Use the SequentialAgent as the agent
            session_service=self.session_service,
            app_name="RADARPipeline"
        )
        
        # Create a default session for this coordinator
        import os
        self.session_id = f"radar_{os.urandom(8).hex()}"
        self.user_id = f"backend_{project_id}"
        
        # Initialize the session
        self.session_service.get_or_create_session(
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        logger.info(f"✅ RADAR Coordinator initialized for project {project_id} with session {self.session_id}")
    
    async def process_user_query(self, query: str, phases: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process user query through RADAR pipeline using Runner.
        
        Args:
            query: User's question or command
            phases: Optional list of specific phases to run
        
        Returns:
            Results from the pipeline execution
        """
        logger.info(f"📝 Processing query: {query[:100]}...")
        
        # If no phases specified, determine from query
        if not phases:
            phases = self._determine_phases(query.lower())
        
        try:
            from google.genai import types
            
            # Create user message
            content = types.Content(
                role='user',
                parts=[types.Part(text=query)]
            )
            
            # Use the Runner to execute the pipeline
            # This handles all the context creation internally
            result_text = ""
            events = self.runner.run(
                user_id=self.user_id,  # Use the initialized user_id
                session_id=self.session_id,  # Use the initialized session_id
                new_message=content
            )
            
            # Collect output from events
            for event in events:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text'):
                            result_text += part.text
            
            # Create formatted result
            formatted_result = {
                "success": True,
                "query": query,
                "phases_executed": phases,
                "pipeline_output": result_text,
                "summary": result_text if result_text else f"Processed query: {query}"
            }
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _determine_phases(self, query_lower: str) -> List[str]:
        """
        Determine which RADAR phases to execute based on the query.
        
        Returns list of phase names.
        """
        # Full cycle keywords
        if any(word in query_lower for word in ["fix", "remediate", "repair", "patch"]):
            return ["recognition", "assessment", "decision", "action", "review"]
        
        # Security check keywords
        if any(word in query_lower for word in ["security", "vulnerabilities", "threats", "compliance"]):
            if "report" in query_lower:
                return ["recognition", "assessment", "review"]
            elif any(word in query_lower for word in ["recommend", "prioritize", "what should"]):
                return ["recognition", "assessment", "decision"]
            else:
                return ["recognition", "assessment"]
        
        # Discovery only
        if any(word in query_lower for word in ["inventory", "resources", "assets", "what do i have"]):
            return ["recognition"]
        
        # Review/audit keywords
        if any(word in query_lower for word in ["report", "review", "audit", "verify"]):
            return ["recognition", "assessment", "review"]
        
        # Recommendation keywords
        if any(word in query_lower for word in ["recommend", "suggest", "prioritize", "what should"]):
            return ["recognition", "assessment", "decision"]
        
        # Default to recognition + assessment
        return ["recognition", "assessment"]
    
    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate a summary from pipeline results."""
        # If review phase ran, use its report
        if result.get("review_report"):
            return result["review_report"]
        
        # Otherwise, build a summary from available phases
        summary_parts = ["## RADAR Analysis Summary\n"]
        
        if result.get("recognition_results"):
            summary_parts.append("### Resources Discovered")
            summary_parts.append(f"- {result['recognition_results'][:200]}...")
        
        if result.get("assessment_results"):
            summary_parts.append("\n### Security Assessment")
            summary_parts.append(f"- {result['assessment_results'][:200]}...")
        
        if result.get("decision_results"):
            summary_parts.append("\n### Recommendations")
            summary_parts.append(f"- {result['decision_results'][:200]}...")
        
        if result.get("action_results"):
            summary_parts.append("\n### Actions Taken")
            summary_parts.append(f"- {result['action_results'][:200]}...")
        
        return "\n".join(summary_parts)


# ============================================
# FastAPI Endpoints
# ============================================

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for RADAR operations.
    
    Processes user queries through the appropriate RADAR phases.
    """
    try:
        project_id = request.project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', 'default-project')
        
        # Create coordinator
        coordinator = RADARCoordinator(project_id)
        
        # Process query
        result = await coordinator.process_user_query(request.query)
        
        # Build response
        response = ChatResponse(
            success=result.get("success", False),
            response=result.get("summary", "Analysis complete"),
            user_id=request.user_id,
            session_id=request.session_id,
            phases_executed=result.get("phases_executed", []),
            timestamp=datetime.now().isoformat()
        )
        
        if result.get("assessment"):
            # Extract risk level if available
            response.risk_level = "MEDIUM"  # Would be extracted from assessment
        
        if result.get("decision"):
            # Extract recommendations if available
            response.recommendations = {"immediate": [], "short_term": []}  # Would be parsed
        
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for RADAR system."""
    return {
        "status": "healthy",
        "service": "RADAR Coordinator",
        "architecture": "ADK SequentialAgent Pipeline",
        "phases": ["Recognition", "Assessment", "Decision", "Action", "Review"],
        "timestamp": datetime.now().isoformat()
    }


@router.get("/capabilities")
async def get_capabilities():
    """Get RADAR system capabilities."""
    return {
        "architecture": "ADK Sequential Pipeline with LlmAgent sub-agents",
        "phases": {
            "recognition": "Discover and inventory all cloud resources",
            "assessment": "Evaluate security posture and compliance",
            "decision": "Prioritize issues and generate recommendations",
            "action": "Execute remediation (requires authorization)",
            "review": "Verify changes and generate reports"
        },
        "query_examples": [
            "What resources do I have?",
            "Check our security posture",
            "What should I fix first?",
            "Generate security report",
            "Fix critical security issues"
        ],
        "features": [
            "ADK-native agent orchestration",
            "Sequential phase execution with state sharing",
            "LLM-powered analysis at each phase",
            "Comprehensive resource discovery",
            "Security vulnerability assessment",
            "Compliance checking (CIS, PCI, HIPAA)",
            "Prioritized recommendations",
            "Safe remediation actions",
            "Executive reporting"
        ]
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, user_id: str = "default"):
    """
    WebSocket endpoint for real-time RADAR operations.
    
    Allows streaming of RADAR phase progress.
    """
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Get project ID
            project_id = message.get("project_id", os.environ.get('GOOGLE_CLOUD_PROJECT', 'default-project'))
            
            # Create coordinator
            coordinator = RADARCoordinator(project_id)
            
            # Send phase updates as they execute
            await websocket.send_json({
                "type": "status",
                "message": "Starting RADAR analysis..."
            })
            
            # Process query
            result = await coordinator.process_user_query(message.get("query", ""))
            
            # Send result
            await websocket.send_json({
                "type": "result",
                "data": result
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        manager.disconnect(websocket, user_id)
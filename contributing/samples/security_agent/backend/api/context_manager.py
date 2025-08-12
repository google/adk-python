"""Context-aware API manager for intelligent chat routing and suggestions."""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

class ContextType(Enum):
    """Types of context information."""
    USER_PREFERENCES = "user_preferences"
    CONVERSATION_HISTORY = "conversation_history" 
    CURRENT_PAGE = "current_page"
    WORKFLOW_STATE = "workflow_state"
    DATA_CONTEXT = "data_context"
    TEMPORAL_CONTEXT = "temporal_context"
    ENVIRONMENTAL_CONTEXT = "environmental_context"

class IntentType(Enum):
    """User intent types for better routing."""
    INFORMATION_SEEKING = "information_seeking"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION = "configuration"
    ANALYSIS = "analysis"
    MONITORING = "monitoring"
    COMPLIANCE = "compliance"
    SECURITY_REVIEW = "security_review"

@dataclass
class UserContext:
    """Comprehensive user context information."""
    user_id: str
    current_page: Optional[str] = None
    current_data: Optional[Dict[str, Any]] = None
    workflow_stage: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    expertise_level: str = "intermediate"  # beginner, intermediate, advanced
    role: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.now)
    session_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """Context from conversation history."""
    topics_discussed: Set[str] = field(default_factory=set)
    agents_used: List[str] = field(default_factory=list)
    last_intent: Optional[IntentType] = None
    unresolved_issues: List[str] = field(default_factory=list)
    successful_patterns: List[str] = field(default_factory=list)
    user_satisfaction: Optional[float] = None  # 0-1 score

@dataclass 
class ContextualSuggestion:
    """A contextual suggestion with metadata."""
    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    confidence: float = 0.0
    context_type: ContextType = ContextType.CONVERSATION_HISTORY
    priority: int = 1  # 1-5, 5 being highest
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiry: Optional[datetime] = None

class ContextualRequest(BaseModel):
    """Request model for contextual operations."""
    user_id: str
    query: Optional[str] = None
    current_page: Optional[str] = None
    current_data: Optional[Dict[str, Any]] = None
    workflow_context: Optional[Dict[str, Any]] = None

class ContextualResponse(BaseModel):
    """Response model for contextual operations."""
    success: bool
    suggestions: List[str]
    routing_recommendation: Optional[str] = None
    context_insights: Dict[str, Any]
    quick_actions: List[Dict[str, Any]]

class ContextAwareManager:
    """Manages contextual awareness for intelligent chat routing."""
    
    def __init__(self):
        self.user_contexts: Dict[str, UserContext] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # Pattern matching for intent detection
        self.intent_patterns = {
            IntentType.INFORMATION_SEEKING: [
                r"what is", r"show me", r"tell me about", r"how does", r"explain",
                r"list", r"describe", r"details about"
            ],
            IntentType.TROUBLESHOOTING: [
                r"fix", r"error", r"problem", r"issue", r"not working", r"failed",
                r"troubleshoot", r"debug", r"resolve"
            ],
            IntentType.CONFIGURATION: [
                r"configure", r"setup", r"enable", r"disable", r"change settings",
                r"modify", r"update configuration", r"set up"
            ],
            IntentType.ANALYSIS: [
                r"analyze", r"review", r"assess", r"evaluate", r"examine",
                r"check", r"audit", r"investigate"
            ],
            IntentType.MONITORING: [
                r"monitor", r"watch", r"track", r"observe", r"metrics",
                r"performance", r"status", r"health"
            ],
            IntentType.COMPLIANCE: [
                r"compliance", r"compliant", r"regulation", r"policy", r"standard",
                r"framework", r"audit", r"certification"
            ],
            IntentType.SECURITY_REVIEW: [
                r"security", r"secure", r"vulnerability", r"threat", r"risk",
                r"breach", r"attack", r"protection"
            ]
        }
        
        # Agent capabilities mapping
        self.agent_capabilities = {
            "security_specialist": {
                "expertise": ["security", "vulnerability", "threat", "compliance"],
                "intents": [IntentType.SECURITY_REVIEW, IntentType.ANALYSIS, IntentType.COMPLIANCE],
                "confidence_boost": 0.3
            },
            "iam_specialist": {
                "expertise": ["iam", "permissions", "roles", "access", "identity"],
                "intents": [IntentType.CONFIGURATION, IntentType.ANALYSIS, IntentType.TROUBLESHOOTING],
                "confidence_boost": 0.25
            },
            "storage_specialist": {
                "expertise": ["storage", "bucket", "data", "backup", "lifecycle"],
                "intents": [IntentType.CONFIGURATION, IntentType.MONITORING, IntentType.ANALYSIS],
                "confidence_boost": 0.25
            },
            "monitoring_specialist": {
                "expertise": ["monitoring", "metrics", "performance", "alerts"],
                "intents": [IntentType.MONITORING, IntentType.ANALYSIS, IntentType.TROUBLESHOOTING],
                "confidence_boost": 0.2
            }
        }
        
        # Page-specific context mappings
        self.page_contexts = {
            "dashboard": {
                "primary_actions": ["overview", "status", "alerts"],
                "data_available": ["metrics", "health", "recent_activity"],
                "default_agent": "monitoring_specialist"
            },
            "security": {
                "primary_actions": ["scan", "findings", "remediation"],
                "data_available": ["findings", "scores", "recommendations"], 
                "default_agent": "security_specialist"
            },
            "iam": {
                "primary_actions": ["analyze", "permissions", "roles"],
                "data_available": ["users", "policies", "access_patterns"],
                "default_agent": "iam_specialist"
            },
            "storage": {
                "primary_actions": ["buckets", "lifecycle", "permissions"],
                "data_available": ["buckets", "usage", "costs"],
                "default_agent": "storage_specialist"
            },
            "compliance": {
                "primary_actions": ["assess", "framework", "gaps"],
                "data_available": ["status", "requirements", "evidence"],
                "default_agent": "compliance_specialist"
            }
        }
    
    def update_user_context(
        self, 
        user_id: str, 
        current_page: str = None,
        current_data: Dict[str, Any] = None,
        workflow_context: Dict[str, Any] = None
    ):
        """Update user context information."""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = UserContext(user_id=user_id)
        
        context = self.user_contexts[user_id]
        
        if current_page:
            context.current_page = current_page
        if current_data:
            context.current_data = current_data
        if workflow_context:
            context.session_context.update(workflow_context)
        
        context.last_activity = datetime.now()
    
    def detect_intent(self, query: str) -> Tuple[IntentType, float]:
        """Detect user intent from query text."""
        query_lower = query.lower()
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
            
            if score > 0:
                intent_scores[intent_type] = score / len(patterns)
        
        if not intent_scores:
            return IntentType.INFORMATION_SEEKING, 0.5
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1]
    
    def recommend_agent(
        self, 
        user_id: str, 
        query: str,
        conversation_context: ConversationContext = None
    ) -> Tuple[str, float]:
        """Recommend the best agent based on context."""
        intent, intent_confidence = self.detect_intent(query)
        user_context = self.user_contexts.get(user_id)
        
        agent_scores = {}
        
        # Score agents based on various factors
        for agent_name, capabilities in self.agent_capabilities.items():
            score = 0.0
            
            # Intent matching
            if intent in capabilities["intents"]:
                score += intent_confidence * 0.4
            
            # Keyword matching
            query_lower = query.lower()
            for expertise in capabilities["expertise"]:
                if expertise in query_lower:
                    score += 0.3
            
            # Context from current page
            if user_context and user_context.current_page:
                page_context = self.page_contexts.get(user_context.current_page, {})
                if agent_name == page_context.get("default_agent"):
                    score += 0.2
            
            # Conversation history boost
            if conversation_context and agent_name in conversation_context.agents_used:
                score += 0.1
            
            # Capability boost
            score += capabilities.get("confidence_boost", 0.0)
            
            agent_scores[agent_name] = score
        
        # Return best agent
        if agent_scores:
            best_agent = max(agent_scores.items(), key=lambda x: x[1])
            return best_agent[0], min(1.0, best_agent[1])
        
        return "general_coordinator", 0.5
    
    def generate_contextual_suggestions(
        self, 
        user_id: str,
        conversation_context: ConversationContext = None
    ) -> List[ContextualSuggestion]:
        """Generate contextual suggestions based on current state."""
        suggestions = []
        user_context = self.user_contexts.get(user_id)
        
        if not user_context:
            return suggestions
        
        # Page-specific suggestions
        if user_context.current_page:
            page_context = self.page_contexts.get(user_context.current_page, {})
            
            for action in page_context.get("primary_actions", []):
                suggestion = ContextualSuggestion(
                    text=f"Show me {action} for current context",
                    confidence=0.8,
                    context_type=ContextType.CURRENT_PAGE,
                    priority=4
                )
                suggestions.append(suggestion)
        
        # Data-driven suggestions
        if user_context.current_data:
            data_keys = list(user_context.current_data.keys())
            
            if "project_id" in data_keys:
                suggestion = ContextualSuggestion(
                    text=f"Analyze security for {user_context.current_data['project_id']}",
                    confidence=0.9,
                    context_type=ContextType.DATA_CONTEXT,
                    priority=5
                )
                suggestions.append(suggestion)
        
        # Conversation-driven suggestions
        if conversation_context:
            if "security" in conversation_context.topics_discussed:
                suggestion = ContextualSuggestion(
                    text="Continue security analysis with detailed findings",
                    confidence=0.7,
                    context_type=ContextType.CONVERSATION_HISTORY,
                    priority=3
                )
                suggestions.append(suggestion)
            
            # Suggest resolution for unresolved issues
            for issue in conversation_context.unresolved_issues:
                suggestion = ContextualSuggestion(
                    text=f"Get help resolving: {issue}",
                    confidence=0.8,
                    context_type=ContextType.CONVERSATION_HISTORY,
                    priority=4
                )
                suggestions.append(suggestion)
        
        # Expertise-level appropriate suggestions
        expertise_level = user_context.preferences.get("expertise_level", "intermediate")
        
        if expertise_level == "beginner":
            suggestion = ContextualSuggestion(
                text="Show me a guided tour of available features",
                confidence=0.6,
                context_type=ContextType.USER_PREFERENCES,
                priority=2
            )
            suggestions.append(suggestion)
        elif expertise_level == "advanced":
            suggestion = ContextualSuggestion(
                text="Show me advanced configuration options",
                confidence=0.7,
                context_type=ContextType.USER_PREFERENCES,
                priority=3
            )
            suggestions.append(suggestion)
        
        # Sort by priority and confidence
        suggestions.sort(key=lambda s: (s.priority, s.confidence), reverse=True)
        return suggestions[:5]  # Return top 5
    
    def generate_quick_actions(
        self, 
        user_id: str,
        current_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate quick action buttons based on context."""
        user_context = self.user_contexts.get(user_id)
        quick_actions = []
        
        if not user_context:
            return quick_actions
        
        # Page-specific quick actions
        if user_context.current_page == "dashboard":
            quick_actions.extend([
                {"action": "security_scan", "label": "Run Security Scan", "icon": "shield"},
                {"action": "performance_check", "label": "Check Performance", "icon": "activity"},
                {"action": "recent_alerts", "label": "View Alerts", "icon": "bell"}
            ])
        
        elif user_context.current_page == "security":
            quick_actions.extend([
                {"action": "detailed_findings", "label": "Show Details", "icon": "search"},
                {"action": "fix_recommendations", "label": "Fix Issues", "icon": "tool"},
                {"action": "compliance_check", "label": "Check Compliance", "icon": "check-circle"}
            ])
        
        elif user_context.current_page == "iam":
            quick_actions.extend([
                {"action": "user_analysis", "label": "Analyze Users", "icon": "users"},
                {"action": "permission_review", "label": "Review Permissions", "icon": "key"},
                {"action": "role_optimization", "label": "Optimize Roles", "icon": "settings"}
            ])
        
        # Data-driven quick actions
        if user_context.current_data:
            if "project_id" in user_context.current_data:
                quick_actions.append({
                    "action": "project_overview",
                    "label": f"Overview of {user_context.current_data['project_id'][:20]}...",
                    "icon": "folder"
                })
        
        return quick_actions[:6]  # Limit to 6 actions
    
    def get_context_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about the current context."""
        user_context = self.user_contexts.get(user_id)
        conversation_context = self.conversation_contexts.get(user_id)
        
        insights = {
            "user_active_time": 0,
            "current_focus": "general",
            "expertise_level": "intermediate",
            "conversation_stage": "beginning",
            "recommended_next_steps": []
        }
        
        if user_context:
            # Calculate active time
            time_diff = datetime.now() - user_context.last_activity
            insights["user_active_time"] = int(time_diff.total_seconds() / 60)
            
            # Determine current focus
            if user_context.current_page:
                insights["current_focus"] = user_context.current_page
            
            # Get expertise level
            insights["expertise_level"] = user_context.preferences.get("expertise_level", "intermediate")
        
        if conversation_context:
            # Determine conversation stage
            agent_count = len(conversation_context.agents_used)
            topic_count = len(conversation_context.topics_discussed)
            
            if agent_count == 0:
                insights["conversation_stage"] = "beginning"
            elif agent_count <= 2 and topic_count <= 1:
                insights["conversation_stage"] = "focused"
            else:
                insights["conversation_stage"] = "exploratory"
            
            # Suggest next steps based on patterns
            if conversation_context.unresolved_issues:
                insights["recommended_next_steps"].append("Resolve outstanding issues")
            
            if "security" in conversation_context.topics_discussed:
                insights["recommended_next_steps"].append("Review security recommendations")
        
        return insights

# Global context manager instance
context_manager = ContextAwareManager()

# API Endpoints
@router.post("/analyze")
async def analyze_context(request: ContextualRequest) -> ContextualResponse:
    """Analyze context and provide recommendations."""
    try:
        # Update user context
        context_manager.update_user_context(
            request.user_id,
            request.current_page,
            request.current_data,
            request.workflow_context
        )
        
        # Get conversation context
        conversation_context = context_manager.conversation_contexts.get(request.user_id)
        
        # Recommend agent if query provided
        routing_recommendation = None
        if request.query:
            agent, confidence = context_manager.recommend_agent(
                request.user_id, 
                request.query,
                conversation_context
            )
            routing_recommendation = {
                "agent": agent,
                "confidence": confidence
            }
        
        # Generate suggestions
        contextual_suggestions = context_manager.generate_contextual_suggestions(
            request.user_id,
            conversation_context
        )
        suggestions = [s.text for s in contextual_suggestions]
        
        # Generate quick actions
        quick_actions = context_manager.generate_quick_actions(
            request.user_id,
            request.current_data
        )
        
        # Get insights
        context_insights = context_manager.get_context_insights(request.user_id)
        
        return ContextualResponse(
            success=True,
            suggestions=suggestions,
            routing_recommendation=routing_recommendation,
            context_insights=context_insights,
            quick_actions=quick_actions
        )
        
    except Exception as e:
        logger.error(f"Error analyzing context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze context: {str(e)}")

@router.put("/user/{user_id}")
async def update_user_context(
    user_id: str,
    context_data: Dict[str, Any]
):
    """Update user context information."""
    try:
        context_manager.update_user_context(
            user_id,
            context_data.get("current_page"),
            context_data.get("current_data"),
            context_data.get("workflow_context")
        )
        
        return {
            "success": True,
            "message": "User context updated successfully",
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update context: {str(e)}")

@router.get("/user/{user_id}")
async def get_user_context(user_id: str):
    """Get current user context."""
    try:
        user_context = context_manager.user_contexts.get(user_id)
        
        if not user_context:
            return {"success": False, "message": "User context not found"}
        
        return {
            "success": True,
            "context": {
                "user_id": user_context.user_id,
                "current_page": user_context.current_page,
                "current_data": user_context.current_data,
                "expertise_level": user_context.expertise_level,
                "last_activity": user_context.last_activity.isoformat(),
                "session_context": user_context.session_context
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user context: {str(e)}")

@router.post("/suggestions/{user_id}")
async def get_contextual_suggestions(user_id: str):
    """Get contextual suggestions for a user."""
    try:
        conversation_context = context_manager.conversation_contexts.get(user_id)
        suggestions = context_manager.generate_contextual_suggestions(user_id, conversation_context)
        
        return {
            "success": True,
            "suggestions": [
                {
                    "text": s.text,
                    "confidence": s.confidence,
                    "priority": s.priority,
                    "context_type": s.context_type.value
                }
                for s in suggestions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@router.post("/quick-actions/{user_id}")
async def get_quick_actions(user_id: str, context_data: Dict[str, Any] = None):
    """Get contextual quick actions for a user."""
    try:
        quick_actions = context_manager.generate_quick_actions(user_id, context_data)
        
        return {
            "success": True,
            "quick_actions": quick_actions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quick actions: {str(e)}")
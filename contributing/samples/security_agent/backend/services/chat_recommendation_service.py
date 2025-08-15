"""
Chat Recommendation Service

Integrates Google Cloud Recommender API with the chat interface to provide
contextual, conversational recommendation experiences.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.recommender_models import (
    RecommendationInsight,
    RecommenderType,
    Priority,
    RecommendationState,
    ChatRecommendationContext,
    ChatRecommendationQuery,
    ChatRecommendationResponse,
    RecommenderContextRequest,
    SessionRecommendationTracking,
    RecommendationProgress
)
from ..services.recommender_service import (
    RecommenderService,
    RecommendationContext
)
from ..chat_manager import EnhancedChatManager, ChatMessage, MessageType

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Chat query intents for recommendations."""
    LIST_RECOMMENDATIONS = "list_recommendations"
    ANALYZE_RECOMMENDATION = "analyze_recommendation"
    APPLY_RECOMMENDATION = "apply_recommendation"
    DISMISS_RECOMMENDATION = "dismiss_recommendation"
    EXPLAIN_RECOMMENDATION = "explain_recommendation"
    PRIORITIZE_RECOMMENDATIONS = "prioritize_recommendations"
    TRACK_PROGRESS = "track_progress"
    GENERAL_SECURITY = "general_security"
    COST_OPTIMIZATION = "cost_optimization"
    COMPLIANCE_STATUS = "compliance_status"

@dataclass
class ConversationState:
    """State management for recommendation conversations."""
    session_id: str
    user_id: str
    project_id: str
    current_recommendations: List[RecommendationInsight]
    active_recommendation_id: Optional[str] = None
    conversation_context: Dict[str, Any] = None
    user_preferences: Dict[str, Any] = None
    last_intent: Optional[QueryIntent] = None
    pending_actions: List[str] = None
    
    def __post_init__(self):
        if self.conversation_context is None:
            self.conversation_context = {}
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.pending_actions is None:
            self.pending_actions = []

class ChatRecommendationService:
    """Service for chat-driven recommendation interactions with LLM agent integration."""
    
    def __init__(self, recommender_service: RecommenderService, chat_manager):
        """Initialize the chat recommendation service.
        
        Args:
            recommender_service: The core recommender service
            chat_manager: The enhanced chat manager
        """
        self.recommender_service = recommender_service
        self.chat_manager = chat_manager
        
        # Natural language processing components
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.response_generator = ResponseGenerator()
        
        # State management with session persistence
        self.conversation_states: Dict[str, ConversationState] = {}
        self.session_tracking: Dict[str, SessionRecommendationTracking] = {}
        
        # Contextual suggestions and learning
        self.suggestion_engine = SuggestionEngine()
        
        # Performance metrics
        self.performance_metrics = {
            "queries_processed": 0,
            "successful_applications": 0,
            "avg_response_time": 0.0,
            "user_satisfaction_score": 0.0,
            "last_optimization": datetime.now()
        }
        
        # Agent integration for LLM routing
        self.agent_integration_enabled = True
        
        logger.info("✅ ChatRecommendationService initialized with enhanced LLM integration")
    
    async def process_query(self, query: ChatRecommendationQuery) -> ChatRecommendationResponse:
        """Process a natural language query about recommendations with enhanced LLM integration.
        
        Args:
            query: The chat recommendation query
            
        Returns:
            Comprehensive response with recommendations and actions
        """
        start_time = time.time()
        request_id = f"rec_query_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"🔍 [REC-{request_id}] Processing recommendation query: '{query.query[:100]}...'")
            
            # Update performance metrics
            self.performance_metrics["queries_processed"] += 1
            
            # Initialize or update conversation state
            state = await self._get_or_create_conversation_state(query)
            
            # Enhanced intent classification with context
            intent = await self.intent_classifier.classify(query.query, state)
            entities = await self.entity_extractor.extract(query.query, state)
            
            logger.info(f"🎯 [REC-{request_id}] Intent: {intent.value}, Entities: {list(entities.keys())}")
            
            # Update query with detected intent and entities
            query.intent = intent.value
            query.entities = entities
            
            # Process based on intent with enhanced error handling
            response = await self._process_by_intent(query, state, intent, entities)
            
            # Enhance response with contextual information
            response = await self._enhance_response_with_context(response, state)
            
            # Update conversation state and tracking
            await self._update_conversation_state(state, query, response)
            await self._update_session_tracking(query.context.session_id, query, response)
            
            # Add to chat history with rich metadata
            await self._add_to_chat_history(query, response, request_id)
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, True)
            
            logger.info(f"✅ [REC-{request_id}] Query processed successfully in {response_time:.2f}s")
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, False)
            
            logger.error(f"❌ [REC-{request_id}] Error processing recommendation query: {e}")
            
            return ChatRecommendationResponse(
                success=False,
                response_text=f"I encountered an error processing your request: {str(e)}. Please try again or rephrase your question.",
                recommendations=[],
                suggested_actions=[
                    "Try rephrasing your question", 
                    "Check your project permissions",
                    "Ask for help with a specific recommendation type"
                ],
                follow_up_questions=[
                    "Would you like me to show all available recommendations?",
                    "Should I focus on security or cost recommendations?"
                ],
                context_updates={"last_error": str(e), "timestamp": datetime.now().isoformat()}
            )
    
    async def _get_or_create_conversation_state(
        self, 
        query: ChatRecommendationQuery
    ) -> ConversationState:
        """Get or create conversation state for the session."""
        session_id = query.context.session_id
        
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = ConversationState(
                session_id=session_id,
                user_id=query.context.user_id,
                project_id=query.context.project_context.project_id,
                current_recommendations=[],
                conversation_context=query.context.user_preferences.copy()
            )
        
        return self.conversation_states[session_id]
    
    async def _process_by_intent(
        self,
        query: ChatRecommendationQuery,
        state: ConversationState,
        intent: QueryIntent,
        entities: Dict[str, Any]
    ) -> ChatRecommendationResponse:
        """Process query based on detected intent."""
        
        if intent == QueryIntent.LIST_RECOMMENDATIONS:
            return await self._handle_list_recommendations(query, state, entities)
        elif intent == QueryIntent.ANALYZE_RECOMMENDATION:
            return await self._handle_analyze_recommendation(query, state, entities)
        elif intent == QueryIntent.APPLY_RECOMMENDATION:
            return await self._handle_apply_recommendation(query, state, entities)
        elif intent == QueryIntent.DISMISS_RECOMMENDATION:
            return await self._handle_dismiss_recommendation(query, state, entities)
        elif intent == QueryIntent.EXPLAIN_RECOMMENDATION:
            return await self._handle_explain_recommendation(query, state, entities)
        elif intent == QueryIntent.PRIORITIZE_RECOMMENDATIONS:
            return await self._handle_prioritize_recommendations(query, state, entities)
        elif intent == QueryIntent.TRACK_PROGRESS:
            return await self._handle_track_progress(query, state, entities)
        elif intent == QueryIntent.GENERAL_SECURITY:
            return await self._handle_general_security(query, state, entities)
        elif intent == QueryIntent.COST_OPTIMIZATION:
            return await self._handle_cost_optimization(query, state, entities)
        elif intent == QueryIntent.COMPLIANCE_STATUS:
            return await self._handle_compliance_status(query, state, entities)
        else:
            return await self._handle_general_query(query, state, entities)
    
    async def _handle_list_recommendations(
        self,
        query: ChatRecommendationQuery,
        state: ConversationState,
        entities: Dict[str, Any]
    ) -> ChatRecommendationResponse:
        """Handle requests to list recommendations."""
        
        # Build context for recommender service
        context = RecommendationContext(
            project_id=state.project_id,
            resource_name=entities.get("resource", ""),
            location=entities.get("location", "global"),
            filters=entities.get("filters", {}),
            user_preferences=state.user_preferences
        )
        
        # Get recommendations
        recommendations = await self.recommender_service.get_all_recommendations(context)
        
        # Apply filters from entities
        filtered_recommendations = await self._apply_chat_filters(
            recommendations, 
            entities
        )
        
        # Update state
        state.current_recommendations = filtered_recommendations
        
        # Generate response
        response_text = await self.response_generator.generate_list_response(
            filtered_recommendations,
            entities.get("priority_filter"),
            entities.get("type_filter")
        )
        
        suggested_actions = [
            f"Analyze recommendation: {rec.name}" 
            for rec in filtered_recommendations[:3]
        ]
        
        follow_up_questions = [
            "Would you like me to prioritize these recommendations?",
            "Should I focus on security or cost optimization recommendations?",
            "Do you want to see the implementation steps for any of these?"
        ]
        
        return ChatRecommendationResponse(
            success=True,
            response_text=response_text,
            recommendations=filtered_recommendations,
            suggested_actions=suggested_actions,
            follow_up_questions=follow_up_questions,
            context_updates={"last_list_count": len(filtered_recommendations)}
        )
    
    async def _handle_analyze_recommendation(
        self,
        query: ChatRecommendationQuery,
        state: ConversationState,
        entities: Dict[str, Any]
    ) -> ChatRecommendationResponse:
        """Handle requests to analyze a specific recommendation."""
        
        # Find the recommendation to analyze
        recommendation = await self._find_recommendation(
            entities.get("recommendation_id") or entities.get("recommendation_name"),
            state
        )
        
        if not recommendation:
            return ChatRecommendationResponse(
                success=False,
                response_text="I couldn't find the recommendation you're referring to. Please specify the recommendation name or ID.",
                recommendations=[],
                suggested_actions=["List current recommendations", "Try a different recommendation name"],
                follow_up_questions=["Would you like me to show all available recommendations?"]
            )
        
        # Update active recommendation
        state.active_recommendation_id = recommendation.recommendation_id
        
        # Generate detailed analysis
        response_text = await self.response_generator.generate_analysis_response(recommendation)
        
        suggested_actions = [
            f"Apply this recommendation",
            f"Show implementation steps",
            f"Calculate cost impact",
            f"Check compliance implications"
        ]
        
        follow_up_questions = [
            "Would you like to see the step-by-step implementation plan?",
            "Should I show you the potential security impact?",
            "Do you want to apply this recommendation now?"
        ]
        
        return ChatRecommendationResponse(
            success=True,
            response_text=response_text,
            recommendations=[recommendation],
            suggested_actions=suggested_actions,
            follow_up_questions=follow_up_questions,
            context_updates={"active_recommendation": recommendation.recommendation_id}
        )
    
    async def _handle_apply_recommendation(
        self,
        query: ChatRecommendationQuery,
        state: ConversationState,
        entities: Dict[str, Any]
    ) -> ChatRecommendationResponse:
        """Handle requests to apply a recommendation."""
        
        # Find the recommendation to apply
        recommendation_id = (
            entities.get("recommendation_id") or 
            state.active_recommendation_id
        )
        
        if not recommendation_id:
            return ChatRecommendationResponse(
                success=False,
                response_text="Please specify which recommendation you'd like to apply.",
                recommendations=[],
                suggested_actions=["List current recommendations"],
                follow_up_questions=["Which recommendation would you like to apply?"]
            )
        
        # Check if it's a dry run request
        dry_run = entities.get("dry_run", True)
        
        # Apply the recommendation
        context = RecommendationContext(
            project_id=state.project_id,
            resource_name="",
            location="global"
        )
        
        result = await self.recommender_service.apply_recommendation(
            recommendation_id,
            context,
            dry_run=dry_run
        )
        
        if result["success"]:
            if dry_run:
                response_text = await self.response_generator.generate_dry_run_response(result)
                suggested_actions = ["Apply for real", "Show implementation steps"]
            else:
                response_text = await self.response_generator.generate_apply_response(result)
                suggested_actions = ["Verify implementation", "Track progress"]
        else:
            response_text = f"Failed to apply recommendation: {result.get('error', 'Unknown error')}"
            suggested_actions = ["Retry application", "Check permissions"]
        
        return ChatRecommendationResponse(
            success=result["success"],
            response_text=response_text,
            recommendations=[],
            suggested_actions=suggested_actions,
            follow_up_questions=[],
            executable_commands=result.get("commands", [])
        )
    
    async def _handle_prioritize_recommendations(
        self,
        query: ChatRecommendationQuery,
        state: ConversationState,
        entities: Dict[str, Any]
    ) -> ChatRecommendationResponse:
        """Handle requests to prioritize recommendations."""
        
        recommendations = state.current_recommendations
        if not recommendations:
            # Get fresh recommendations
            context = RecommendationContext(
                project_id=state.project_id,
                resource_name="",
                location="global"
            )
            recommendations = await self.recommender_service.get_all_recommendations(context)
            state.current_recommendations = recommendations
        
        # Sort by priority and impact
        prioritized = sorted(
            recommendations,
            key=lambda r: (
                self._priority_weight(r.priority),
                -r.security_impact_score,
                -r.cost_savings_usd
            )
        )
        
        response_text = await self.response_generator.generate_prioritization_response(prioritized)
        
        top_recommendations = prioritized[:5]
        
        suggested_actions = [
            f"Analyze: {rec.name}" for rec in top_recommendations[:3]
        ]
        
        follow_up_questions = [
            "Should I focus on the highest priority items first?",
            "Would you like to see the implementation timeline?",
            "Do you want to group these by category?"
        ]
        
        return ChatRecommendationResponse(
            success=True,
            response_text=response_text,
            recommendations=top_recommendations,
            suggested_actions=suggested_actions,
            follow_up_questions=follow_up_questions
        )
    
    async def _handle_track_progress(
        self,
        query: ChatRecommendationQuery,
        state: ConversationState,
        entities: Dict[str, Any]
    ) -> ChatRecommendationResponse:
        """Handle requests to track recommendation progress."""
        
        # Get session tracking
        session_tracking = self.session_tracking.get(query.context.session_id)
        
        if not session_tracking:
            return ChatRecommendationResponse(
                success=True,
                response_text="No recommendations have been applied in this session yet.",
                recommendations=[],
                suggested_actions=["List available recommendations", "Apply a recommendation"],
                follow_up_questions=["Would you like to see available recommendations?"]
            )
        
        response_text = await self.response_generator.generate_progress_response(session_tracking)
        
        return ChatRecommendationResponse(
            success=True,
            response_text=response_text,
            recommendations=[],
            suggested_actions=["Continue with next recommendation", "Verify completed items"],
            follow_up_questions=["Would you like to apply another recommendation?"]
        )
    
    async def _handle_general_security(
        self,
        query: ChatRecommendationQuery,
        state: ConversationState,
        entities: Dict[str, Any]
    ) -> ChatRecommendationResponse:
        """Handle general security queries."""
        
        # Focus on security-related recommendations
        context = RecommendationContext(
            project_id=state.project_id,
            resource_name="",
            location="global",
            filters={"category": "SECURITY"}
        )
        
        recommendations = await self.recommender_service.get_all_recommendations(context)
        
        # Filter for security-focused recommenders
        security_recommendations = [
            rec for rec in recommendations
            if rec.recommender_type in [
                RecommenderType.IAM_POLICY,
                RecommenderType.FIREWALL,
                RecommenderType.SERVICE_ACCOUNT
            ]
        ]
        
        response_text = await self.response_generator.generate_security_response(
            security_recommendations
        )
        
        return ChatRecommendationResponse(
            success=True,
            response_text=response_text,
            recommendations=security_recommendations,
            suggested_actions=[
                "Review IAM recommendations",
                "Check firewall settings",
                "Audit service accounts"
            ],
            follow_up_questions=[
                "Would you like me to focus on IAM or firewall recommendations?",
                "Should I prioritize by risk level?"
            ]
        )
    
    async def _handle_cost_optimization(
        self,
        query: ChatRecommendationQuery,
        state: ConversationState,
        entities: Dict[str, Any]
    ) -> ChatRecommendationResponse:
        """Handle cost optimization queries."""
        
        context = RecommendationContext(
            project_id=state.project_id,
            resource_name="",
            location="global",
            filters={"category": "COST"}
        )
        
        recommendations = await self.recommender_service.get_all_recommendations(context)
        
        # Filter and sort by cost savings
        cost_recommendations = [
            rec for rec in recommendations
            if rec.cost_savings_usd > 0
        ]
        cost_recommendations.sort(key=lambda r: -r.cost_savings_usd)
        
        response_text = await self.response_generator.generate_cost_response(
            cost_recommendations
        )
        
        return ChatRecommendationResponse(
            success=True,
            response_text=response_text,
            recommendations=cost_recommendations,
            suggested_actions=[
                "Apply highest savings recommendation",
                "Show implementation timeline",
                "Calculate total potential savings"
            ],
            follow_up_questions=[
                "Would you like to see the ROI analysis?",
                "Should I prioritize by implementation effort?"
            ]
        )
    
    async def _apply_chat_filters(
        self,
        recommendations: List[RecommendationInsight],
        entities: Dict[str, Any]
    ) -> List[RecommendationInsight]:
        """Apply filters extracted from natural language."""
        
        filtered = recommendations
        
        # Priority filter
        if "priority_filter" in entities:
            priorities = entities["priority_filter"]
            if isinstance(priorities, str):
                priorities = [priorities]
            filtered = [r for r in filtered if r.priority.value in priorities]
        
        # Type filter
        if "type_filter" in entities:
            types = entities["type_filter"]
            if isinstance(types, str):
                types = [types]
            filtered = [r for r in filtered if any(t in r.recommender_type.value.lower() for t in types)]
        
        # Cost threshold filter
        if "min_cost_savings" in entities:
            min_cost = float(entities["min_cost_savings"])
            filtered = [r for r in filtered if r.cost_savings_usd >= min_cost]
        
        # Resource filter
        if "resource_filter" in entities:
            resource_terms = entities["resource_filter"]
            if isinstance(resource_terms, str):
                resource_terms = [resource_terms]
            filtered = [
                r for r in filtered
                if any(term.lower() in resource.lower() 
                      for resource in r.target_resources 
                      for term in resource_terms)
            ]
        
        return filtered
    
    async def _find_recommendation(
        self,
        identifier: Optional[str],
        state: ConversationState
    ) -> Optional[RecommendationInsight]:
        """Find a recommendation by ID or name."""
        
        if not identifier:
            return None
        
        identifier_lower = identifier.lower()
        
        # Search in current recommendations
        for rec in state.current_recommendations:
            if (rec.recommendation_id.lower() == identifier_lower or
                identifier_lower in rec.name.lower()):
                return rec
        
        return None
    
    async def _update_conversation_state(
        self,
        state: ConversationState,
        query: ChatRecommendationQuery,
        response: ChatRecommendationResponse
    ):
        """Update conversation state after processing."""
        
        state.last_intent = QueryIntent(query.intent) if query.intent else None
        
        if response.context_updates:
            state.conversation_context.update(response.context_updates)
        
        # Update pending actions based on suggested actions
        if response.suggested_actions:
            state.pending_actions = response.suggested_actions[:3]  # Keep top 3
    
    async def _update_session_tracking(
        self,
        session_id: str,
        query: ChatRecommendationQuery,
        response: ChatRecommendationResponse
    ):
        """Update session-level recommendation tracking."""
        
        if session_id not in self.session_tracking:
            from ..models.recommender_models import RecommendationAnalytics
            self.session_tracking[session_id] = SessionRecommendationTracking(
                session_id=session_id,
                user_id=query.context.user_id,
                session_analytics=RecommendationAnalytics()
            )
        
        tracking = self.session_tracking[session_id]
        
        # Track discussed recommendations
        for rec in response.recommendations:
            if rec.recommendation_id not in tracking.recommendations_discussed:
                tracking.recommendations_discussed.append(rec.recommendation_id)
        
        tracking.last_updated = datetime.now()
    
    async def _add_to_chat_history(
        self,
        query: ChatRecommendationQuery,
        response: ChatRecommendationResponse,
        request_id: str
    ):
        """Add interaction to chat history with enhanced metadata."""
        
        try:
            # Add user query with metadata
            await self.chat_manager.add_message(
                session_id=query.context.session_id,
                content=query.query,
                sender_type="user",
                metadata={
                    "intent": query.intent,
                    "entities": query.entities,
                    "request_id": request_id,
                    "recommendation_context": True
                }
            )
            
            # Add assistant response with comprehensive metadata
            await self.chat_manager.add_message(
                session_id=query.context.session_id,
                content=response.response_text,
                sender_type="assistant",
                agent_used="recommendation_agent",
                metadata={
                    "recommendations_count": len(response.recommendations),
                    "suggestions_count": len(response.suggested_actions),
                    "success": response.success,
                    "request_id": request_id,
                    "context_updates": response.context_updates
                },
                performance_data={
                    "recommendations_processed": len(response.recommendations),
                    "agent_type": "recommendation_specialist"
                }
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to add to chat history: {e}")
    
    def _priority_weight(self, priority: Priority) -> int:
        """Get numeric weight for priority sorting."""
        weights = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }
        return weights.get(priority, 4)
    
    def _update_performance_metrics(self, response_time: float, success: bool):
        """Update service performance metrics."""
        current_avg = self.performance_metrics["avg_response_time"]
        total_queries = self.performance_metrics["queries_processed"]
        
        # Update average response time
        self.performance_metrics["avg_response_time"] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
        
        if success:
            self.performance_metrics["successful_applications"] += 1
    
    async def _enhance_response_with_context(
        self, 
        response: ChatRecommendationResponse, 
        state: ConversationState
    ) -> ChatRecommendationResponse:
        """Enhance response with conversational context."""
        
        # Add contextual follow-up questions based on conversation history
        if state.last_intent == QueryIntent.LIST_RECOMMENDATIONS:
            response.follow_up_questions.extend([
                "Would you like me to prioritize these by security impact?",
                "Should I show you the cost savings potential?"
            ])
        
        # Add user preference-based suggestions
        if state.user_preferences.get("focus_area") == "security":
            security_actions = [action for action in response.suggested_actions 
                             if "security" in action.lower()]
            if security_actions:
                response.suggested_actions = security_actions + response.suggested_actions
        
        # Add conversation continuity
        if state.active_recommendation_id:
            response.context_updates["active_recommendation"] = state.active_recommendation_id
        
        return response
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        return {
            "performance": self.performance_metrics.copy(),
            "cache_stats": getattr(self.recommender_service, 'cache_stats', {}),
            "active_sessions": len(self.conversation_states),
            "tracked_sessions": len(self.session_tracking),
            "health_status": "healthy" if self.performance_metrics["queries_processed"] > 0 else "idle"
        }

class IntentClassifier:
    """Classifies user intents from natural language queries."""
    
    def __init__(self):
        """Initialize with intent patterns."""
        self.intent_patterns = {
            QueryIntent.LIST_RECOMMENDATIONS: [
                r'\b(show|list|get|what are|display)\b.*\brecommendation',
                r'\bwhat.*recommend',
                r'\brecommendations?\b',
                r'\bsuggest',
            ],
            QueryIntent.ANALYZE_RECOMMENDATION: [
                r'\b(analyze|examine|review|detail|explain)\b.*\brecommendation',
                r'\btell me (more )?about\b.*\brecommendation',
                r'\bwhat does.*recommendation\b',
            ],
            QueryIntent.APPLY_RECOMMENDATION: [
                r'\b(apply|implement|execute|do|run)\b.*\brecommendation',
                r'\bfix\b.*\bissue',
                r'\bmake.*change',
            ],
            QueryIntent.DISMISS_RECOMMENDATION: [
                r'\b(dismiss|ignore|skip|remove)\b.*\brecommendation',
                r'\bnot interested\b',
                r'\bdon\'t want\b',
            ],
            QueryIntent.PRIORITIZE_RECOMMENDATIONS: [
                r'\b(prioritize|rank|order|sort)\b.*\brecommendation',
                r'\bwhich.*first',
                r'\bmost important\b',
                r'\bhighest priority\b',
            ],
            QueryIntent.TRACK_PROGRESS: [
                r'\b(progress|status|how.*doing)\b',
                r'\bwhat.*completed\b',
                r'\btrack\b',
            ],
            QueryIntent.GENERAL_SECURITY: [
                r'\bsecurity\b',
                r'\bvulnerab',
                r'\brisk',
                r'\bthreat',
            ],
            QueryIntent.COST_OPTIMIZATION: [
                r'\bcost',
                r'\bsaving',
                r'\boptimiz',
                r'\bexpensive\b',
                r'\bmoney\b',
            ],
            QueryIntent.COMPLIANCE_STATUS: [
                r'\bcompliance\b',
                r'\baudit\b',
                r'\bregulation\b',
                r'\bstandard\b',
            ],
        }
    
    async def classify(self, query: str, state: ConversationState) -> QueryIntent:
        """Classify the intent of a query."""
        
        query_lower = query.lower()
        
        # Check for explicit patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Context-based classification
        if state.active_recommendation_id and any(word in query_lower for word in ["it", "this", "that"]):
            return QueryIntent.ANALYZE_RECOMMENDATION
        
        # Default to general listing
        return QueryIntent.LIST_RECOMMENDATIONS

class EntityExtractor:
    """Extracts entities and parameters from natural language queries."""
    
    def __init__(self):
        """Initialize with entity patterns."""
        self.priority_patterns = {
            r'\b(critical|urgent|high priority)\b': [Priority.CRITICAL.value],
            r'\bhigh\b': [Priority.HIGH.value],
            r'\bmedium\b': [Priority.MEDIUM.value],
            r'\blow\b': [Priority.LOW.value],
        }
        
        self.type_patterns = {
            r'\biam\b': ["iam"],
            r'\bfirewall\b': ["firewall"],
            r'\bmachine.type\b': ["machine", "type"],
            r'\bservice.account\b': ["service", "account"],
            r'\bsql\b': ["sql"],
            r'\bdisk\b': ["disk"],
        }
    
    async def extract(self, query: str, state: ConversationState) -> Dict[str, Any]:
        """Extract entities from query."""
        
        query_lower = query.lower()
        entities = {}
        
        # Extract priorities
        priorities = []
        for pattern, priority_list in self.priority_patterns.items():
            if re.search(pattern, query_lower):
                priorities.extend(priority_list)
        if priorities:
            entities["priority_filter"] = priorities
        
        # Extract types
        types = []
        for pattern, type_list in self.type_patterns.items():
            if re.search(pattern, query_lower):
                types.extend(type_list)
        if types:
            entities["type_filter"] = types
        
        # Extract cost thresholds
        cost_match = re.search(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', query)
        if cost_match:
            cost_str = cost_match.group(1).replace(',', '')
            entities["min_cost_savings"] = cost_str
        
        # Extract dry run indicators
        if any(phrase in query_lower for phrase in ["dry run", "test", "simulate", "preview"]):
            entities["dry_run"] = True
        elif any(phrase in query_lower for phrase in ["for real", "actually", "live"]):
            entities["dry_run"] = False
        
        # Extract recommendation IDs or names from context
        if state.active_recommendation_id and any(word in query_lower for word in ["it", "this", "that"]):
            entities["recommendation_id"] = state.active_recommendation_id
        
        return entities

class ResponseGenerator:
    """Generates natural language responses for recommendation interactions."""
    
    async def generate_list_response(
        self,
        recommendations: List[RecommendationInsight],
        priority_filter: Optional[List[str]] = None,
        type_filter: Optional[List[str]] = None
    ) -> str:
        """Generate response for listing recommendations."""
        
        if not recommendations:
            return "I didn't find any recommendations matching your criteria. This could mean your environment is well-configured, or we might need to check different parameters."
        
        total_savings = sum(rec.cost_savings_usd for rec in recommendations)
        high_priority_count = len([r for r in recommendations if r.priority in [Priority.CRITICAL, Priority.HIGH]])
        
        response = f"I found {len(recommendations)} recommendations for your project"
        
        if priority_filter:
            response += f" (filtered by priority: {', '.join(priority_filter)})"
        if type_filter:
            response += f" (filtered by type: {', '.join(type_filter)})"
        
        response += ":\n\n"
        
        # Summary statistics
        if total_savings > 0:
            response += f"💰 **Total potential savings**: ${total_savings:,.2f}/month\n"
        
        if high_priority_count > 0:
            response += f"🚨 **High priority items**: {high_priority_count}\n"
        
        response += "\n**Top recommendations:**\n"
        
        # List top 5 recommendations with enhanced formatting
        for i, rec in enumerate(recommendations[:5], 1):
            priority_emoji = self._get_priority_emoji(rec.priority)
            cost_text = f" (💰 ${rec.cost_savings_usd:,.2f}/month savings)" if rec.cost_savings_usd > 0 else ""
            effort_text = f" | 🔧 {rec.implementation_effort} effort" if hasattr(rec, 'implementation_effort') else ""
            
            response += f"{i}. {priority_emoji} **{rec.name}**{cost_text}{effort_text}\n"
            response += f"   {rec.description[:100]}{'...' if len(rec.description) > 100 else ''}\n"
            if hasattr(rec, 'target_resources') and rec.target_resources:
                response += f"   📋 Affects: {', '.join(rec.target_resources[:2])}{'...' if len(rec.target_resources) > 2 else ''}\n"
            response += "\n"
        
        if len(recommendations) > 5:
            response += f"... and {len(recommendations) - 5} more recommendations.\n"
        
        return response
    
    async def generate_analysis_response(self, recommendation: RecommendationInsight) -> str:
        """Generate detailed analysis response for a recommendation."""
        
        priority_emoji = self._get_priority_emoji(recommendation.priority)
        
        response = f"{priority_emoji} **{recommendation.name}**\n\n"
        response += f"**Description**: {recommendation.description}\n\n"
        
        # Impact analysis
        response += "**Impact Analysis:**\n"
        if recommendation.cost_savings_usd > 0:
            response += f"💰 Cost savings: ${recommendation.cost_savings_usd:,.2f}/month\n"
        
        response += f"🔒 Security impact: {recommendation.security_impact_score:.1%}\n"
        response += f"⚠️  Risk score: {recommendation.risk_score:.1%}\n"
        response += f"⏱️  Estimated implementation time: {recommendation.estimated_time_hours:.1f} hours\n"
        response += f"🔧 Implementation effort: {recommendation.implementation_effort.value}\n\n"
        
        # Affected resources
        if recommendation.target_resources:
            response += "**Affected Resources:**\n"
            for resource in recommendation.target_resources[:3]:
                response += f"- {resource}\n"
            if len(recommendation.target_resources) > 3:
                response += f"- ... and {len(recommendation.target_resources) - 3} more\n"
            response += "\n"
        
        # Compliance impact
        if recommendation.compliance_impacts:
            response += "**Compliance Impact:**\n"
            for compliance in recommendation.compliance_impacts:
                response += f"- {compliance.framework}: {compliance.impact_level} impact\n"
            response += "\n"
        
        # Implementation steps
        if recommendation.remediation_steps:
            response += "**Implementation Steps:**\n"
            for step in recommendation.remediation_steps[:3]:
                response += f"{step.step_number}. {step.title} ({step.estimated_minutes}min)\n"
            if len(recommendation.remediation_steps) > 3:
                response += f"... and {len(recommendation.remediation_steps) - 3} more steps\n"
        
        return response
    
    async def generate_dry_run_response(self, result: Dict[str, Any]) -> str:
        """Generate response for dry run results."""
        
        response = "✅ **Dry Run Successful**\n\n"
        response += "I've simulated applying this recommendation. Here's what would happen:\n\n"
        
        if "estimated_changes" in result:
            response += f"**Changes**: {result['estimated_changes']}\n"
        
        response += "**Next Steps:**\n"
        response += "- Review the proposed changes above\n"
        response += "- If everything looks good, confirm to apply for real\n"
        response += "- Or ask me to show you the detailed implementation steps\n\n"
        
        response += "Would you like me to apply this recommendation for real?"
        
        return response
    
    async def generate_apply_response(self, result: Dict[str, Any]) -> str:
        """Generate response for applied recommendations."""
        
        response = "✅ **Recommendation Applied Successfully**\n\n"
        response += f"The recommendation has been applied to your project.\n\n"
        
        if "state" in result:
            response += f"**Status**: {result['state']}\n"
        
        response += "**Next Steps:**\n"
        response += "- Monitor the changes to ensure they work as expected\n"
        response += "- Verify that your applications still function properly\n"
        response += "- Check back in a few days to see the impact\n\n"
        
        response += "Would you like me to help you verify the implementation or apply another recommendation?"
        
        return response
    
    async def generate_prioritization_response(self, recommendations: List[RecommendationInsight]) -> str:
        """Generate response for prioritized recommendations."""
        
        response = "📊 **Prioritized Recommendations**\n\n"
        response += "I've prioritized your recommendations based on security impact, cost savings, and implementation effort:\n\n"
        
        for i, rec in enumerate(recommendations[:5], 1):
            priority_emoji = self._get_priority_emoji(rec.priority)
            cost_text = f" | ${rec.cost_savings_usd:,.0f}/mo" if rec.cost_savings_usd > 0 else ""
            effort_text = f" | {rec.implementation_effort.value} effort"
            
            response += f"**{i}. {priority_emoji} {rec.name}**\n"
            response += f"   Security: {rec.security_impact_score:.0%}{cost_text}{effort_text}\n"
            response += f"   {rec.description[:80]}{'...' if len(rec.description) > 80 else ''}\n\n"
        
        response += "**Recommendation**: Start with the top 2-3 items for maximum impact."
        
        return response
    
    async def generate_progress_response(self, tracking: SessionRecommendationTracking) -> str:
        """Generate response for progress tracking."""
        
        response = "📈 **Progress Summary**\n\n"
        
        discussed_count = len(tracking.recommendations_discussed)
        applied_count = len(tracking.recommendations_applied)
        dismissed_count = len(tracking.recommendations_dismissed)
        
        response += f"**This Session:**\n"
        response += f"- Recommendations discussed: {discussed_count}\n"
        response += f"- Recommendations applied: {applied_count}\n"
        response += f"- Recommendations dismissed: {dismissed_count}\n\n"
        
        if applied_count > 0:
            response += "**Applied Recommendations:**\n"
            for rec_id in tracking.recommendations_applied:
                response += f"✅ {rec_id}\n"
            response += "\n"
        
        if applied_count == 0:
            response += "No recommendations have been applied yet. Would you like to start with the highest priority item?"
        
        return response
    
    async def generate_security_response(self, recommendations: List[RecommendationInsight]) -> str:
        """Generate response for security-focused queries."""
        
        if not recommendations:
            return "🔒 **Security Status**: Great news! I didn't find any urgent security recommendations. Your current configuration appears to follow security best practices."
        
        high_security_recs = [r for r in recommendations if r.security_impact_score > 0.7]
        
        response = f"🔒 **Security Analysis**: I found {len(recommendations)} security-related recommendations.\n\n"
        
        if high_security_recs:
            response += f"⚠️  **{len(high_security_recs)} High Impact Security Items:**\n\n"
            
            for rec in high_security_recs[:3]:
                response += f"• **{rec.name}**\n"
                response += f"  Security impact: {rec.security_impact_score:.0%} | Priority: {rec.priority.value}\n"
                response += f"  {rec.description[:100]}{'...' if len(rec.description) > 100 else ''}\n\n"
        
        response += "I recommend addressing the high-impact items first to improve your security posture."
        
        return response
    
    async def generate_cost_response(self, recommendations: List[RecommendationInsight]) -> str:
        """Generate response for cost optimization queries."""
        
        if not recommendations:
            return "💰 **Cost Optimization**: Your resources appear to be well-optimized! I didn't find any immediate cost-saving opportunities."
        
        total_savings = sum(rec.cost_savings_usd for rec in recommendations)
        annual_savings = total_savings * 12
        
        response = f"💰 **Cost Optimization Analysis**\n\n"
        response += f"**Potential Savings**: ${total_savings:,.2f}/month (${annual_savings:,.2f}/year)\n\n"
        response += f"**Top Cost-Saving Opportunities:**\n\n"
        
        for i, rec in enumerate(recommendations[:5], 1):
            monthly_savings = rec.cost_savings_usd
            annual = monthly_savings * 12
            
            response += f"{i}. **{rec.name}**\n"
            response += f"   💰 ${monthly_savings:,.2f}/month (${annual:,.2f}/year)\n"
            response += f"   🔧 {rec.implementation_effort.value} effort\n"
            response += f"   {rec.description[:80]}{'...' if len(rec.description) > 80 else ''}\n\n"
        
        response += f"**ROI Timeline**: Most changes pay for themselves within the first month."
        
        return response
    
    def _get_priority_emoji(self, priority: Priority) -> str:
        """Get emoji for priority level."""
        emoji_map = {
            Priority.CRITICAL: "🚨",
            Priority.HIGH: "⚠️",
            Priority.MEDIUM: "📋",
            Priority.LOW: "📝"
        }
        return emoji_map.get(priority, "📋")

class SuggestionEngine:
    """Generates contextual suggestions for users."""
    
    def generate_contextual_suggestions(
        self,
        state: ConversationState,
        recent_recommendations: List[RecommendationInsight]
    ) -> List[str]:
        """Generate contextual suggestions based on conversation state."""
        
        suggestions = []
        
        # Based on current recommendations
        if recent_recommendations:
            high_priority = [r for r in recent_recommendations if r.priority == Priority.CRITICAL]
            if high_priority:
                suggestions.append(f"Apply critical recommendation: {high_priority[0].name}")
            
            cost_savers = [r for r in recent_recommendations if r.cost_savings_usd > 100]
            if cost_savers:
                top_saver = max(cost_savers, key=lambda r: r.cost_savings_usd)
                suggestions.append(f"Save ${top_saver.cost_savings_usd:,.0f}/month with: {top_saver.name}")
        
        # Based on conversation context
        if state.last_intent == QueryIntent.LIST_RECOMMENDATIONS:
            suggestions.extend([
                "Prioritize these recommendations",
                "Focus on security recommendations",
                "Show cost optimization opportunities"
            ])
        elif state.last_intent == QueryIntent.ANALYZE_RECOMMENDATION:
            suggestions.extend([
                "Apply this recommendation",
                "Show implementation steps",
                "Check compliance impact"
            ])
        
        return suggestions[:5]  # Return top 5
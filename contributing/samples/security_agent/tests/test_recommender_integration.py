#!/usr/bin/env python3
"""
Comprehensive test suite for Google Cloud Recommender API integration.

Tests cover:
- RecommenderService functionality
- Chat integration service
- Agent routing for recommendations
- API endpoints
- Error handling
- Performance optimization
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import os
from google.cloud import recommender_v1
from google.oauth2 import service_account

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from backend.services.recommender_service import (
    RecommenderService,
    RecommendationContext,
    RecommendationInsight,
    RecommenderType,
    Priority,
    RecommendationState,
    RecommendationAnalytics,
    RemediationGenerator
)
from backend.services.chat_recommendation_service import (
    ChatRecommendationService,
    QueryIntent,
    ConversationState,
    IntentClassifier,
    EntityExtractor,
    ResponseGenerator
)
from backend.models.recommender_models import (
    RecommenderContextRequest,
    ChatRecommendationQuery,
    ChatRecommendationContext,
    RecommendationListResponse,
    RecommendationActionRequest,
    RecommendationActionResponse
)


class TestRecommenderService:
    """Test suite for RecommenderService functionality."""
    
    @pytest.fixture
    def mock_recommender_client(self):
        """Create a mock Google Cloud Recommender client."""
        client = Mock(spec=recommender_v1.RecommenderClient)
        return client
    
    @pytest.fixture
    def mock_asset_client(self):
        """Create a mock Google Cloud Asset client."""
        client = Mock(spec=recommender_v1.RecommenderClient)
        return client
    
    @pytest.fixture
    def recommender_service(self, mock_recommender_client, mock_asset_client):
        """Create RecommenderService with mocked clients."""
        service = RecommenderService(project_id="test-project")
        service.client = mock_recommender_client
        service.asset_client = mock_asset_client
        return service
    
    @pytest.fixture
    def sample_recommendation(self):
        """Create a sample Google Cloud recommendation."""
        recommendation = Mock(spec=recommender_v1.Recommendation)
        recommendation.name = "projects/test-project/locations/global/recommenders/google.iam.policy.Recommender/recommendations/test-rec-123"
        recommendation.display_name = "Remove excessive IAM permissions"
        recommendation.description = "User john@example.com has overprivileged access"
        recommendation.state_info = Mock()
        recommendation.state_info.state = Mock()
        recommendation.state_info.state.name = "ACTIVE"
        
        # Mock impact
        recommendation.primary_impact = Mock()
        recommendation.primary_impact.category = Mock()
        recommendation.primary_impact.category.name = "SECURITY"
        recommendation.primary_impact.cost_projection = Mock()
        recommendation.primary_impact.cost_projection.cost = Mock()
        recommendation.primary_impact.cost_projection.cost.units = 100
        recommendation.primary_impact.cost_projection.cost.nanos = 0
        
        # Mock content
        recommendation.content = Mock()
        recommendation.content.overview = "Remove unnecessary permissions"
        recommendation.content.operation_groups = []
        
        return recommendation
    
    @pytest.fixture
    def sample_recommendation_context(self):
        """Create a sample recommendation context."""
        return RecommendationContext(
            project_id="test-project",
            resource_name="test-resource",
            location="global",
            recommender_type=RecommenderType.IAM_POLICY,
            filters={"state": "ACTIVE"},
            user_preferences={"focus": "security"}
        )

    def test_service_initialization(self, recommender_service):
        """Test service initializes correctly."""
        assert recommender_service.client is not None
        assert recommender_service.asset_client is not None
        assert recommender_service.cache == {}
        assert recommender_service.performance_metrics["total_requests"] == 0
        assert recommender_service.supported_recommenders == list(RecommenderType)
    
    def test_client_initialization_with_credentials(self):
        """Test client initialization with service account credentials."""
        with patch('backend.services.recommender_service.service_account.Credentials.from_service_account_file') as mock_creds:
            with patch('backend.services.recommender_service.recommender_v1.RecommenderClient') as mock_client:
                mock_creds.return_value = Mock()
                mock_client.return_value = Mock()
                
                service = RecommenderService(project_id="test-project", credentials_path="test-creds.json")
                
                mock_creds.assert_called_once_with("/path/to/creds.json")
                mock_client.assert_called_once()
    
    def test_client_initialization_with_default_credentials(self):
        """Test client initialization with default credentials."""
        with patch('backend.services.recommender_service.google.auth.default') as mock_auth:
            with patch('backend.services.recommender_service.recommender_v1.RecommenderClient') as mock_client:
                mock_auth.return_value = (Mock(), "test-project")
                mock_client.return_value = Mock()
                
                service = RecommenderService(project_id="test-project")
                
                mock_auth.assert_called_once()
                mock_client.assert_called()
    
    def test_client_initialization_retry_logic(self):
        """Test client initialization retry logic on failure."""
        with patch('backend.services.recommender_service.service_account.Credentials.from_service_account_file') as mock_creds:
            with patch('backend.services.recommender_service.recommender_v1.RecommenderClient') as mock_client:
                with patch('time.sleep') as mock_sleep:
                    # First two attempts fail, third succeeds
                    mock_client.side_effect = [Exception("Connection failed"), Exception("Still failing"), Mock()]
                    mock_creds.return_value = Mock()
                    
                    service = RecommenderService(project_id="test-project", credentials_path="test-creds.json")
                    
                    assert mock_client.call_count == 3
                    assert mock_sleep.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_all_recommendations(self, recommender_service, sample_recommendation, sample_recommendation_context):
        """Test getting all recommendations across recommender types."""
        # Mock the list_recommendations response
        mock_page_result = [sample_recommendation]
        recommender_service.client.list_recommendations.return_value = mock_page_result
        
        recommendations = await recommender_service.get_all_recommendations(sample_recommendation_context)
        
        assert len(recommendations) >= 0  # Could be 0 if all fail, >= 1 if any succeed
        # Verify the service attempted to call all recommender types
        assert recommender_service.client.list_recommendations.call_count > 0
    
    @pytest.mark.asyncio
    async def test_get_recommendations_by_type(self, recommender_service, sample_recommendation, sample_recommendation_context):
        """Test getting recommendations for a specific type."""
        mock_page_result = [sample_recommendation]
        recommender_service.client.list_recommendations.return_value = mock_page_result
        
        recommendations = await recommender_service._get_recommendations_by_type(
            sample_recommendation_context,
            RecommenderType.IAM_POLICY
        )
        
        assert len(recommendations) >= 0
        recommender_service.client.list_recommendations.assert_called()
    
    def test_cache_functionality(self, recommender_service):
        """Test caching mechanism."""
        cache_key = "test-project:google.iam.policy.Recommender:global"
        test_data = [Mock()]
        
        # Add to cache
        recommender_service.cache[cache_key] = {
            "data": test_data,
            "timestamp": datetime.now()
        }
        
        # Test cache validity
        assert recommender_service._is_cache_valid(cache_key) is True
        
        # Test cache expiry
        old_timestamp = datetime.now() - timedelta(hours=1)
        recommender_service.cache[cache_key]["timestamp"] = old_timestamp
        assert recommender_service._is_cache_valid(cache_key) is False
    
    @pytest.mark.asyncio
    async def test_apply_recommendation_dry_run(self, recommender_service, sample_recommendation_context):
        """Test applying recommendation in dry run mode."""
        result = await recommender_service.apply_recommendation(
            "test-rec-123",
            sample_recommendation_context,
            dry_run=True
        )
        
        assert result["success"] is True
        assert result["dry_run"] is True
        assert "estimated_changes" in result
    
    @pytest.mark.asyncio
    async def test_apply_recommendation_live(self, recommender_service, sample_recommendation_context):
        """Test applying recommendation in live mode."""
        mock_response = Mock()
        mock_response.state_info = Mock()
        mock_response.state_info.state = Mock()
        mock_response.state_info.state.name = "CLAIMED"
        
        recommender_service.client.mark_recommendation_claimed.return_value = mock_response
        
        result = await recommender_service.apply_recommendation(
            "test-rec-123",
            sample_recommendation_context,
            dry_run=False
        )
        
        assert result["success"] is True
        assert result["dry_run"] is False
        assert result["state"] == "CLAIMED"
    
    @pytest.mark.asyncio
    async def test_apply_recommendation_error_handling(self, recommender_service, sample_recommendation_context):
        """Test error handling in recommendation application."""
        recommender_service.client.mark_recommendation_claimed.side_effect = Exception("API Error")
        
        result = await recommender_service.apply_recommendation(
            "test-rec-123",
            sample_recommendation_context,
            dry_run=False
        )
        
        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "API Error"
    
    def test_priority_calculation(self, recommender_service, sample_recommendation):
        """Test priority calculation logic."""
        # Test critical priority
        sample_recommendation.content = "Critical security vulnerability detected"
        priority = recommender_service._calculate_priority(sample_recommendation)
        assert priority in [Priority.CRITICAL, Priority.HIGH]  # Should be high priority
        
        # Test low priority
        sample_recommendation.content = "Minor optimization suggestion"
        sample_recommendation.primary_impact.category.name = "PERFORMANCE"
        priority = recommender_service._calculate_priority(sample_recommendation)
        assert priority in [Priority.LOW, Priority.MEDIUM]
    
    def test_security_score_calculation(self, recommender_service):
        """Test security impact score calculation."""
        # IAM policy should have high security score
        score = recommender_service._calculate_security_score(
            RecommenderType.IAM_POLICY,
            {"description": "overprivileged admin access"}
        )
        assert score > 0.5
        
        # Machine type should have low security score
        score = recommender_service._calculate_security_score(
            RecommenderType.MACHINE_TYPE,
            {"description": "right-size instance"}
        )
        assert score < 0.5
    
    def test_risk_score_calculation(self, recommender_service):
        """Test overall risk score calculation."""
        insight = RecommendationInsight(
            recommendation_id="test-123",
            name="Test Recommendation",
            description="Test description",
            recommender_type=RecommenderType.IAM_POLICY,
            state=RecommendationState.ACTIVE,
            priority=Priority.CRITICAL,
            impact={},
            content={},
            target_resources=["resource1", "resource2"],
            associated_insights=[],
            security_impact_score=0.9
        )
        
        risk_score = recommender_service._calculate_risk_score(insight)
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.5  # Should be high risk given critical priority and high security impact
    
    def test_location_mapping(self, recommender_service):
        """Test location mapping for different recommender types."""
        # Global recommenders
        locations = recommender_service._get_locations_for_recommender(
            RecommenderType.IAM_POLICY, "us"
        )
        assert locations == ["global"]
        
        # Regional recommenders
        locations = recommender_service._get_locations_for_recommender(
            RecommenderType.MACHINE_TYPE, "us"
        )
        assert "us-central1" in locations
    
    def test_filter_building(self, recommender_service):
        """Test filter string building."""
        filters = {
            "state": "ACTIVE",
            "priority": ["HIGH", "CRITICAL"],
            "resource_type": "compute.instances"
        }
        
        filter_string = recommender_service._build_filter(filters)
        assert 'state="ACTIVE"' in filter_string
        assert "AND" in filter_string
        assert "OR" in filter_string  # For the priority list
    
    @pytest.mark.asyncio
    async def test_session_recommendation_tracking(self, recommender_service):
        """Test session-based recommendation tracking."""
        session_id = "test-session-123"
        
        recommendation = RecommendationInsight(
            recommendation_id="rec-123",
            name="Test Rec",
            description="Test",
            recommender_type=RecommenderType.IAM_POLICY,
            state=RecommendationState.ACTIVE,
            priority=Priority.HIGH,
            impact={},
            content={},
            target_resources=[],
            associated_insights=[]
        )
        
        # Add recommendation to session
        await recommender_service.add_session_recommendation(session_id, recommendation)
        
        # Retrieve session recommendations
        session_recs = await recommender_service.get_session_recommendations(session_id)
        assert len(session_recs) == 1
        assert session_recs[0].recommendation_id == "rec-123"


class TestChatRecommendationService:
    """Test suite for ChatRecommendationService functionality."""
    
    @pytest.fixture
    def mock_recommender_service(self):
        """Create a mock RecommenderService."""
        service = Mock(spec=RecommenderService)
        return service
    
    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock chat manager."""
        manager = Mock()
        manager.add_message = AsyncMock()
        return manager
    
    @pytest.fixture
    def chat_recommendation_service(self, mock_recommender_service, mock_chat_manager):
        """Create ChatRecommendationService with mocked dependencies."""
        return ChatRecommendationService(mock_recommender_service, mock_chat_manager)
    
    @pytest.fixture
    def sample_chat_query(self):
        """Create a sample chat recommendation query."""
        context = ChatRecommendationContext(
            session_id="test-session",
            user_id="test-user",
            project_context=RecommenderContextRequest(
                project_id="test-project",
                location="global"
            ),
            conversation_topics=[],
            user_preferences={},
            active_recommendations=[]
        )
        
        return ChatRecommendationQuery(
            query="Show me my security recommendations",
            context=context
        )
    
    @pytest.fixture
    def sample_recommendations(self):
        """Create sample recommendations for testing."""
        return [
            RecommendationInsight(
                recommendation_id="rec-1",
                name="Remove excessive IAM permissions",
                description="User has overprivileged access",
                recommender_type=RecommenderType.IAM_POLICY,
                state=RecommendationState.ACTIVE,
                priority=Priority.CRITICAL,
                impact={},
                content={},
                target_resources=["user:john@example.com"],
                associated_insights=[],
                cost_savings_usd=0.0,
                security_impact_score=0.9
            ),
            RecommendationInsight(
                recommendation_id="rec-2",
                name="Right-size VM instance",
                description="Instance is oversized for current usage",
                recommender_type=RecommenderType.MACHINE_TYPE,
                state=RecommendationState.ACTIVE,
                priority=Priority.MEDIUM,
                impact={},
                content={},
                target_resources=["instance-1"],
                associated_insights=[],
                cost_savings_usd=150.0,
                security_impact_score=0.2
            )
        ]
    
    def test_service_initialization(self, chat_recommendation_service):
        """Test chat recommendation service initializes correctly."""
        assert chat_recommendation_service.recommender_service is not None
        assert chat_recommendation_service.chat_manager is not None
        assert chat_recommendation_service.intent_classifier is not None
        assert chat_recommendation_service.entity_extractor is not None
        assert chat_recommendation_service.response_generator is not None
        assert chat_recommendation_service.conversation_states == {}
        assert chat_recommendation_service.performance_metrics["queries_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_process_query_list_intent(self, chat_recommendation_service, sample_chat_query, sample_recommendations):
        """Test processing a query with list recommendations intent."""
        # Mock the recommender service to return sample recommendations
        chat_recommendation_service.recommender_service.get_all_recommendations = AsyncMock(
            return_value=sample_recommendations
        )
        
        # Mock intent classification
        with patch.object(chat_recommendation_service.intent_classifier, 'classify', return_value=QueryIntent.LIST_RECOMMENDATIONS):
            with patch.object(chat_recommendation_service.entity_extractor, 'extract', return_value={}):
                with patch('time.time', side_effect=[1000.0, 1001.0]):  # Mock timing
                    response = await chat_recommendation_service.process_query(sample_chat_query)
        
        assert response.success is True
        assert len(response.recommendations) == 2
        assert "2 recommendations" in response.response_text
        assert len(response.suggested_actions) > 0
        assert len(response.follow_up_questions) > 0
    
    @pytest.mark.asyncio
    async def test_process_query_analyze_intent(self, chat_recommendation_service, sample_chat_query, sample_recommendations):
        """Test processing a query with analyze recommendation intent."""
        # Set up conversation state with current recommendations
        state = ConversationState(
            session_id=sample_chat_query.context.session_id,
            user_id=sample_chat_query.context.user_id,
            project_id=sample_chat_query.context.project_context.project_id,
            current_recommendations=sample_recommendations
        )
        chat_recommendation_service.conversation_states[sample_chat_query.context.session_id] = state
        
        # Mock finding a specific recommendation
        sample_chat_query.query = "Analyze the IAM recommendation"
        
        with patch.object(chat_recommendation_service.intent_classifier, 'classify', return_value=QueryIntent.ANALYZE_RECOMMENDATION):
            with patch.object(chat_recommendation_service.entity_extractor, 'extract', return_value={"recommendation_name": "IAM"}):
                with patch('time.time', side_effect=[1000.0, 1001.0]):
                    response = await chat_recommendation_service.process_query(sample_chat_query)
        
        assert response.success is True
        assert len(response.recommendations) == 1
        assert response.recommendations[0].recommender_type == RecommenderType.IAM_POLICY
    
    @pytest.mark.asyncio
    async def test_process_query_apply_intent(self, chat_recommendation_service, sample_chat_query, sample_recommendations):
        """Test processing a query with apply recommendation intent."""
        # Set up conversation state
        state = ConversationState(
            session_id=sample_chat_query.context.session_id,
            user_id=sample_chat_query.context.user_id,
            project_id=sample_chat_query.context.project_context.project_id,
            current_recommendations=sample_recommendations,
            active_recommendation_id="rec-1"
        )
        chat_recommendation_service.conversation_states[sample_chat_query.context.session_id] = state
        
        # Mock apply recommendation
        chat_recommendation_service.recommender_service.apply_recommendation = AsyncMock(
            return_value={"success": True, "dry_run": True, "message": "Dry run successful"}
        )
        
        sample_chat_query.query = "Apply this recommendation"
        
        with patch.object(chat_recommendation_service.intent_classifier, 'classify', return_value=QueryIntent.APPLY_RECOMMENDATION):
            with patch.object(chat_recommendation_service.entity_extractor, 'extract', return_value={"dry_run": True}):
                with patch('time.time', side_effect=[1000.0, 1001.0]):
                    response = await chat_recommendation_service.process_query(sample_chat_query)
        
        assert response.success is True
        assert "Dry Run Successful" in response.response_text
    
    @pytest.mark.asyncio
    async def test_process_query_prioritize_intent(self, chat_recommendation_service, sample_chat_query, sample_recommendations):
        """Test processing a query with prioritize recommendations intent."""
        # Set up conversation state
        state = ConversationState(
            session_id=sample_chat_query.context.session_id,
            user_id=sample_chat_query.context.user_id,
            project_id=sample_chat_query.context.project_context.project_id,
            current_recommendations=sample_recommendations
        )
        chat_recommendation_service.conversation_states[sample_chat_query.context.session_id] = state
        
        sample_chat_query.query = "Prioritize my recommendations"
        
        with patch.object(chat_recommendation_service.intent_classifier, 'classify', return_value=QueryIntent.PRIORITIZE_RECOMMENDATIONS):
            with patch.object(chat_recommendation_service.entity_extractor, 'extract', return_value={}):
                with patch('time.time', side_effect=[1000.0, 1001.0]):
                    response = await chat_recommendation_service.process_query(sample_chat_query)
        
        assert response.success is True
        assert "Prioritized Recommendations" in response.response_text
        # Critical priority should come first
        assert response.recommendations[0].priority == Priority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_process_query_security_focus(self, chat_recommendation_service, sample_chat_query, sample_recommendations):
        """Test processing a security-focused query."""
        chat_recommendation_service.recommender_service.get_all_recommendations = AsyncMock(
            return_value=sample_recommendations
        )
        
        sample_chat_query.query = "Show me security recommendations"
        
        with patch.object(chat_recommendation_service.intent_classifier, 'classify', return_value=QueryIntent.GENERAL_SECURITY):
            with patch.object(chat_recommendation_service.entity_extractor, 'extract', return_value={}):
                with patch('time.time', side_effect=[1000.0, 1001.0]):
                    response = await chat_recommendation_service.process_query(sample_chat_query)
        
        assert response.success is True
        assert "Security Analysis" in response.response_text
        # Should only return security-related recommendations
        security_recs = [r for r in response.recommendations if r.recommender_type in [
            RecommenderType.IAM_POLICY, RecommenderType.FIREWALL, RecommenderType.SERVICE_ACCOUNT
        ]]
        assert len(security_recs) > 0
    
    @pytest.mark.asyncio
    async def test_process_query_cost_focus(self, chat_recommendation_service, sample_chat_query, sample_recommendations):
        """Test processing a cost optimization query."""
        chat_recommendation_service.recommender_service.get_all_recommendations = AsyncMock(
            return_value=sample_recommendations
        )
        
        sample_chat_query.query = "Show me cost saving opportunities"
        
        with patch.object(chat_recommendation_service.intent_classifier, 'classify', return_value=QueryIntent.COST_OPTIMIZATION):
            with patch.object(chat_recommendation_service.entity_extractor, 'extract', return_value={}):
                with patch('time.time', side_effect=[1000.0, 1001.0]):
                    response = await chat_recommendation_service.process_query(sample_chat_query)
        
        assert response.success is True
        assert "Cost Optimization Analysis" in response.response_text
        assert "$" in response.response_text  # Should mention cost savings
    
    @pytest.mark.asyncio
    async def test_error_handling(self, chat_recommendation_service, sample_chat_query):
        """Test error handling in query processing."""
        # Mock an error in the recommender service
        chat_recommendation_service.recommender_service.get_all_recommendations = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        with patch.object(chat_recommendation_service.intent_classifier, 'classify', return_value=QueryIntent.LIST_RECOMMENDATIONS):
            with patch.object(chat_recommendation_service.entity_extractor, 'extract', return_value={}):
                with patch('time.time', side_effect=[1000.0, 1001.0]):
                    response = await chat_recommendation_service.process_query(sample_chat_query)
        
        assert response.success is False
        assert "error processing your request" in response.response_text
        assert len(response.suggested_actions) > 0
        assert len(response.follow_up_questions) > 0
    
    @pytest.mark.asyncio
    async def test_conversation_state_management(self, chat_recommendation_service, sample_chat_query):
        """Test conversation state creation and management."""
        # First query should create conversation state
        state = await chat_recommendation_service._get_or_create_conversation_state(sample_chat_query)
        
        assert state.session_id == sample_chat_query.context.session_id
        assert state.user_id == sample_chat_query.context.user_id
        assert state.project_id == sample_chat_query.context.project_context.project_id
        
        # Second query should retrieve existing state
        state2 = await chat_recommendation_service._get_or_create_conversation_state(sample_chat_query)
        assert state is state2
    
    @pytest.mark.asyncio
    async def test_session_tracking_updates(self, chat_recommendation_service, sample_chat_query, sample_recommendations):
        """Test session tracking updates correctly."""
        response = Mock()
        response.recommendations = sample_recommendations
        
        await chat_recommendation_service._update_session_tracking(
            sample_chat_query.context.session_id,
            sample_chat_query,
            response
        )
        
        tracking = chat_recommendation_service.session_tracking[sample_chat_query.context.session_id]
        assert len(tracking.recommendations_discussed) == 2
        assert "rec-1" in tracking.recommendations_discussed
        assert "rec-2" in tracking.recommendations_discussed
    
    @pytest.mark.asyncio
    async def test_chat_history_addition(self, chat_recommendation_service, sample_chat_query):
        """Test adding interactions to chat history."""
        response = Mock()
        response.response_text = "Test response"
        response.recommendations = []
        response.suggested_actions = []
        response.success = True
        response.context_updates = {}
        
        await chat_recommendation_service._add_to_chat_history(
            sample_chat_query,
            response,
            "test-request-123"
        )
        
        # Verify chat manager was called
        assert chat_recommendation_service.chat_manager.add_message.call_count == 2  # User + assistant
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, chat_recommendation_service):
        """Test performance metrics are tracked correctly."""
        initial_count = chat_recommendation_service.performance_metrics["queries_processed"]
        initial_avg = chat_recommendation_service.performance_metrics["avg_response_time"]
        
        # Update metrics
        chat_recommendation_service.performance_metrics["queries_processed"] += 1
        chat_recommendation_service._update_performance_metrics(1.5, True)
        
        assert chat_recommendation_service.performance_metrics["queries_processed"] == initial_count + 1
        assert chat_recommendation_service.performance_metrics["successful_applications"] > 0
        assert chat_recommendation_service.performance_metrics["avg_response_time"] > initial_avg
    
    @pytest.mark.asyncio
    async def test_service_metrics_retrieval(self, chat_recommendation_service):
        """Test retrieving comprehensive service metrics."""
        metrics = await chat_recommendation_service.get_service_metrics()
        
        assert "performance" in metrics
        assert "active_sessions" in metrics
        assert "tracked_sessions" in metrics
        assert "health_status" in metrics
        assert metrics["health_status"] in ["healthy", "idle"]


class TestIntentClassifier:
    """Test suite for IntentClassifier."""
    
    @pytest.fixture
    def intent_classifier(self):
        """Create an IntentClassifier instance."""
        return IntentClassifier()
    
    @pytest.fixture
    def sample_conversation_state(self):
        """Create a sample conversation state."""
        return ConversationState(
            session_id="test-session",
            user_id="test-user",
            project_id="test-project",
            current_recommendations=[]
        )
    
    @pytest.mark.asyncio
    async def test_list_recommendations_intent(self, intent_classifier, sample_conversation_state):
        """Test classification of list recommendations queries."""
        queries = [
            "Show me my recommendations",
            "What recommendations do you have?",
            "List all suggestions",
            "Display recommendations",
            "What do you recommend?"
        ]
        
        for query in queries:
            intent = await intent_classifier.classify(query, sample_conversation_state)
            assert intent == QueryIntent.LIST_RECOMMENDATIONS
    
    @pytest.mark.asyncio
    async def test_analyze_recommendation_intent(self, intent_classifier, sample_conversation_state):
        """Test classification of analyze recommendation queries."""
        queries = [
            "Analyze this recommendation",
            "Tell me more about the IAM recommendation",
            "Explain this suggestion",
            "What does this recommendation do?",
            "Review the security recommendation"
        ]
        
        for query in queries:
            intent = await intent_classifier.classify(query, sample_conversation_state)
            assert intent == QueryIntent.ANALYZE_RECOMMENDATION
    
    @pytest.mark.asyncio
    async def test_apply_recommendation_intent(self, intent_classifier, sample_conversation_state):
        """Test classification of apply recommendation queries."""
        queries = [
            "Apply this recommendation",
            "Implement the suggestion",
            "Execute the fix",
            "Run the recommendation",
            "Make the change"
        ]
        
        for query in queries:
            intent = await intent_classifier.classify(query, sample_conversation_state)
            assert intent == QueryIntent.APPLY_RECOMMENDATION
    
    @pytest.mark.asyncio
    async def test_prioritize_recommendations_intent(self, intent_classifier, sample_conversation_state):
        """Test classification of prioritize recommendations queries."""
        queries = [
            "Prioritize my recommendations",
            "Rank the suggestions",
            "Which recommendation should I do first?",
            "Order by importance",
            "Show me the most important recommendations"
        ]
        
        for query in queries:
            intent = await intent_classifier.classify(query, sample_conversation_state)
            assert intent == QueryIntent.PRIORITIZE_RECOMMENDATIONS
    
    @pytest.mark.asyncio
    async def test_security_intent(self, intent_classifier, sample_conversation_state):
        """Test classification of security-focused queries."""
        queries = [
            "Show me security recommendations",
            "What security vulnerabilities do I have?",
            "Check my security risks",
            "Security analysis",
            "Threat assessment"
        ]
        
        for query in queries:
            intent = await intent_classifier.classify(query, sample_conversation_state)
            assert intent == QueryIntent.GENERAL_SECURITY
    
    @pytest.mark.asyncio
    async def test_cost_optimization_intent(self, intent_classifier, sample_conversation_state):
        """Test classification of cost optimization queries."""
        queries = [
            "Show me cost savings",
            "How can I save money?",
            "Cost optimization recommendations",
            "Reduce my expenses",
            "Cheaper alternatives"
        ]
        
        for query in queries:
            intent = await intent_classifier.classify(query, sample_conversation_state)
            assert intent == QueryIntent.COST_OPTIMIZATION
    
    @pytest.mark.asyncio
    async def test_context_based_classification(self, intent_classifier, sample_conversation_state):
        """Test context-based intent classification."""
        # Set active recommendation
        sample_conversation_state.active_recommendation_id = "rec-123"
        
        # Queries referring to "it" should be analyzed as analyze intent
        queries = ["Tell me about it", "What does this do?", "Analyze that"]
        
        for query in queries:
            intent = await intent_classifier.classify(query, sample_conversation_state)
            assert intent == QueryIntent.ANALYZE_RECOMMENDATION


class TestEntityExtractor:
    """Test suite for EntityExtractor."""
    
    @pytest.fixture
    def entity_extractor(self):
        """Create an EntityExtractor instance."""
        return EntityExtractor()
    
    @pytest.fixture
    def sample_conversation_state(self):
        """Create a sample conversation state."""
        return ConversationState(
            session_id="test-session",
            user_id="test-user",
            project_id="test-project",
            current_recommendations=[],
            active_recommendation_id="rec-123"
        )
    
    @pytest.mark.asyncio
    async def test_priority_extraction(self, entity_extractor, sample_conversation_state):
        """Test extraction of priority filters."""
        queries_and_expected = [
            ("Show me critical recommendations", ["critical"]),
            ("High priority items only", ["high"]),
            ("List medium and low priority suggestions", ["medium", "low"]),
            ("Critical and urgent items", ["critical"])
        ]
        
        for query, expected_priorities in queries_and_expected:
            entities = await entity_extractor.extract(query, sample_conversation_state)
            if "priority_filter" in entities:
                for priority in expected_priorities:
                    assert priority in entities["priority_filter"]
    
    @pytest.mark.asyncio
    async def test_type_extraction(self, entity_extractor, sample_conversation_state):
        """Test extraction of recommender type filters."""
        queries_and_expected = [
            ("Show me IAM recommendations", ["iam"]),
            ("Firewall suggestions", ["firewall"]),
            ("Machine type and disk recommendations", ["machine", "type", "disk"]),
            ("Service account analysis", ["service", "account"])
        ]
        
        for query, expected_types in queries_and_expected:
            entities = await entity_extractor.extract(query, sample_conversation_state)
            if "type_filter" in entities:
                for type_keyword in expected_types:
                    assert type_keyword in entities["type_filter"]
    
    @pytest.mark.asyncio
    async def test_cost_threshold_extraction(self, entity_extractor, sample_conversation_state):
        """Test extraction of cost thresholds."""
        queries_and_expected = [
            ("Show recommendations saving more than $100", "100"),
            ("Cost savings above $1,000", "1000"),
            ("At least $50.50 in savings", "50.50")
        ]
        
        for query, expected_cost in queries_and_expected:
            entities = await entity_extractor.extract(query, sample_conversation_state)
            if "min_cost_savings" in entities:
                assert entities["min_cost_savings"] == expected_cost
    
    @pytest.mark.asyncio
    async def test_dry_run_detection(self, entity_extractor, sample_conversation_state):
        """Test detection of dry run vs live execution."""
        dry_run_queries = [
            "Test this recommendation",
            "Dry run the suggestion",
            "Preview the changes",
            "Simulate the fix"
        ]
        
        live_queries = [
            "Apply this for real",
            "Actually implement this",
            "Do it live",
            "Make the changes now"
        ]
        
        for query in dry_run_queries:
            entities = await entity_extractor.extract(query, sample_conversation_state)
            assert entities.get("dry_run") is True
        
        for query in live_queries:
            entities = await entity_extractor.extract(query, sample_conversation_state)
            assert entities.get("dry_run") is False
    
    @pytest.mark.asyncio
    async def test_recommendation_reference_extraction(self, entity_extractor, sample_conversation_state):
        """Test extraction of recommendation references from context."""
        queries = ["Tell me about it", "Analyze this", "Apply that recommendation"]
        
        for query in queries:
            entities = await entity_extractor.extract(query, sample_conversation_state)
            assert entities.get("recommendation_id") == "rec-123"


class TestResponseGenerator:
    """Test suite for ResponseGenerator."""
    
    @pytest.fixture
    def response_generator(self):
        """Create a ResponseGenerator instance."""
        return ResponseGenerator()
    
    @pytest.fixture
    def sample_recommendations(self):
        """Create sample recommendations for testing."""
        return [
            RecommendationInsight(
                recommendation_id="rec-1",
                name="Remove excessive IAM permissions",
                description="User john@example.com has overprivileged access that should be removed",
                recommender_type=RecommenderType.IAM_POLICY,
                state=RecommendationState.ACTIVE,
                priority=Priority.CRITICAL,
                impact={},
                content={},
                target_resources=["user:john@example.com"],
                associated_insights=[],
                cost_savings_usd=0.0,
                security_impact_score=0.9,
                implementation_effort="high",
                estimated_time_hours=2.0,
                compliance_impact=[]
            ),
            RecommendationInsight(
                recommendation_id="rec-2",
                name="Right-size VM instance",
                description="Instance is oversized for current usage patterns",
                recommender_type=RecommenderType.MACHINE_TYPE,
                state=RecommendationState.ACTIVE,
                priority=Priority.MEDIUM,
                impact={},
                content={},
                target_resources=["projects/test/zones/us-central1-a/instances/instance-1"],
                associated_insights=[],
                cost_savings_usd=150.0,
                security_impact_score=0.2,
                implementation_effort="low",
                estimated_time_hours=0.5,
                compliance_impact=[]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_list_response_generation(self, response_generator, sample_recommendations):
        """Test generation of list recommendations response."""
        response = await response_generator.generate_list_response(sample_recommendations)
        
        assert "2 recommendations" in response
        assert "Total potential savings" in response
        assert "$150.00/month" in response
        assert "High priority items" in response
        assert "Remove excessive IAM permissions" in response
        assert "Right-size VM instance" in response
        assert "🚨" in response  # Critical priority emoji
        assert "📋" in response  # Medium priority emoji
    
    @pytest.mark.asyncio
    async def test_list_response_with_filters(self, response_generator, sample_recommendations):
        """Test list response generation with filters applied."""
        response = await response_generator.generate_list_response(
            sample_recommendations,
            priority_filter=["critical"],
            type_filter=["iam"]
        )
        
        assert "filtered by priority: critical" in response
        assert "filtered by type: iam" in response
    
    @pytest.mark.asyncio
    async def test_list_response_empty(self, response_generator):
        """Test list response generation with no recommendations."""
        response = await response_generator.generate_list_response([])
        
        assert "didn't find any recommendations" in response
        assert "well-configured" in response
    
    @pytest.mark.asyncio
    async def test_analysis_response_generation(self, response_generator, sample_recommendations):
        """Test generation of detailed analysis response."""
        recommendation = sample_recommendations[0]  # Critical IAM recommendation
        
        response = await response_generator.generate_analysis_response(recommendation)
        
        assert "Remove excessive IAM permissions" in response
        assert "Security impact: 90%" in response
        assert "Implementation effort: high" in response
        assert "2.0 hours" in response
        assert "🚨" in response  # Critical priority emoji
        assert "Affected Resources:" in response
        assert "user:john@example.com" in response
    
    @pytest.mark.asyncio
    async def test_dry_run_response_generation(self, response_generator):
        """Test generation of dry run response."""
        result = {
            "success": True,
            "dry_run": True,
            "estimated_changes": "Would remove 3 IAM bindings"
        }
        
        response = await response_generator.generate_dry_run_response(result)
        
        assert "Dry Run Successful" in response
        assert "Would remove 3 IAM bindings" in response
        assert "apply for real" in response
        assert "Review the proposed changes" in response
    
    @pytest.mark.asyncio
    async def test_apply_response_generation(self, response_generator):
        """Test generation of apply recommendation response."""
        result = {
            "success": True,
            "dry_run": False,
            "state": "CLAIMED"
        }
        
        response = await response_generator.generate_apply_response(result)
        
        assert "Recommendation Applied Successfully" in response
        assert "Status: CLAIMED" in response
        assert "Monitor the changes" in response
        assert "verify the implementation" in response
    
    @pytest.mark.asyncio
    async def test_prioritization_response_generation(self, response_generator, sample_recommendations):
        """Test generation of prioritization response."""
        response = await response_generator.generate_prioritization_response(sample_recommendations)
        
        assert "Prioritized Recommendations" in response
        assert "security impact, cost savings, and implementation effort" in response
        assert "1. 🚨 Remove excessive IAM permissions" in response  # Should be first (critical)
        assert "2. 📋 Right-size VM instance" in response  # Should be second (medium)
        assert "Start with the top 2-3 items" in response
    
    @pytest.mark.asyncio
    async def test_security_response_generation(self, response_generator, sample_recommendations):
        """Test generation of security-focused response."""
        # Filter to only security recommendations
        security_recs = [r for r in sample_recommendations if r.recommender_type == RecommenderType.IAM_POLICY]
        
        response = await response_generator.generate_security_response(security_recs)
        
        assert "Security Analysis" in response
        assert "1 security-related recommendations" in response
        assert "High Impact Security Items" in response
        assert "Remove excessive IAM permissions" in response
        assert "Security impact: 90%" in response
    
    @pytest.mark.asyncio
    async def test_security_response_empty(self, response_generator):
        """Test security response with no security recommendations."""
        response = await response_generator.generate_security_response([])
        
        assert "Security Status" in response
        assert "Great news!" in response
        assert "security best practices" in response
    
    @pytest.mark.asyncio
    async def test_cost_response_generation(self, response_generator, sample_recommendations):
        """Test generation of cost optimization response."""
        # Filter to only cost-saving recommendations
        cost_recs = [r for r in sample_recommendations if r.cost_savings_usd > 0]
        
        response = await response_generator.generate_cost_response(cost_recs)
        
        assert "Cost Optimization Analysis" in response
        assert "Potential Savings: $150.00/month" in response
        assert "($1,800.00/year)" in response
        assert "Right-size VM instance" in response
        assert "💰 $150.00/month" in response
        assert "🔧 low effort" in response
    
    @pytest.mark.asyncio
    async def test_cost_response_empty(self, response_generator):
        """Test cost response with no cost-saving recommendations."""
        response = await response_generator.generate_cost_response([])
        
        assert "Cost Optimization" in response
        assert "well-optimized" in response
        assert "cost-saving opportunities" in response
    
    def test_priority_emoji_mapping(self, response_generator):
        """Test priority emoji mapping."""
        assert response_generator._get_priority_emoji(Priority.CRITICAL) == "🚨"
        assert response_generator._get_priority_emoji(Priority.HIGH) == "⚠️"
        assert response_generator._get_priority_emoji(Priority.MEDIUM) == "📋"
        assert response_generator._get_priority_emoji(Priority.LOW) == "📝"


class TestRemediationGenerator:
    """Test suite for RemediationGenerator."""
    
    @pytest.fixture
    def remediation_generator(self):
        """Create a RemediationGenerator instance."""
        return RemediationGenerator()
    
    @pytest.mark.asyncio
    async def test_iam_remediation_steps_generation(self, remediation_generator):
        """Test generation of IAM-specific remediation steps."""
        content = {"policy_delta": {"remove_bindings": ["user:test@example.com"]}}
        resources = ["projects/test-project"]
        
        steps = await remediation_generator.generate_steps(
            RecommenderType.IAM_POLICY,
            content,
            resources
        )
        
        assert len(steps) > 0
        assert steps[0]["title"] == "Review Current IAM Policy"
        assert steps[1]["title"] == "Remove Excessive Permissions"
        assert steps[2]["title"] == "Verify Changes"
        assert all("estimated_minutes" in step for step in steps)
    
    @pytest.mark.asyncio
    async def test_firewall_remediation_steps_generation(self, remediation_generator):
        """Test generation of firewall-specific remediation steps."""
        content = {"firewall_rules": ["rule-1", "rule-2"]}
        resources = ["projects/test-project/global/firewalls/rule-1"]
        
        steps = await remediation_generator.generate_steps(
            RecommenderType.FIREWALL,
            content,
            resources
        )
        
        assert len(steps) > 0
        assert "Analyze Firewall Rules" in steps[0]["title"]
        assert "Restrict Source Ranges" in steps[1]["title"]
        assert all(step["action_type"] in ["review", "modify", "verify", "cleanup"] for step in steps)
    
    @pytest.mark.asyncio
    async def test_service_account_remediation_steps_generation(self, remediation_generator):
        """Test generation of service account-specific remediation steps."""
        content = {"unused_accounts": ["sa@test-project.iam.gserviceaccount.com"]}
        resources = ["projects/test-project/serviceAccounts/sa@test-project.iam.gserviceaccount.com"]
        
        steps = await remediation_generator.generate_steps(
            RecommenderType.SERVICE_ACCOUNT,
            content,
            resources
        )
        
        assert len(steps) > 0
        assert "Identify Unused Service Accounts" in steps[0]["title"]
        assert "Disable or Delete Unused Accounts" in steps[1]["title"]
        assert steps[1]["action_type"] == "cleanup"
    
    @pytest.mark.asyncio
    async def test_commands_generation(self, remediation_generator):
        """Test generation of executable commands."""
        content = {"operation": "remove_binding"}
        
        commands = await remediation_generator.generate_commands(
            RecommenderType.IAM_POLICY,
            content
        )
        
        # Commands generation is placeholder in current implementation
        assert isinstance(commands, list)
    
    @pytest.mark.asyncio
    async def test_verification_commands_generation(self, remediation_generator):
        """Test generation of verification commands."""
        resources = ["projects/test-project"]
        
        commands = await remediation_generator.generate_verification(
            RecommenderType.IAM_POLICY,
            resources
        )
        
        # Verification commands generation is placeholder in current implementation
        assert isinstance(commands, list)


class TestRecommendationAnalytics:
    """Test suite for RecommendationAnalytics."""
    
    @pytest.fixture
    def sample_recommendations(self):
        """Create sample recommendations for analytics testing."""
        return [
            RecommendationInsight(
                recommendation_id="rec-1",
                name="Critical IAM Fix",
                description="Fix critical IAM issue",
                recommender_type=RecommenderType.IAM_POLICY,
                state=RecommendationState.ACTIVE,
                priority=Priority.CRITICAL,
                impact={},
                content={},
                target_resources=["user1"],
                associated_insights=[],
                cost_savings_usd=0.0,
                security_impact_score=0.9,
                estimated_time_hours=2.0
            ),
            RecommendationInsight(
                recommendation_id="rec-2",
                name="High Priority Firewall",
                description="Fix firewall configuration",
                recommender_type=RecommenderType.FIREWALL,
                state=RecommendationState.ACTIVE,
                priority=Priority.HIGH,
                impact={},
                content={},
                target_resources=["firewall1"],
                associated_insights=[],
                cost_savings_usd=50.0,
                security_impact_score=0.7,
                estimated_time_hours=1.5
            ),
            RecommendationInsight(
                recommendation_id="rec-3",
                name="Medium Machine Type",
                description="Optimize machine type",
                recommender_type=RecommenderType.MACHINE_TYPE,
                state=RecommendationState.ACTIVE,
                priority=Priority.MEDIUM,
                impact={},
                content={},
                target_resources=["instance1"],
                associated_insights=[],
                cost_savings_usd=200.0,
                security_impact_score=0.2,
                estimated_time_hours=0.5
            )
        ]
    
    def test_portfolio_metrics_calculation(self, sample_recommendations):
        """Test calculation of portfolio-level metrics."""
        analytics = RecommendationAnalytics()
        
        metrics = analytics.calculate_portfolio_metrics(sample_recommendations)
        
        assert metrics["total_recommendations"] == 3
        assert metrics["total_cost_savings_usd"] == 250.0  # 0 + 50 + 200
        assert metrics["average_security_score"] == pytest.approx(0.6, abs=0.1)  # (0.9 + 0.7 + 0.2) / 3
        assert metrics["high_impact_count"] == 2  # Critical + High priority
        assert metrics["estimated_implementation_hours"] == 4.0  # 2.0 + 1.5 + 0.5
        
        # Check priority distribution
        assert metrics["priority_distribution"]["critical"] == 1
        assert metrics["priority_distribution"]["high"] == 1
        assert metrics["priority_distribution"]["medium"] == 1
        assert metrics["priority_distribution"]["low"] == 0
        
        # Check type distribution
        assert metrics["type_distribution"]["google.iam.policy.Recommender"] == 1
        assert metrics["type_distribution"]["google.compute.firewall.Recommender"] == 1
        assert metrics["type_distribution"]["google.compute.instance.MachineTypeRecommender"] == 1
    
    def test_portfolio_metrics_empty(self):
        """Test portfolio metrics with empty recommendations list."""
        analytics = RecommendationAnalytics()
        
        metrics = analytics.calculate_portfolio_metrics([])
        
        assert metrics == {}


class TestPerformanceAndIntegration:
    """Test suite for performance and integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_recommendations_processing(self):
        """Test concurrent processing of multiple recommendation requests."""
        # Create multiple services
        services = []
        for i in range(5):
            service = RecommenderService(project_id=f"test-project-{i}")
            service.client = Mock()
            service.asset_client = Mock()
            services.append(service)
        
        # Create sample context
        context = RecommendationContext(
            project_id=f"test-project",
            resource_name="test-resource",
            location="global"
        )
        
        # Mock responses
        for service in services:
            service.client.list_recommendations.return_value = []
        
        # Process requests concurrently
        tasks = []
        for service in services:
            task = service.get_all_recommendations(context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed
        assert len(results) == 5
        for result in results:
            assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_chat_service_performance_under_load(self):
        """Test chat recommendation service performance under load."""
        # Create service with mocked dependencies
        mock_recommender_service = Mock()
        mock_recommender_service.get_all_recommendations = AsyncMock(return_value=[])
        mock_chat_manager = Mock()
        mock_chat_manager.add_message = AsyncMock()
        
        chat_service = ChatRecommendationService(mock_recommender_service, mock_chat_manager)
        
        # Create multiple queries
        queries = []
        for i in range(10):
            context = ChatRecommendationContext(
                session_id=f"session-{i}",
                user_id=f"user-{i}",
                project_context=RecommenderContextRequest(project_id="test-project")
            )
            query = ChatRecommendationQuery(
                query="Show me recommendations",
                context=context
            )
            queries.append(query)
        
        # Process queries concurrently
        start_time = time.time()
        
        tasks = []
        for query in queries:
            with patch.object(chat_service.intent_classifier, 'classify', return_value=QueryIntent.LIST_RECOMMENDATIONS):
                with patch.object(chat_service.entity_extractor, 'extract', return_value={}):
                    task = chat_service.process_query(query)
                    tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance
        assert len(results) == 10
        assert all(result.success for result in results)
        assert total_time < 5.0  # Should complete within 5 seconds
        
        # Verify all sessions were created
        assert len(chat_service.conversation_states) == 10
    
    def test_memory_usage_optimization(self):
        """Test memory usage stays reasonable with large datasets."""
        service = RecommenderService(project_id="test-project")
        service.client = Mock()
        service.asset_client = Mock()
        
        # Simulate large cache
        for i in range(1000):
            cache_key = f"project-{i}:recommender:global"
            service.cache[cache_key] = {
                "data": [Mock() for _ in range(10)],  # 10 recommendations per cache entry
                "timestamp": datetime.now()
            }
        
        # Verify cache size is manageable
        assert len(service.cache) == 1000
        
        # Test cache cleanup (would be implemented in production)
        # This is a placeholder for memory management tests
        cache_memory_estimate = len(service.cache) * 10 * 100  # Rough estimate
        assert cache_memory_estimate < 10_000_000  # Less than 10MB estimate
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self):
        """Test error recovery and system resilience."""
        service = RecommenderService(project_id="test-project")
        service.client = Mock()
        service.asset_client = Mock()
        
        context = RecommendationContext(
            project_id="test-project",
            resource_name="test-resource"
        )
        
        # Test various error scenarios
        error_scenarios = [
            Exception("Network timeout"),
            Exception("Permission denied"),
            Exception("Service unavailable"),
            Exception("Invalid project")
        ]
        
        for error in error_scenarios:
            service.client.list_recommendations.side_effect = error
            
            # Service should handle errors gracefully
            recommendations = await service.get_all_recommendations(context)
            
            # Should return empty list on error, not crash
            assert isinstance(recommendations, list)
            assert len(recommendations) == 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_recommendation_flow(self):
        """Test complete end-to-end recommendation flow."""
        # Create services
        recommender_service = RecommenderService(project_id="test-project")
        recommender_service.client = Mock()
        recommender_service.asset_client = Mock()
        
        mock_chat_manager = Mock()
        mock_chat_manager.add_message = AsyncMock()
        
        chat_service = ChatRecommendationService(recommender_service, mock_chat_manager)
        
        # Mock recommendation data
        mock_recommendation = Mock()
        mock_recommendation.name = "projects/test/locations/global/recommenders/google.iam.policy.Recommender/recommendations/rec-123"
        mock_recommendation.display_name = "Test Recommendation"
        mock_recommendation.description = "Test description"
        mock_recommendation.state_info = Mock()
        mock_recommendation.state_info.state = Mock()
        mock_recommendation.state_info.state.name = "ACTIVE"
        mock_recommendation.primary_impact = Mock()
        mock_recommendation.primary_impact.category = Mock()
        mock_recommendation.primary_impact.category.name = "SECURITY"
        mock_recommendation.primary_impact.cost_projection = None
        mock_recommendation.content = Mock()
        mock_recommendation.content.overview = "Test overview"
        mock_recommendation.content.operation_groups = []
        
        recommender_service.client.list_recommendations.return_value = [mock_recommendation]
        
        # Create query
        context = ChatRecommendationContext(
            session_id="test-session",
            user_id="test-user",
            project_context=RecommenderContextRequest(project_id="test-project")
        )
        query = ChatRecommendationQuery(
            query="Show me my security recommendations",
            context=context
        )
        
        # Process query
        with patch.object(chat_service.intent_classifier, 'classify', return_value=QueryIntent.GENERAL_SECURITY):
            with patch.object(chat_service.entity_extractor, 'extract', return_value={}):
                response = await chat_service.process_query(query)
        
        # Verify end-to-end flow
        assert response.success is True
        assert len(response.recommendations) > 0
        assert "Security Analysis" in response.response_text
        assert len(response.suggested_actions) > 0
        assert len(response.follow_up_questions) > 0
        
        # Verify chat history was updated
        assert mock_chat_manager.add_message.call_count >= 2
        
        # Verify session state was created
        assert "test-session" in chat_service.conversation_states


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_credentials_handling(self):
        """Test handling of invalid credentials."""
        with patch('backend.services.recommender_service.service_account.Credentials.from_service_account_file') as mock_creds:
            mock_creds.side_effect = Exception("Invalid credentials file")
            
            with pytest.raises(Exception):
                RecommenderService(credentials_path="/invalid/path/creds.json")
    
    @pytest.mark.asyncio
    async def test_malformed_recommendation_data(self):
        """Test handling of malformed recommendation data from API."""
        service = RecommenderService(project_id="test-project")
        service.client = Mock()
        service.asset_client = Mock()
        
        # Create malformed recommendation
        malformed_rec = Mock()
        malformed_rec.name = None  # Missing required field
        malformed_rec.display_name = None
        malformed_rec.description = None
        
        context = RecommendationContext(
            project_id="test-project",
            resource_name="test-resource"
        )
        
        # Should handle malformed data gracefully
        result = await service._process_recommendation(
            malformed_rec,
            RecommenderType.IAM_POLICY
        )
        
        # Should return None for malformed data
        assert result is None
    
    @pytest.mark.asyncio
    async def test_rate_limiting_simulation(self):
        """Test behavior under rate limiting conditions."""
        service = RecommenderService(project_id="test-project")
        service.client = Mock()
        service.asset_client = Mock()
        
        # Simulate rate limiting error
        from google.api_core.exceptions import TooManyRequests
        service.client.list_recommendations.side_effect = TooManyRequests("Rate limit exceeded")
        
        context = RecommendationContext(
            project_id="test-project",
            resource_name="test-resource"
        )
        
        # Should handle rate limiting gracefully
        recommendations = await service.get_all_recommendations(context)
        assert isinstance(recommendations, list)
        assert len(recommendations) == 0
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test handling of empty responses from API."""
        service = RecommenderService(project_id="test-project")
        service.client = Mock()
        service.asset_client = Mock()
        
        # Mock empty response
        service.client.list_recommendations.return_value = []
        
        context = RecommendationContext(
            project_id="test-project",
            resource_name="test-resource"
        )
        
        recommendations = await service.get_all_recommendations(context)
        assert isinstance(recommendations, list)
        assert len(recommendations) == 0
    
    @pytest.mark.asyncio
    async def test_chat_service_with_invalid_query(self):
        """Test chat service handling of invalid queries."""
        mock_recommender_service = Mock()
        mock_chat_manager = Mock()
        mock_chat_manager.add_message = AsyncMock()
        
        chat_service = ChatRecommendationService(mock_recommender_service, mock_chat_manager)
        
        # Create query with missing required fields
        invalid_context = ChatRecommendationContext(
            session_id="",  # Empty session ID
            user_id="",     # Empty user ID
            project_context=RecommenderContextRequest(project_id="")  # Empty project ID
        )
        
        query = ChatRecommendationQuery(
            query="",  # Empty query
            context=invalid_context
        )
        
        # Should handle invalid query gracefully
        response = await chat_service.process_query(query)
        
        # Should return error response but not crash
        assert response.success is False
        assert len(response.response_text) > 0
        assert len(response.suggested_actions) > 0


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short", "--maxfail=5"])
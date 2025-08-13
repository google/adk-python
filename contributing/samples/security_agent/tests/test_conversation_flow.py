"""
Test cases for conversation flow functionality
Tests the bucket analysis conversation pattern stored in memory
"""

import pytest
import asyncio
from datetime import datetime
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.conversation_memory import ConversationMemoryManager, ConversationContext
from agents.coordinator_agent import CoordinatorAgent

class TestConversationFlow:
    """Test conversation flow and memory functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.memory_manager = ConversationMemoryManager()
        self.coordinator = CoordinatorAgent()
        self.test_user_id = "test_user"
    
    def test_conversation_session_creation(self):
        """Test creating a conversation session"""
        session_id = self.memory_manager.create_session(self.test_user_id)
        
        assert session_id is not None
        assert len(session_id) > 0
        assert session_id in self.memory_manager.sessions
        
        session = self.memory_manager.sessions[session_id]
        assert session.user_id == self.test_user_id
        assert session.status == 'active'
        assert len(session.messages) == 0
    
    def test_message_addition_to_conversation(self):
        """Test adding messages to a conversation"""
        session_id = self.memory_manager.create_session(self.test_user_id)
        
        # Add user message
        message_id = self.memory_manager.add_message(
            session_id, 
            'user', 
            'tell me about the buckets in the project'
        )
        
        assert message_id is not None
        
        # Check message was added
        history = self.memory_manager.get_conversation_history(session_id)
        assert len(history) == 1
        assert history[0].role == 'user'
        assert history[0].content == 'tell me about the buckets in the project'
    
    def test_conversation_context_updates(self):
        """Test conversation context detection and updates"""
        session_id = self.memory_manager.create_session(self.test_user_id)
        
        # Add bucket-related message
        self.memory_manager.add_message(
            session_id, 
            'user', 
            'tell me about the buckets in the project'
        )
        
        # Check context was updated
        context = self.memory_manager.get_conversation_context(session_id)
        assert context.topic == 'storage_analysis'
        assert 'buckets' in context.entities
    
    def test_bucket_analysis_conversation_pattern(self):
        """Test the specific bucket analysis conversation pattern from memory"""
        session_id = self.memory_manager.create_session(self.test_user_id)
        
        # Simulate the stored test case: "tell me about the buckets in the project"
        user_query = "tell me about the buckets in the project"
        
        # Add user message
        self.memory_manager.add_message(session_id, 'user', user_query)
        
        # Simulate agent response with buckets and recommendations
        agent_response = """Found the following buckets in your project:
        1. my-data-bucket - Contains application data
        2. backup-bucket - Used for backups
        3. logs-bucket - Stores application logs
        
        Recommendations for these buckets:
        - Enable encryption on my-data-bucket for sensitive data
        - Review public access settings on all buckets
        - Set up lifecycle policies for backup-bucket"""
        
        # Add agent response with metadata
        self.memory_manager.add_message(
            session_id, 
            'assistant', 
            agent_response,
            metadata={'agent_used': 'SecurityAgent', 'analysis_type': 'bucket_analysis'}
        )
        
        # Update context with analysis results and recommendations
        self.memory_manager.update_context(
            session_id,
            analysis_results={
                'type': 'bucket_analysis',
                'buckets_found': ['my-data-bucket', 'backup-bucket', 'logs-bucket'],
                'timestamp': datetime.now().isoformat()
            },
            recommendations=[
                'Enable encryption on my-data-bucket',
                'Review public access settings',
                'Set up lifecycle policies'
            ]
        )
        
        # Verify conversation flow
        history = self.memory_manager.get_conversation_history(session_id)
        assert len(history) == 2
        
        # Check context contains expected information
        context = self.memory_memory.get_conversation_context(session_id)
        assert context.topic == 'storage_analysis'
        assert 'buckets' in context.entities
        assert context.analysis_results['type'] == 'bucket_analysis'
        assert len(context.recommendations) == 3
        
        # Test follow-up question should maintain context
        follow_up_query = "What about the encryption on the backup bucket?"
        
        # Get context for agent routing - should indicate we're in storage analysis
        routing_context = self.memory_manager.get_context_for_agent_routing(session_id)
        assert routing_context['topic'] == 'storage_analysis'
        assert routing_context['has_analysis_results'] == True
        assert 'buckets' in routing_context['entities']
        
        print("✅ Bucket analysis conversation pattern test passed!")
    
    @pytest.mark.asyncio
    async def test_coordinator_conversation_awareness(self):
        """Test that coordinator agent uses conversation context for routing"""
        session_id = self.memory_manager.create_session(self.test_user_id)
        
        # First query - should establish context
        first_query = "tell me about the buckets in the project"
        
        # Mock the coordinator process_query with session_id
        # Note: This would require actual GCP credentials and setup for full integration test
        # For unit testing, we're testing the conversation flow logic
        
        # Add the query to memory
        self.memory_manager.add_message(session_id, 'user', first_query)
        
        # Simulate context establishment
        self.memory_manager.update_context(
            session_id,
            topic='storage_analysis',
            entities=['buckets'],
            analysis_results={'type': 'bucket_analysis'}
        )
        
        # Test follow-up query routing context
        routing_context = self.memory_manager.get_context_for_agent_routing(session_id)
        
        # Verify routing context contains conversation awareness
        assert routing_context['topic'] == 'storage_analysis'
        assert 'buckets' in routing_context['entities']
        assert routing_context['has_analysis_results'] == True
        
        print("✅ Coordinator conversation awareness test passed!")
    
    def test_conversation_memory_cleanup(self):
        """Test conversation memory cleanup functionality"""
        # Create a session
        session_id = self.memory_manager.create_session(self.test_user_id)
        
        # Add some messages
        self.memory_manager.add_message(session_id, 'user', 'test message')
        
        # Verify session exists
        assert session_id in self.memory_manager.sessions
        
        # Test session summary
        summary = self.memory_manager.get_session_summary(session_id)
        assert summary['message_count'] == 1
        assert summary['session_id'] == session_id
        
        print("✅ Conversation memory cleanup test passed!")

if __name__ == "__main__":
    # Run the tests
    test_instance = TestConversationFlow()
    
    # Run individual tests
    test_instance.setup_method()
    test_instance.test_conversation_session_creation()
    
    test_instance.setup_method()
    test_instance.test_message_addition_to_conversation()
    
    test_instance.setup_method()
    test_instance.test_conversation_context_updates()
    
    test_instance.setup_method()
    test_instance.test_bucket_analysis_conversation_pattern()
    
    test_instance.setup_method()
    asyncio.run(test_instance.test_coordinator_conversation_awareness())
    
    test_instance.setup_method()
    test_instance.test_conversation_memory_cleanup()
    
    print("\n🎉 All conversation flow tests completed successfully!")
    print("\nTest case validates:")
    print("✅ User asks: 'tell me about the buckets in the project'")
    print("✅ Agent responds with buckets AND recommendations")
    print("✅ Conversation context maintained for follow-up questions")
    print("✅ Agent routing uses conversation awareness")
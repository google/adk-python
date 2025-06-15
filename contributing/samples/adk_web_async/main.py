"""
Test script for ADK Web async agent compatibility.

Simple test to verify the agent works correctly.
"""

import asyncio
import logging
import os

from agent import create_adk_web_agent, session_state_preprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_session_preprocessor():
    """Test the session state preprocessor."""
    logger.info("🧪 Testing session state preprocessor...")
    
    # Simulate ADK Web session state
    test_state = {"user_id": "1"}  # ADK Web default
    
    # Call preprocessor
    enhanced_state = await session_state_preprocessor(test_state)
    
    # Verify enhancement
    expected_keys = ["user_id", "user_name", "user_role", "app_name", "features"]
    for key in expected_keys:
        assert key in enhanced_state, f"Missing key: {key}"
    
    logger.info(f"✅ Session state enhanced: {enhanced_state}")


def test_agent_creation():
    """Test agent creation for ADK Web."""
    logger.info("🧪 Testing agent creation...")
    
    # Create agent (simulates ADK Web calling create_adk_web_agent)
    agent = create_adk_web_agent()
    
    assert agent is not None, "Agent creation failed"
    assert agent.name == "adk_web_async_demo"
    
    logger.info(f"✅ Agent created: {agent.name}")
    return agent


async def main():
    """Run all tests."""
    logger.info("🚀 Starting ADK Web async compatibility tests...")
    
    try:
        # Test session preprocessor
        await test_session_preprocessor()
        
        # Test agent creation
        agent = test_agent_creation()
        
        logger.info("🎉 All tests passed!")
        logger.info("💡 Ready for ADK Web: adk web --port 8088")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
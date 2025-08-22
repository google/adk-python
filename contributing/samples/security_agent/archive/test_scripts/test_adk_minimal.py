#!/usr/bin/env python3
"""
Minimal ADK Agent Test Script

This script tests the most basic ADK agent configuration to isolate 
the 'model_copy' context initialization error.
"""

import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    print("⚠️ No .env file found")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test ADK imports
try:
    from google.adk import Agent
    from google.adk.tools import FunctionTool
    print("✅ ADK imports successful")
except ImportError as e:
    print(f"❌ ADK import failed: {e}")
    exit(1)

def simple_tool() -> str:
    """A simple test tool that returns a basic response."""
    return "✅ Simple tool executed successfully!"

def test_minimal_agent():
    """Test a minimal ADK agent with a single simple tool."""
    
    # Create the simplest possible agent
    agent = Agent(
        name="test_agent",
        model="gemini-2.0-flash-exp",
        instruction="You are a test agent. Use the simple_tool when asked to test.",
        tools=[
            FunctionTool(simple_tool)
        ]
    )
    
    print("✅ Agent created successfully")
    return agent

async def test_agent_execution():
    """Test agent execution to identify the context error."""
    try:
        agent = test_minimal_agent()
        
        # Test query
        query = "Please run the simple tool to test functionality."
        
        print(f"🧪 Testing agent with query: '{query}'")
        
        # This is where the error occurs
        response_parts = []
        async for chunk in agent.run_async(query):
            print(f"📨 Received chunk type: {type(chunk)}")
            if isinstance(chunk, str):
                response_parts.append(chunk)
                print(f"📝 Chunk content: {chunk[:100]}...")
            else:
                print(f"🔍 Non-string chunk: {chunk}")
        
        response = ''.join(response_parts)
        print(f"✅ Agent execution successful!")
        print(f"📋 Response: {response}")
        
        return response
        
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function."""
    print("🚀 Starting minimal ADK agent test...")
    print("=" * 60)
    
    # Test basic ADK functionality
    response = await test_agent_execution()
    
    print("=" * 60)
    if response:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed - need to investigate ADK setup")

if __name__ == "__main__":
    asyncio.run(main())
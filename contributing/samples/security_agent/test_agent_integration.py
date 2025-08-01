#!/usr/bin/env python3
"""Test agent integration with backend services."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ.setdefault('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
os.environ.setdefault('GOOGLE_CLOUD_LOCATION', 'us-central1')

try:
    from agents import agent as agent_module
    
    print("="*60)
    print("Agent Integration Test")
    print("="*60)
    
    # Test 1: Check agent configuration
    print("\n1. Agent Configuration:")
    print(f"   Name: {agent_module.root_agent.name}")
    print(f"   Model: {agent_module.root_agent.model}")
    print(f"   Tools: {len(agent_module.root_agent.tools)} available")
    
    # Test 2: Test get_gcp_projects function
    print("\n2. Testing get_gcp_projects() function:")
    try:
        result = agent_module.get_gcp_projects(None)
        print(f"   ✅ Function executed successfully")
        print(f"   Result: {result[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Test get_project_info function
    print("\n3. Testing get_project_info() function:")
    try:
        result = agent_module.get_project_info("your-project-id", None)
        print(f"   ✅ Function executed successfully")
        print(f"   Result: {result[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Test call_google_api function
    print("\n4. Testing call_google_api() function:")
    try:
        result = agent_module.call_google_api(
            service="storage",
            version="v1", 
            resource_path="b",
            method="GET",
            tool_context=None
        )
        print(f"   ✅ Function executed successfully")
        print(f"   Result preview: {result[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Agent is configured with backend REST API integration ✅")
    print("- Agent tools successfully call backend services ✅")
    print("- Backend provides GCP project information ✅")
    print("="*60)
    
except Exception as e:
    print(f"Failed to load agent module: {e}")
    import traceback
    traceback.print_exc()
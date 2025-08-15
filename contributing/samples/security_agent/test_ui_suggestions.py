#!/usr/bin/env python3
"""
Test script to verify that the UI suggestions functionality works correctly
"""

import requests
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_suggestions_api():
    """Test that the backend API returns suggestions"""
    print("🧪 Testing Backend API for Suggestions")
    print("=" * 50)
    
    try:
        # Test API call
        response = requests.post(
            'http://localhost:8000/api/v1/agent/chat',
            json={
                'query': 'Tell me about my bucket security issues',
                'user_id': 'test_user',
                'project_id': 'mgm-digitalconcierge'
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Status: {response.status_code}")
            print(f"✅ Success: {data.get('success', False)}")
            print(f"✅ Agent: {data.get('agent_used', 'Unknown')}")
            
            suggestions = data.get('suggestions', [])
            print(f"✅ Suggestions Count: {len(suggestions)}")
            
            if suggestions:
                print("✅ Suggestions Generated:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"   {i}. {suggestion}")
                return True, suggestions
            else:
                print("❌ No suggestions returned")
                return False, []
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False, []
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, []

def test_streamlit_ui_simulation():
    """Simulate how Streamlit would process the suggestions"""
    print("\n🎯 Testing Streamlit UI Simulation")
    print("=" * 50)
    
    # Get suggestions from API
    success, suggestions = test_suggestions_api()
    
    if not success:
        print("❌ Cannot test UI - API failed")
        return False
    
    # Simulate Streamlit session state
    class MockSessionState:
        def __init__(self):
            self.current_suggestions = []
            self.chat_messages = []
    
    session_state = MockSessionState()
    
    # Simulate the render_suggestions function logic
    if suggestions:
        print(f"🎯 render_suggestions called with {len(suggestions)} suggestions")
        session_state.current_suggestions = suggestions
        
        print("💡 Follow-up Questions would be displayed:")
        print("*Click any suggestion to continue the conversation:*")
        
        # Simulate button creation (max 5 suggestions, 2 per row)
        suggestions_to_show = suggestions[:5]
        
        for i in range(0, len(suggestions_to_show), 2):
            row_suggestions = suggestions_to_show[i:i+2]
            print(f"\nRow {i//2 + 1}:")
            
            for j, suggestion in enumerate(row_suggestions):
                button_key = f"suggestion_{hash(suggestion)}_{i}_{j}"
                print(f"  [Button] ❓ {suggestion} (key: {button_key[:20]}...)")
        
        print(f"\n✅ UI Simulation successful! {len(suggestions_to_show)} buttons would be rendered")
        return True
    else:
        print("❌ No suggestions to display")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing UI Suggestions Flow")
    print("=" * 60)
    
    # Test 1: Backend API
    api_success, suggestions = test_suggestions_api()
    
    # Test 2: UI simulation
    ui_success = test_streamlit_ui_simulation()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    print(f"✅ Backend API: {'PASS' if api_success else 'FAIL'}")
    print(f"✅ UI Simulation: {'PASS' if ui_success else 'FAIL'}")
    
    if api_success and ui_success:
        print("\n🎉 All tests passed! Suggestions should work in Streamlit UI")
        print("\n🎯 Next steps:")
        print("1. Open http://localhost:8501 in browser")
        print("2. Ask: 'Tell me about my bucket security issues'")
        print("3. Look for '💡 Follow-up Questions' section")
        print("4. Click any suggestion button to continue conversation")
        return True
    else:
        print("\n❌ Tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
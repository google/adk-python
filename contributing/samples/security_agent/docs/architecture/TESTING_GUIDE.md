# ADK Session Management Testing Guide

This guide provides comprehensive testing procedures for the ADK session management system.

## Prerequisites

1. **Install Dependencies**:
   ```bash
   cd /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent
   pip install fastapi uvicorn streamlit requests pydantic
   ```

2. **Set Environment Variables**:
   ```bash
   export GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json  # optional
   ```

## Test Methods

### Method 1: Automated Test Script (Recommended)

Run the comprehensive test script that validates the entire session flow:

```bash
# Make the test script executable
chmod +x test_chat_session.py

# Run the test
python test_chat_session.py
```

**What this tests:**
- ✅ Backend health check
- ✅ Session creation via API
- ✅ Chat message processing with session
- ✅ Message retrieval from session
- ✅ Session analytics
- ✅ Session continuity across messages

### Method 2: Manual Backend Testing

1. **Start the Backend**:
   ```bash
   cd /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent
   python run_backend.py
   ```
   
   You should see:
   ```
   ✅ LLM Agent router included at /api/v1/agent (with intelligent steering)
   ✅ Sessions router included at /api/v1/sessions (ADK thin client support)
   INFO:     Application startup complete.
   ```

2. **Test Session Creation**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/sessions/create" \
        -H "Content-Type: application/json" \
        -d '{"user_id": "test_user", "project_id": "test-project"}'
   ```
   
   Expected response:
   ```json
   {
     "success": true,
     "session_id": "test_user_1697123456_abc12345",
     "user_id": "test_user",
     "created_at": "2023-10-12T15:30:56.789Z"
   }
   ```

3. **Test Chat with Session**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/agent/chat" \
        -H "Content-Type: application/json" \
        -d '{
          "query": "Tell me about storage buckets",
          "user_id": "test_user",
          "session_id": "test_user_1697123456_abc12345",
          "project_id": "test-project"
        }'
   ```

### Method 3: Frontend Testing

1. **Start the Backend** (if not already running):
   ```bash
   python run_backend.py
   ```

2. **Start the Frontend**:
   ```bash
   python run_frontend.py
   ```

3. **Test Session Flow in UI**:
   - Navigate to the chat interface
   - Check that session info is displayed in the expandable section
   - Send a message: "Tell me about buckets in my project"
   - Verify response appears
   - Send follow-up: "How do I fix public access?"
   - Verify conversation continuity

4. **Test Session Management**:
   - Click "Clear Chat" - should clear messages but keep session
   - Click "New Session" - should create fresh session

### Method 4: Unit Test Suite

Run the comprehensive test suite:

```bash
cd tests
python test_session_flow.py
```

This runs detailed unit tests including:
- Module import validation
- Chat manager functionality
- Session API logic
- Complete conversation flows
- Session restoration after interruption

## Expected Behavior

### ✅ Working System Indicators

1. **Backend Startup**:
   ```
   ✅ Enhanced chat manager loaded
   ✅ LLM Agent router included at /api/v1/agent
   ✅ Sessions router included at /api/v1/sessions
   ```

2. **Frontend Session Display**:
   - Session ID shown in expandable panel
   - Status shows as "🔴 Active"
   - Message count increases with conversation

3. **API Responses**:
   - Session creation returns valid session_id
   - Chat responses include same session_id
   - Messages are retrievable via sessions API

4. **Conversation Continuity**:
   - Follow-up questions reference previous context
   - Session maintains state across multiple interactions
   - Agent routing works correctly

### ❌ Common Issues

1. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'fastapi'
   ```
   **Solution**: Install dependencies with `pip install fastapi uvicorn`

2. **Session Not Persisting**:
   - Check backend logs for session creation
   - Verify session_id is being passed in requests
   - Ensure chat_manager is properly imported

3. **Frontend Session Issues**:
   - Check browser developer console for API errors
   - Verify backend is running on port 8000
   - Look for session initialization messages in backend logs

4. **Memory/Context Issues**:
   - Verify conversation_memory service is available
   - Check for topic detection in chat manager logs

## Debug Mode

Enable detailed logging by setting:

```bash
export LOG_LEVEL=DEBUG
```

Then restart both backend and frontend to see detailed session flow logs.

## Performance Testing

For load testing the session system:

```bash
# Run multiple concurrent sessions
for i in {1..5}; do
  python test_chat_session.py &
done
wait
```

## Troubleshooting

### Backend Issues

1. **Check port availability**:
   ```bash
   lsof -i :8000
   ```

2. **Verify imports work**:
   ```bash
   cd backend
   python -c "from chat_manager import chat_manager; print('✅ Chat manager works')"
   ```

3. **Test session creation directly**:
   ```python
   from backend.chat_manager import chat_manager
   session_id = chat_manager.create_session("debug_user")
   print(f"Created: {session_id}")
   ```

### Frontend Issues

1. **Check Streamlit logs**:
   ```bash
   streamlit run frontend/app.py --logger.level debug
   ```

2. **Verify API connectivity**:
   ```bash
   curl http://localhost:8000/api/v1/agent/
   ```

### Session Flow Issues

1. **Monitor session state**:
   - Check `st.session_state` in Streamlit debugger
   - Verify `adk_session_id` is set and persistent
   - Look for session restoration logs

2. **Verify message persistence**:
   - Send message and refresh page
   - Check if conversation history loads
   - Verify session analytics show correct message count

## Success Criteria

The system is working correctly when:

- ✅ All automated tests pass
- ✅ Sessions are created and persist across requests
- ✅ Messages are stored and retrievable
- ✅ Conversation context is maintained
- ✅ Frontend shows session information
- ✅ Chat responses include agent routing
- ✅ Session analytics provide meaningful data

## Next Steps

Once testing is complete:

1. **Deploy to staging environment**
2. **Run integration tests with real GCP data**
3. **Performance test with multiple concurrent users**
4. **Monitor session cleanup and memory usage**
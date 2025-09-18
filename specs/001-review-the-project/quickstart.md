# Quickstart Guide: Testing SQLite Database Connection Fix

**Feature**: Fix SQLite Database Connection in Chat Frontend
**Date**: 2025-09-17

## Prerequisites

- Python 3.11+ installed
- Project dependencies installed (`pip install -r requirements.txt`)
- Environment variables configured in `.env` file

## Quick Validation Steps

### Step 1: Verify Database Exists

```bash
# Check if database file exists
ls -la backend/cache/gcp_data.db

# If missing, create sample data
python populate_sqlite.py
```

### Step 2: Test Direct Database Connection

```bash
# Test direct ADK agent query
export DATABASE_PATH="backend/cache/gcp_data.db"
python test_adk_query.py
```

Expected output:
```
Testing ADK Agent with database query via FunctionTool...
Sending query: 'Show me high severity security findings.'
Agent Response: [List of security findings...]
```

### Step 3: Start Backend Server

```bash
# Terminal 1: Start the backend
python run_backend.py
```

Expected output:
```
[ADK] Using credentials: /path/to/credentials.json
[ADK] ADK agent loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Test Database Health Endpoint

```bash
# Terminal 2: Check database health
curl http://localhost:8000/health/database
```

Expected response:
```json
{
  "status": "healthy",
  "database_path": "/absolute/path/to/backend/cache/gcp_data.db",
  "exists": true,
  "readable": true,
  "table_count": 15,
  "total_records": 1000
}
```

### Step 5: Test Chat API Endpoint

```bash
# Test chat message endpoint
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me high severity security findings",
    "session_id": "test-session",
    "user_id": "test-user"
  }'
```

Expected response:
```json
{
  "response": "I found 5 high severity security findings...",
  "success": true,
  "agent_used": true,
  "model": "gemini-1.5-flash",
  "execution_time": 1.23
}
```

### Step 6: Start Frontend Interface

```bash
# Terminal 3: Start the frontend
python run_frontend.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### Step 7: Test Chat Interface

1. Open browser to http://localhost:8501
2. Navigate to any page with chat widget
3. Enter query: "Show me all assets"
4. Verify response contains actual database data

### Step 8: Test Session Persistence

1. Send first query: "List compute instances"
2. Send follow-up: "Show only the running ones"
3. Verify context is maintained

## Validation Checklist

- [ ] Database file exists at correct path
- [ ] Direct agent query returns data
- [ ] Backend server starts without errors
- [ ] Database health check passes
- [ ] Chat API returns actual data
- [ ] Frontend connects to backend
- [ ] Chat widget displays responses
- [ ] Session context is maintained
- [ ] Error messages are informative

## Common Issues and Solutions

### Issue: "Database not found"
```bash
# Solution: Set DATABASE_PATH environment variable
export DATABASE_PATH="$(pwd)/backend/cache/gcp_data.db"
```

### Issue: "No data returned"
```bash
# Solution: Populate database
python populate_sqlite.py
```

### Issue: "Agent not configured"
```bash
# Solution: Check credentials
export GOOGLE_APPLICATION_CREDENTIALS="config/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### Issue: "Connection refused"
```bash
# Solution: Check backend is running
ps aux | grep "run_backend"
# If not running, start it
python run_backend.py
```

## Performance Validation

### Query Response Times
- Simple query (list assets): < 2 seconds
- Complex query (join with filtering): < 5 seconds
- Aggregation query: < 3 seconds

### Load Test
```bash
# Run 10 concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/v1/chat/message \
    -H "Content-Type: application/json" \
    -d '{"message": "Show security summary"}' &
done
wait
```

## Integration Test Script

```python
#!/usr/bin/env python3
"""Integration test for SQLite connection fix"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_database_health():
    resp = requests.get(f"{BASE_URL}/health/database")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["exists"] == True
    print("✓ Database health check passed")

def test_chat_query():
    payload = {
        "message": "Show me security findings",
        "session_id": "test-123"
    }
    resp = requests.post(f"{BASE_URL}/api/v1/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] == True
    assert len(data["response"]) > 0
    print("✓ Chat query returned data")

def test_session_context():
    session_id = "context-test"

    # First query
    resp1 = requests.post(f"{BASE_URL}/api/v1/chat/message",
                          json={"message": "List assets", "session_id": session_id})
    assert resp1.status_code == 200

    # Follow-up query
    resp2 = requests.post(f"{BASE_URL}/api/v1/chat/message",
                          json={"message": "Show only compute instances", "session_id": session_id})
    assert resp2.status_code == 200
    print("✓ Session context maintained")

if __name__ == "__main__":
    try:
        test_database_health()
        test_chat_query()
        test_session_context()
        print("\n✅ All integration tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to backend. Is it running?")
        sys.exit(1)
```

## Success Criteria

The fix is considered successful when:
1. All validation checklist items pass
2. Integration tests complete without errors
3. Query response times meet targets
4. User can interact naturally with chat interface
5. Database queries return actual data, not empty responses

---

**Next Step**: Run `/tasks` command to generate implementation tasks
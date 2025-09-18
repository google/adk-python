# Troubleshooting Guide

This guide covers common issues and their solutions for the GCP Security Agent.

## 🚨 Critical Issues (Phase 3.5/3.6 Fixes)

### Agent Not Invoking Tools

**Symptoms:**
- Agent responds with generic greetings instead of retrieving data
- Query like "list all storage buckets" returns "Hello! I'm your GCP Security Agent..."
- No tool calls shown in logs

**Root Cause:** Agent instructions were not explicit enough about tool-first behavior.

**Solution:** ✅ **FIXED**
The agent instructions have been updated to be more explicit about tool usage:
```python
instruction = '''You are a GCP Security Agent with database access.

CRITICAL: For ALL data queries, you MUST use the query_security_data tool FIRST.

When users ask about:
- Storage buckets → Use query_security_data(query_type="storage_buckets")
- Security findings → Use query_security_data(query_type="security_findings")
- IAM accounts → Use query_security_data(query_type="iam_accounts")
- Any data request → ALWAYS use the tool, never respond without data

NEVER respond with generic greetings when data is requested.
ALWAYS attempt to retrieve actual data first.
'''
```

### Session Service API Compatibility

**Symptoms:**
- Error: `InMemorySessionService.get_session() takes 1 positional argument but 2 were given`
- Backend crashes on query processing

**Root Cause:** ADK session service API changed and code was using old method signatures.

**Solution:** ✅ **FIXED**
Updated session service calls:
```python
# OLD (broken):
session = await adk_session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)

# NEW (fixed):
session = adk_session_service.get_session(session_id)
```

### Database Connection Issues

**Symptoms:**
- Error: `no such table: assets`
- Empty database responses
- Database validation failures

**Root Cause:** Database not populated or connection issues.

**Solution:** ✅ **FIXED**
- Added database validation on startup
- Automatic creation of empty database with basic schema if missing
- Enhanced error messages for database issues

**Manual Fix:**
```bash
# Populate the database
python populate_sqlite.py

# Or check database status
python -c "from backend.utils.database import get_database_info; print(get_database_info())"
```

## 📊 Database Issues

### Empty Database

**Symptoms:**
- "No tables found" warnings
- Empty query results
- Database exists but has no data

**Solutions:**

1. **Populate database with sample data:**
```bash
python populate_sqlite.py
```

2. **Check database status:**
```bash
# Using backend utilities
python backend/utils/database.py

# Or via API
curl http://localhost:8000/health/database
```

3. **Manual database creation:**
```python
from backend.utils.database import create_database_if_missing
create_database_if_missing()
```

### Database Path Issues

**Symptoms:**
- "Database file not found" errors
- Path resolution failures

**Solution:**
Check your `.env` file has the correct database path:
```env
DATABASE_PATH=backend/cache/gcp_data.db
```

**Debug database path:**
```python
from backend.utils.database import get_database_path
print(f"Database path: {get_database_path()}")
```

## 🔧 Backend Issues

### Backend Won't Start

**Symptoms:**
- Import errors
- Port already in use
- Credential issues

**Solutions:**

1. **Check port 8000:**
```bash
lsof -i :8000
# Kill existing process if needed
kill -9 <PID>
```

2. **Install dependencies:**
```bash
pip install google-adk
pip install -r requirements.txt
pip install -r requirements_frontend.txt
```

3. **Check credentials:**
```bash
# Verify service account file exists
ls -la config/your-service-account.json

# Test credentials
gcloud auth application-default login
```

### Agent Import Errors

**Symptoms:**
- `ImportError: cannot import name 'security_agent'`
- Agent not available errors

**Solution:** ✅ **FIXED**
Updated import paths in backend:
```python
# backend/main.py and adk_wrapper.py now use:
from agents.adk_agent import root_agent as security_agent
```

### Session Management Errors

**Symptoms:**
- Session creation failures
- "Session management error" warnings

**Solution:** ✅ **FIXED**
- Updated to use new session service API
- Added graceful fallback for session errors
- Non-critical session errors no longer block queries

## 🖥️ Frontend Issues

### Streamlit Connection Errors

**Symptoms:**
- "Cannot connect to backend" errors
- Empty responses from chat widget
- Connection refused errors

**Solutions:**

1. **Start backend first:**
```bash
python run_backend.py
# Wait for "Application startup complete"
```

2. **Check backend URL:**
```python
# In frontend/utils/config.py
BACKEND_URL = "http://localhost:8000"  # Should match backend port
```

3. **Test backend connection:**
```bash
curl http://localhost:8000/health
```

### Chat Widget Errors

**Symptoms:**
- Generic error messages
- Poor error handling
- Unclear failure reasons

**Solution:** ✅ **FIXED**
Enhanced error messages now provide specific guidance:

- 📁 **Database Connection Issue** → Run `python populate_sqlite.py`
- 🔄 **Session Issue** → Refresh page or start new session
- ⏰ **Request Timeout** → Try simpler question or check backend load
- 🔧 **Tool Execution Issue** → Database may need refresh
- ❌ **Backend Connection Error** → Ensure backend running on port 8000

## ⚡ Performance Issues

### Slow Query Performance

**Symptoms:**
- Queries taking > 30 seconds
- Timeout errors
- Poor response times

**Solutions:** ✅ **FIXED**

1. **Query timeout handling:**
   - Automatic 30-second timeout on queries
   - Clear timeout error messages
   - Graceful fallback handling

2. **Performance monitoring:**
   - Real-time query performance tracking
   - Detailed metrics via `/api/v1/performance`
   - Request/response logging middleware

3. **Check performance metrics:**
```bash
curl http://localhost:8000/api/v1/performance
```

### Memory Issues

**Symptoms:**
- Backend using excessive memory
- Out of memory errors
- Slow response times

**Solutions:**

1. **Monitor agent memory:**
```bash
# Check memory usage
ps aux | grep python

# Restart backend if needed
python run_backend.py
```

2. **Reduce cache size:**
```python
# In backend/utils/performance.py
performance_monitor.max_cache_size = 500  # Reduce from 1000
```

## 🔐 Authentication Issues

### Service Account Errors

**Symptoms:**
- "Credentials not found" errors
- Permission denied errors
- Invalid service account errors

**Solutions:**

1. **Verify service account file:**
```bash
# Check file exists and has correct content
cat config/your-service-account.json | jq .project_id
```

2. **Set environment variable:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/config/your-service-account.json"
```

3. **Test credentials:**
```bash
gcloud auth activate-service-account --key-file=config/your-service-account.json
gcloud projects list
```

### ADK Authentication

**Symptoms:**
- "ADK authentication failed" errors
- Vertex AI permission errors

**Solutions:**

1. **Check environment variables:**
```bash
echo $GOOGLE_GENAI_USE_VERTEXAI  # Should be "1"
echo $GOOGLE_CLOUD_PROJECT       # Should be your project ID
echo $GOOGLE_CLOUD_LOCATION      # Should be your region
```

2. **Enable required APIs:**
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable cloudasset.googleapis.com
gcloud services enable securitycenter.googleapis.com
```

## 🧪 Testing & Validation

### Run Validation Suite

**Test the complete setup:**

1. **Test database:**
```bash
python -c "
from backend.utils.database import validate_database, get_database_info
valid, msg = validate_database()
print(f'Database valid: {valid}')
print(f'Message: {msg}')
print('Info:', get_database_info())
"
```

2. **Test backend APIs:**
```bash
# Health check
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# Performance metrics
curl http://localhost:8000/api/v1/performance

# Test query
curl -X POST http://localhost:8000/api/v1/database/test \
  -H "Content-Type: application/json" \
  -d '{"query_type": "statistics"}'
```

3. **Test ADK agent:**
```bash
python test_agent.py
```

4. **Test frontend:**
```bash
python run_frontend.py
# Visit http://localhost:8501
```

### Performance Benchmarks

**Expected performance targets:**
- Database queries: < 1 second
- Agent responses: < 10 seconds
- Frontend load time: < 3 seconds
- API endpoint response: < 500ms

**Run benchmarks:**
```bash
# API response time
time curl http://localhost:8000/health

# Database query performance
python -c "
import time
from agents.tools.sqlite_tool import query_security_data
start = time.time()
result = query_security_data('statistics')
print(f'Query time: {time.time() - start:.3f}s')
print(f'Success: {result.get(\"success\", False)}')
"
```

## 📝 Debugging Tips

### Enable Debug Logging

1. **Backend debug logs:**
```python
# In backend/main.py
logging.basicConfig(level=logging.DEBUG)
```

2. **Agent debug logs:**
```python
# In agents/adk_agent.py
logger.setLevel(logging.DEBUG)
```

3. **Frontend debug logs:**
```python
# In frontend/utils/config.py
DEBUG_MODE = True
LOG_AGENT_ACTIVITY = True
```

### Common Log Patterns

**Successful query:**
```
✅ Query agent_123 (storage_buckets) - 2.341s
🔧 Tool call detected: query_security_data
📁 Using SQLite cached data for storage buckets (cache hit)
```

**Failed query:**
```
❌ Query agent_456 (assets) - 0.123s
Database error: no such table: assets
💡 Run 'python populate_sqlite.py' to populate the database
```

**Tool invocation working:**
```
🚀 Sending message to ADK agent...
🔧 Tool call detected: query_security_data
✅ ADK QUERY COMPLETED SUCCESSFULLY
```

## 🆘 Getting Help

### Escalation Path

1. **Check this troubleshooting guide** ← You are here
2. **Review logs** in `backend_logs.txt` and `frontend_logs.txt`
3. **Run validation suite** (see Testing section above)
4. **Check GitHub issues** for similar problems
5. **Create issue** with logs and error details

### Information to Include

When reporting issues, include:

- ✅ **Error messages** (exact text)
- ✅ **Log output** (last 50 lines)
- ✅ **Environment details** (Python version, OS, ADK version)
- ✅ **Reproduction steps** (what you did)
- ✅ **Expected vs actual behavior**

**Quick environment info:**
```bash
python --version
pip show google-adk
echo "OS: $(uname -s)"
echo "Project: $GOOGLE_CLOUD_PROJECT"
ls -la config/*.json
```

## 🔄 Recent Fixes (Phase 3.5/3.6)

The following critical issues have been resolved:

- ✅ **Agent tool invocation** - Fixed agent instructions for tool-first behavior
- ✅ **Session service compatibility** - Updated to new ADK session API
- ✅ **Database validation** - Added startup validation and auto-creation
- ✅ **Request timeout handling** - 30-second timeout with clear error messages
- ✅ **Performance monitoring** - Real-time metrics and logging
- ✅ **Enhanced error messages** - User-friendly error guidance in chat widget
- ✅ **Import path fixes** - Resolved agent import issues
- ✅ **Graceful fallbacks** - Better error handling throughout the stack

These fixes address the core issues identified in the logs where agents were not invoking tools properly and sessions were failing.
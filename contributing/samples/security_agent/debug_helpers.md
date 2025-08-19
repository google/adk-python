# 🐛 Security Agent Debug Helpers

## Log Locations
- **Backend logs**: `logs/backend.log`
- **Frontend logs**: `logs/frontend.log`

## Real-time Log Monitoring
```bash
# Monitor all logs in real-time
./monitor_logs.sh

# Or monitor individually:
tail -f logs/backend.log
tail -f logs/frontend.log
```

## Quick Debug Commands

### Check Services Status
```bash
# Check backend health
curl -s http://localhost:8002/health | jq .

# Check running processes
ps aux | grep -E "(uvicorn|streamlit)"

# Check log file sizes
ls -la logs/
```

### Test Individual Functions
```bash
source venv/bin/activate

# Test agent functions directly
python -c "import agent; print(agent.analyze_iam())"
python -c "import agent; print(agent.analyze_storage())"
python -c "import agent; print(agent.discover_assets())"
```

### API Testing
```bash
# Test session creation
curl -X POST "http://localhost:8002/api/v1/sessions/api/v1/sessions/create" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "debug-user"}' | jq .

# Test IAM analysis
curl -s http://localhost:8002/api/v1/iam/analyze | jq . | head -20

# Test storage analysis  
curl -s "http://localhost:8002/api/v1/storage/analyze/mgm-digitalconcierge" | jq . | head -20
```

## URLs
- **Frontend UI**: http://localhost:8503
- **Backend API Docs**: http://localhost:8002/docs
- **Backend Health**: http://localhost:8002/health

## Common Issues & Solutions

### Port Already in Use
```bash
# Kill existing processes
pkill -f uvicorn
pkill -f streamlit

# Find what's using a port
lsof -i :8503
```

### Clear Logs
```bash
> logs/backend.log
> logs/frontend.log
```

### Start/Restart Services (Standard Method)
```bash
# Backend (port 8000)
python run_backend.py

# Frontend (port 8501)  
python run_frontend.py
```

## Log Patterns to Watch For

### Normal Operations
```
✅ Backend configured as API service
✅ Sessions router included at /api/v1/sessions
✅ Security router included at /api/v1/security
✅ IAM router included at /api/v1/iam
✅ Storage router included at /api/v1/storage
```

### Error Patterns
```
ERROR: - Authentication issues
WARNING: - Missing dependencies (expected)
HTTP Request: - API call tracking
```

### Performance Monitoring
```
INFO:httpx:HTTP Request: GET http://localhost:8002/api/v1/iam/analyze "HTTP/1.1 200 OK"
```
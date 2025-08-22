# Troubleshooting Guide

## Common Issues and Solutions

### Backend Issues

#### Issue: Backend won't start
**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
pip install -r requirements.txt
```

---

#### Issue: Port 8000 already in use
**Error:** `[Errno 48] Address already in use`

**Solution:**
```bash
# Find the process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
BACKEND_PORT=8001 python run_backend.py
```

---

#### Issue: Database not found
**Error:** `Database not found at backend/cache/gcp_data.db`

**Solution:**
```bash
# Create the cache directory
mkdir -p backend/cache

# Run data population
python backend/services/populate_sqlite.py

# Or set absolute path in .env
DATABASE_PATH=/absolute/path/to/gcp_data.db
```

---

### Frontend Issues

#### Issue: Streamlit won't start
**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
pip install streamlit google-genai
```

---

#### Issue: Agent not appearing in ADK web dropdown
**Error:** Wrong agent shows in dropdown

**Solution:**
```bash
# Start ADK web from the agent directory
cd agents/gcp_security
adk web
```

---

#### Issue: Token streaming not working
**Error:** Responses appear all at once instead of streaming

**Solution:**
1. Ensure you're using the correct streaming client:
   ```bash
   python run_frontend.py  # Uses unified_streaming_client.py
   ```

2. Check agent configuration:
   ```python
   # Agent should use Runner with proper session management
   runner = Runner(
       app_name="gcp_security_agent",
       agent=root_agent,
       session_service=InMemorySessionService()
   )
   ```

---

### GCP Authentication Issues

#### Issue: Application Default Credentials not found
**Error:** `google.auth.exceptions.DefaultCredentialsError`

**Solution:**
```bash
# Option 1: Set service account key path
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Option 2: Use gcloud auth
gcloud auth application-default login

# Option 3: Add to .env file
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

---

#### Issue: Insufficient permissions
**Error:** `403 Permission denied`

**Solution:**
Ensure your service account has these IAM roles:
- Cloud Asset Viewer
- Security Center Admin Viewer
- Storage Admin
- IAM Security Reviewer
- Recommender Viewer
- Secret Manager Viewer
- Monitoring Viewer

Grant roles:
```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SA_EMAIL" \
    --role="roles/cloudasset.viewer"
```

---

### Docker Issues

#### Issue: Docker build fails
**Error:** `Package installation failed`

**Solution:**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

---

#### Issue: Container can't access GCP
**Error:** `Credentials not found in container`

**Solution:**
1. Mount credentials in docker-compose.yml:
   ```yaml
   volumes:
     - ${GOOGLE_APPLICATION_CREDENTIALS}:/app/credentials/key.json:ro
   environment:
     - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json
   ```

2. Ensure .env file exists:
   ```bash
   cp .env.template .env
   # Edit .env with your values
   ```

---

### Performance Issues

#### Issue: Slow response times
**Symptoms:** API calls taking > 5 seconds

**Solution:**
1. Check cache status:
   ```bash
   curl http://localhost:8000/status | jq .database
   ```

2. Refresh cache:
   ```bash
   python backend/services/data_fetcher.py
   ```

3. Optimize database:
   ```bash
   sqlite3 backend/cache/gcp_data.db "VACUUM;"
   ```

---

#### Issue: High memory usage
**Symptoms:** Memory usage > 4GB

**Solution:**
1. Check for memory leaks:
   ```bash
   curl http://localhost:8000/metrics | grep memory
   ```

2. Restart services:
   ```bash
   docker-compose restart
   ```

3. Adjust cache settings in .env:
   ```
   DATA_REFRESH_INTERVAL=3600
   CACHE_MAX_SIZE=1000
   ```

---

### Monitoring Issues

#### Issue: Metrics endpoint not working
**Error:** `404 Not Found on /metrics`

**Solution:**
Ensure you're running the updated backend:
```bash
# Pull latest changes
git pull

# Restart backend
python run_backend.py
```

---

#### Issue: Health check failing
**Error:** `Service unhealthy`

**Solution:**
1. Check individual components:
   ```bash
   curl http://localhost:8000/status | jq
   ```

2. Review logs:
   ```bash
   docker-compose logs backend
   ```

3. Verify database connectivity:
   ```bash
   sqlite3 backend/cache/gcp_data.db ".tables"
   ```

---

## Debug Mode

Enable debug logging for more information:

```bash
# In .env file
LOG_LEVEL=DEBUG

# Or via environment variable
LOG_LEVEL=DEBUG python run_backend.py
```

## Getting Help

If you're still experiencing issues:

1. Check the logs:
   ```bash
   tail -f logs/application.log
   ```

2. Run the evaluation suite:
   ```bash
   cd evaluation
   python service_evaluation_orchestrator.py
   ```

3. Create an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version)
   - Relevant log output
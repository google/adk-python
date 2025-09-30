# Confluence Tool Troubleshooting Guide

## Common Issues and Solutions

### 🔐 Authentication Issues

#### Problem: "401 Unauthorized" or "Authentication Failed"
**Symptoms:**
- Connection test fails with 401 error
- "Invalid credentials" messages
- API requests return authentication errors

**Solutions:**
1. **Verify API Token**
   ```bash
   # Test token manually
   curl -u your-email@example.com:your-api-token \
     https://your-domain.atlassian.net/wiki/rest/api/space
   ```

2. **Check Environment Variables**
   ```bash
   # Verify variables are set correctly
   echo $CONFLUENCE_URL
   echo $CONFLUENCE_USERNAME
   echo $CONFLUENCE_API_TOKEN
   ```

3. **Regenerate API Token**
   - Go to https://id.atlassian.com/manage-profile/security/api-tokens
   - Delete old token and create new one
   - Update `.env` file immediately

**Common Mistakes:**
- Using password instead of API token
- Incorrect email format (must match Atlassian account)
- Expired or revoked tokens
- URL format issues (ensure https:// prefix)

---

### 🔍 Search and Data Issues

#### Problem: "No Search Results Found"
**Symptoms:**
- Search returns empty results
- Expected documents not appearing
- Tool reports "0 results" for known content

**Solutions:**
1. **Check Space Access**
   ```python
   # Test space access
   from agents._tools.confluence_tool import ConfluenceTool
   tool = ConfluenceTool()
   spaces = tool._confluence.get_all_spaces()
   for space in spaces['results']:
       print(f"{space['key']}: {space['name']}")
   ```

2. **Verify CQL Query Syntax**
   ```python
   # Debug search query
   query = "text ~ 'security' AND space.key = 'SEC'"
   print(f"Query: {query}")
   results = tool._confluence.cql(query)
   ```

3. **Test Without Space Filters**
   ```python
   # Remove space restrictions temporarily
   results = tool.search_documentation("security", spaces=None)
   ```

**Common Mistakes:**
- Insufficient permissions for specified spaces
- Case-sensitive space keys
- Complex queries with syntax errors
- Searching archived or restricted content

---

### 💾 Cache Issues

#### Problem: "Cache Not Working" or "SQLite Errors"
**Symptoms:**
- Repeated API calls for same content
- Cache hit rate shows 0%
- SQLite database errors
- Performance issues

**Solutions:**
1. **Check Cache Directory Permissions**
   ```bash
   # Create cache directory if missing
   mkdir -p backend/cache
   chmod 755 backend/cache

   # Check database file permissions
   ls -la backend/cache/confluence_cache.db
   ```

2. **Rebuild Cache Database**
   ```bash
   # Remove corrupted cache
   rm backend/cache/confluence_cache.db

   # Restart tool to recreate
   python -c "from agents._tools.confluence_tool import ConfluenceTool; ConfluenceTool()"
   ```

3. **Verify Cache Configuration**
   ```python
   # Check cache settings
   import os
   print(f"Cache TTL: {os.getenv('CONFLUENCE_CACHE_TTL', '1440')} minutes")
   print(f"Cache Dir: {os.getenv('CONFLUENCE_CACHE_DIR', 'backend/cache')}")
   ```

**Common Mistakes:**
- Missing cache directory
- Insufficient disk space
- SQLite version incompatibility
- Concurrent access conflicts

---

### ⚡ Performance Issues

#### Problem: "Slow Response Times" or "Timeout Errors"
**Symptoms:**
- Requests taking >10 seconds
- Timeout errors from Confluence API
- High memory usage
- Rate limiting errors

**Solutions:**
1. **Enable Debug Logging**
   ```bash
   # Add to .env file
   LOG_LEVEL=DEBUG
   CONFLUENCE_DEBUG=true
   ```

2. **Check Rate Limiting**
   ```python
   # Monitor rate limit status
   from agents._tools.confluence_tool import ConfluenceTool
   tool = ConfluenceTool()
   stats = tool.get_performance_stats()
   print(f"Requests this minute: {stats['requests_per_minute']}")
   print(f"Rate limit triggered: {stats['rate_limited']}")
   ```

3. **Optimize Query Patterns**
   ```python
   # Use cache-friendly queries
   results = tool.search_documentation(
       query="security",
       spaces=["SEC"],  # Limit to specific spaces
       limit=10,        # Reduce result set
       use_cache=True   # Enable caching
   )
   ```

**Common Mistakes:**
- Too many parallel requests
- Large result sets without pagination
- Disabled caching
- Complex CQL queries

---

### 🔗 Integration Issues

#### Problem: "ADK Agent Integration Failed"
**Symptoms:**
- Agent doesn't recognize Confluence tool
- Import errors in agent code
- Tool not available in agent interface
- Missing tool responses

**Solutions:**
1. **Verify Tool Registration**
   ```python
   # Check tool is properly registered
   from agents.security_agent import SecurityAgent
   agent = SecurityAgent()
   tool_names = [tool.name for tool in agent.tools]
   print("Available tools:", tool_names)
   ```

2. **Check Import Paths**
   ```python
   # Test direct import
   try:
       from agents._tools.confluence_tool import ConfluenceTool
       print("✅ Tool import successful")
   except ImportError as e:
       print(f"❌ Import failed: {e}")
   ```

3. **Validate Tool Dependencies**
   ```bash
   # Check required packages
   pip list | grep -E "(atlassian|confluence)"
   ```

**Common Mistakes:**
- Missing tool registration in agent
- Incorrect import paths
- Missing dependencies
- Tool initialization errors

---

### 🛠️ Development and Testing Issues

#### Problem: "Unit Tests Failing"
**Symptoms:**
- Test failures in CI/CD
- Mock objects not working
- Database conflicts in tests
- Authentication issues in test environment

**Solutions:**
1. **Use Test Configuration**
   ```python
   # Create .env.test file
   CONFLUENCE_URL=https://test.atlassian.net
   CONFLUENCE_USERNAME=test@example.com
   CONFLUENCE_API_TOKEN=test-token
   CONFLUENCE_CACHE_DIR=tests/cache
   ```

2. **Mock External Dependencies**
   ```python
   # Mock Confluence API in tests
   import unittest.mock as mock

   with mock.patch('agents._tools.confluence_tool.Confluence') as mock_confluence:
       mock_confluence.return_value.get_all_spaces.return_value = {
           'results': [{'key': 'TEST', 'name': 'Test Space'}]
       }
       # Run tests
   ```

3. **Isolate Database for Tests**
   ```python
   # Use separate test database
   import tempfile
   test_db = tempfile.mktemp(suffix='.db')
   tool = ConfluenceTool(cache_db_path=test_db)
   ```

---

## Diagnostic Commands

### Quick Health Check
```bash
# Complete system check
python -c "
from agents._tools.confluence_tool import ConfluenceTool
import os

print('=== Confluence Tool Health Check ===')

# 1. Environment check
required_vars = ['CONFLUENCE_URL', 'CONFLUENCE_USERNAME', 'CONFLUENCE_API_TOKEN']
for var in required_vars:
    value = os.getenv(var)
    status = '✅' if value else '❌'
    print(f'{status} {var}: {\"Set\" if value else \"Missing\"}')

# 2. Tool initialization
try:
    tool = ConfluenceTool()
    print('✅ Tool initialization: Success')
except Exception as e:
    print(f'❌ Tool initialization: {e}')

# 3. Basic connectivity
try:
    spaces = tool._confluence.get_all_spaces(limit=1)
    print(f'✅ API connectivity: Success ({len(spaces[\"results\"])} spaces)')
except Exception as e:
    print(f'❌ API connectivity: {e}')

# 4. Cache check
try:
    stats = tool.get_cache_stats()
    print(f'✅ Cache system: {stats[\"total_entries\"]} entries')
except Exception as e:
    print(f'❌ Cache system: {e}')

print('=== End Health Check ===')
"
```

### Performance Monitoring
```bash
# Monitor tool performance
python -c "
from agents._tools.confluence_tool import ConfluenceTool
import time

tool = ConfluenceTool()

# Test search performance
start_time = time.time()
results = tool.search_documentation('test', limit=1)
search_time = time.time() - start_time

# Check cache stats
stats = tool.get_cache_stats()

print(f'Search time: {search_time:.2f}s')
print(f'Cache hit rate: {stats[\"hit_rate\"]:.1f}%')
print(f'Total cache entries: {stats[\"total_entries\"]}')
print(f'Cache size: {stats[\"total_size\"] / 1024 / 1024:.1f} MB')

# Recent audit logs
logs = tool.get_audit_logs(limit=5)
print(f'Recent operations: {len(logs)}')
for log in logs[-3:]:
    print(f'  {log[\"timestamp\"]}: {log[\"action\"]} ({log[\"response_time_ms\"]}ms)')
"
```

### Cache Management
```bash
# Clear cache
python -c "from agents._tools.confluence_tool import ConfluenceTool; ConfluenceTool().clear_cache(); print('Cache cleared')"

# View cache contents
sqlite3 backend/cache/confluence_cache.db "SELECT key, expires_at, LENGTH(content) as size FROM confluence_cache ORDER BY created_at DESC LIMIT 10;"

# Cache statistics
sqlite3 backend/cache/confluence_cache.db "SELECT COUNT(*) as total_entries, SUM(LENGTH(content)) as total_size FROM confluence_cache;"
```

## Getting Help

### Log Files
- **Tool logs**: `logs/confluence_tool.log`
- **ADK logs**: `logs/adk.log`
- **API debug logs**: Set `CONFLUENCE_DEBUG=true` in `.env`

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
export CONFLUENCE_DEBUG=true

# Run with debug output
python your_script.py 2>&1 | tee debug.log
```

### Support Channels
1. **Documentation**: Check `/docs/quickstart.md` for setup guidance
2. **API Reference**: https://atlassian-python-api.readthedocs.io/
3. **Confluence API Docs**: https://developer.atlassian.com/cloud/confluence/rest/
4. **Issue Tracking**: Create tickets with debug logs and error details

### Reporting Issues
When reporting issues, include:
1. **Environment details**: Python version, OS, package versions
2. **Configuration**: Sanitized `.env` file (remove tokens)
3. **Error logs**: Complete error messages and stack traces
4. **Steps to reproduce**: Minimal example that triggers the issue
5. **Expected vs actual behavior**: Clear description of the problem

---

*Last updated: September 2025*
*For the most current troubleshooting information, check the project documentation.*
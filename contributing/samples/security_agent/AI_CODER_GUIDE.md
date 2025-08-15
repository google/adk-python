# 🤖 AI Coder Guide - GCP Security Agent

## Quick Start for AI Coders

This guide helps AI assistants (Claude, GPT, etc.) quickly understand and work with this codebase.

## 🎯 Project Overview

**Purpose**: GCP Security Agent with real-time asset inventory and AI-powered security recommendations
**Stack**: Python (FastAPI backend) + Streamlit (frontend)
**Key Feature**: Works with partial GCP services - gracefully degrades when services are unavailable

## 📁 Project Structure

```
security_agent/
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── api/                    # API endpoints
│   │   ├── agent_llm.py       # AI agent coordination
│   │   ├── asset_inventory.py # Asset discovery endpoints
│   │   └── *.py              # Other API modules
│   └── services/              # Business logic
│       ├── enhanced_asset_inventory_service.py  # Real GCP integration
│       └── *.py              # Service implementations
├── frontend/
│   ├── main_app.py            # Streamlit entry point
│   ├── components/            # UI components
│   │   ├── chat/             # Chat interface
│   │   └── dashboard/        # Dashboard views
│   └── services/             # Frontend services
└── cache/                    # JSON cache for asset data

```

## 🚀 Quick Commands

```bash
# Start backend
python run_backend.py

# Start frontend  
python run_frontend.py

# Test endpoints
python test_endpoints.py

# Initialize asset data
python initialize_asset_data.py
```

## 🔧 Common Tasks

### 1. Adding a New API Endpoint

```python
# In backend/api/your_module.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/your-endpoint")
async def your_endpoint():
    try:
        # Your logic here
        return {"success": True, "data": result}
    except Exception as e:
        # Always handle errors gracefully
        logger.debug(f"Error: {e}")
        return {"success": False, "error": str(e)}
```

### 2. Working with GCP Services

```python
# Always use safe imports with fallbacks
try:
    from google.cloud import some_service
    SERVICE_AVAILABLE = True
except ImportError:
    SERVICE_AVAILABLE = False
    
# Provide fallback behavior
if SERVICE_AVAILABLE:
    # Real implementation
    result = some_service.do_something()
else:
    # Mock or cached data
    result = get_mock_data()
```

### 3. Adding Frontend Components

```python
# In frontend/components/your_component.py
import streamlit as st

def render_your_component():
    """Always add docstrings for AI understanding"""
    st.header("Your Component")
    
    # Use centralized services
    from services.asset_data_service import asset_data_service
    data = asset_data_service.get_asset_summary(project_id)
```

## 🛡️ Error Handling Philosophy

**NEVER let the app crash** - Always provide fallbacks:

1. **Service Unavailable**: Return mock/cached data
2. **API Error**: Log at debug level, return safe defaults
3. **Import Error**: Provide mock implementation
4. **Permission Denied**: Gracefully degrade functionality

## 📊 Key Data Flows

### Asset Discovery Flow
```
User Request → Frontend → Backend API → GCP API (or Cache) → Response
                                ↓
                          If GCP fails → Mock Data
```

### Chat Flow
```
User Query → Chat UI → Agent Router → Appropriate Agent → Response
                              ↓
                     Context & Memory Management
```

## 🔍 Important Files to Know

### Backend Core Files
- `backend/main.py` - Application setup and routing
- `backend/api/agent_llm.py` - AI agent coordination
- `backend/services/enhanced_asset_inventory_service.py` - Real GCP integration

### Frontend Core Files  
- `frontend/main_app.py` - UI entry point
- `frontend/components/chat/chat_view.py` - Chat interface
- `frontend/components/dashboard/dashboard_view.py` - Main dashboard

### Configuration
- `backend/requirements.txt` - Python dependencies
- `.env` - Environment variables (create from .env.example)

## 🐛 Debugging Tips

### Check Service Status
```python
# Backend health check
curl http://localhost:8000/health

# Test specific endpoint
curl http://localhost:8000/api/v1/assets/snapshot/PROJECT_ID
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Service not available" | Service is using fallback mode - this is OK |
| "403 Permission denied" | GCP service not enabled - fallback active |
| "Import error" | Missing optional dependency - mock active |
| "Connection refused" | Backend not running - start with `python run_backend.py` |

## 🎨 UI Patterns

### Streamlit Best Practices
```python
# Always use session state for persistence
if "key" not in st.session_state:
    st.session_state.key = default_value

# Use columns for layout
col1, col2 = st.columns(2)
with col1:
    st.metric("Label", value)

# Show loading states
with st.spinner("Loading..."):
    data = fetch_data()
```

## 🔐 Security Considerations

1. **Never hardcode credentials** - Use environment variables
2. **Always validate input** - Sanitize user queries
3. **Use debug logging** for sensitive errors (not warning/error)
4. **Implement rate limiting** for API endpoints

## 📝 Code Style

### Python Style
- Type hints when possible
- Comprehensive docstrings
- Error handling on all external calls
- Async/await for I/O operations

### Naming Conventions
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

## 🚦 Testing

```bash
# Run tests
pytest tests/

# Test specific module
pytest tests/test_asset_inventory.py

# Test with coverage
pytest --cov=backend tests/
```

## 💡 AI Coder Tips

1. **Read error messages carefully** - They often indicate missing optional dependencies, not bugs
2. **Check imports first** - Many issues are import-related with easy fallbacks
3. **Use existing patterns** - Copy patterns from similar files
4. **Test incrementally** - Use test_endpoints.py to verify changes
5. **Preserve fallbacks** - Never remove fallback logic, only enhance it

## 🔄 Workflow for Changes

1. **Understand current behavior** - Read relevant files
2. **Check for existing patterns** - Similar functionality may exist
3. **Implement with fallbacks** - Always handle service unavailability
4. **Test locally** - Run backend and frontend to verify
5. **Document changes** - Update this guide if adding new patterns

## 📚 Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Google Cloud Python Client](https://cloud.google.com/python/docs/reference)

## 🆘 Getting Help

When stuck:
1. Check existing similar implementations in the codebase
2. Look for mock/fallback patterns to copy
3. Test with `test_endpoints.py` for quick validation
4. Use debug logging to understand flow

## 🎯 Key Principles

1. **Graceful Degradation** - System works with partial functionality
2. **Mock Everything** - Provide fallbacks for all external dependencies  
3. **Clear Logging** - Use appropriate log levels (debug for expected issues)
4. **User First** - Never show technical errors to users
5. **AI Friendly** - Code should be self-documenting with clear patterns

---

**Remember**: This system is designed to work even when things fail. Always provide fallbacks and never let the app crash!
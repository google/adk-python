# Claude's Architectural Deep Dive: Client-Server Agent Design

This document outlines the sophisticated, decoupled client-server architecture for the ADK Security Agent. This design maximizes frontend responsiveness while centralizing intelligence and security on the backend.

## Core Philosophy

The architecture separates the user interaction layer (frontend) from the data processing and intelligence layer (backend).

- **Frontend (The Cockpit):** A sleek, responsive Streamlit application. Its primary job is to manage the user interface and the direct, streaming interaction with the GenAI agent. It is fast, lightweight, and never handles sensitive data or complex logic.
- **Backend (The Engine Room):** A powerful, secure FastAPI application. It serves as the brain of the operation, providing the agent with intelligent tools, enriched data, and persistent context.

---

## 1. Frontend: The Asynchronous & Streaming Client

The frontend uses the Google GenAI SDK directly, leveraging `async/await` and `run_async` for a non-blocking, streaming-first user experience.

### Key Responsibilities:
- **Direct Agent Interaction:** Manages the `Agent` and `Runner` from the GenAI SDK to handle the core conversational loop.
- **UI Rendering:** Uses Streamlit to build a dynamic, component-based user interface.
- **Real-time Token Streaming:** MUST render agent responses token-by-token as they are generated, providing an immediate, "ChatGPT-like" feel using `st.write_stream`
- **Tool Delegation:** The "tools" available to the frontend agent are merely lightweight wrappers that make secure API calls to the backend. They contain no business logic.

### Token Streaming Implementation (REQUIRED):
The Streamlit thin client MUST implement token streaming like ADK web UI:
- Use `stream=True` parameter in agent calls
- Implement async streaming with `Runner.run_async()`
- Display tokens in real-time using `st.write_stream()`
- Use the vertex_sqlite agent pattern from `/agents/gcp_security/`
- Ensure streaming works with SQLite tool for security queries

### Example Flow (User sends a message):
1.  The user types a query in the Streamlit chat input.
2.  The `chat_view` captures the query and calls the `async def call_agent_async` function.
3.  This function uses the GenAI `Runner` to start an async stream of events from the agent.
4.  As `ToolCode` or `ContentPart` events arrive, `st.write_stream` updates the UI in real-time.
5.  If a `ToolCode` event arrives for a tool like `discover_gcp_resources`, the frontend tool wrapper makes an `httpx` call to the backend's `/api/v1/assets/discover` endpoint and returns the result to the agent.

---

## 2. Backend: Intelligent Tools & Data as a Service

The backend is where the heavy lifting and true "intelligence" of the system reside. It exposes a series of API endpoints that serve as the concrete implementation for the agent's tools.

### Key Responsibilities:
- **Security & Credential Management:** The backend is the only component that holds and uses GCP service account keys or other credentials. The frontend is completely unprivileged.
- **Complex Tool Implementation:** It contains the complex logic for interacting with external services. For example, the `discover_gcp_resources` tool's backend logic might involve:
    - Calling the Google Cloud Asset Inventory API.
    - Calling the Security Command Center API.
    - Calling the Cloud Monitoring API.
    - Aggregating, enriching, and filtering data from all three sources.
    - Parsing the final result into clean Pydantic models for a structured response.
- **Intelligent Data Provisioning:**
    - **Persistent Sessions:** Manages long-term conversation history in a database (e.g., Firestore, Redis), providing the agent with crucial context for follow-up questions.
    - **Data Enrichment:** Combines data from various sources to give the agent a holistic view.
    - **Sophisticated Caching:** Implements a robust caching layer for expensive API calls to optimize performance and reduce costs.
- **Serving Pydantic Models:** All data endpoints return clean, validated Pydantic models, ensuring type safety and a clear data contract between the frontend and backend.

---

## 🔧 Environment Configuration

The security agent is designed to work with any GCP project through environment-driven configuration. **No hardcoded values** - everything is configurable.

### Required Setup for Engineers

1. **Copy the environment template:**
   ```bash
   cp .env.template .env
   ```

2. **Configure your GCP project:**
   ```bash
   # Edit .env with your project details
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
   ```

3. **Run automated setup:**
   ```bash
   python setup.py  # Interactive configuration
   ```

### Key Environment Variables

**Required:**
- `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account JSON key

**Application URLs:**
- `BACKEND_URL` - Backend API URL (default: http://localhost:8000)
- `FRONTEND_URL` - Frontend URL (default: http://localhost:8501)
- `BACKEND_PORT` - Backend port (default: 8000)
- `FRONTEND_PORT` - Frontend port (default: 8501)

**Database & Caching:**
- `DATABASE_PATH` - SQLite database path (default: cache/gcp_data.db)
- `DATA_REFRESH_INTERVAL` - Auto-refresh seconds (default: 1800)

**Security:**
- `RATE_LIMIT_CHAT` - Chat requests/minute (default: 30)
- `ENABLE_RATE_LIMITING` - Enable/disable rate limiting (default: true)

### Service Account Permissions

Your service account needs these IAM roles:
- **Cloud Asset Viewer** - For asset inventory
- **Security Center Admin Viewer** - For security findings  
- **Storage Admin** - For bucket analysis
- **IAM Security Reviewer** - For IAM analysis
- **Recommender Viewer** - For recommendations
- **Secret Manager Viewer** - For secrets analysis
- **Monitoring Viewer** - For performance metrics

### Quick Start for Engineers

```bash
# 1. Clone and setup
git clone <repo>
cd security_agent

# 2. Configure environment  
python setup.py

# 3. Start services (OFFICIAL ENTRYPOINTS)
python run_backend.py   # Terminal 1 - FastAPI backend
python run_frontend.py  # Terminal 2 - Streamlit with token streaming

# 4. Open http://localhost:8501
```

### 🎯 CRITICAL: Application Entrypoints

**These are the ONLY official entrypoints for the application:**

1. **`python run_backend.py`** - FastAPI Backend Server
   - Manages SQLite database and GCP API data refresh
   - Provides security analysis endpoints
   - Runs on port 8000
   - Supports Cloud Run deployment with `--cloud` flag

2. **`python run_frontend.py`** - Streamlit Frontend with Token Streaming
   - Uses vertex_sqlite agent directly
   - Implements real-time token streaming with `st.write_stream()`
   - Displays responses token-by-token like ADK web UI
   - Runs on port 8501
   - Supports Cloud Run deployment with `--cloud` flag

**Frontend Features (run_frontend.py):**
- ✨ Token-by-token streaming (like ADK web UI)
- 🤖 Direct agent integration with vertex_sqlite
- 🗄️ SQLite tool for all security queries
- 📊 Real-time display with st.write_stream
- 🚀 Quick queries sidebar for common security checks

### Data Import Status

The frontend displays:
- ✅ **Last import time** (e.g., "2 hours ago")
- 📊 **Total records cached** (all GCP APIs)
- 🔄 **Manual refresh button** 
- ⏰ **Auto-refresh every 30 minutes**

All GCP data flows: **APIs → SQLite → Vertex AI Agent**

---

## 💡 Security Agent Data Flow Architecture

### Critical Data Flow (MUST FOLLOW FOR ALL CHANGES):

```
Google GCP APIs → SQLite Database → Tool Call → Agent Response
```

**Every feature or improvement MUST follow this exact flow:**

1. **GCP APIs** → Data fetched from Google Cloud APIs (Asset Inventory, Security Command Center, IAM, Storage, etc.)
2. **SQLite Database** → Data cached in `backend/cache/gcp_data.db` with proper schema
3. **Tool Call** → Agent calls `query_security_data()` with appropriate query_type
4. **Agent Response** → Agent uses embedded instructions to provide analysis + remediation

### Testing Checklist for Any Change:

Before moving to the next feature, VERIFY:
- [ ] Data populates correctly in SQLite database (`sqlite3 backend/cache/gcp_data.db`)
- [ ] Tool can query the new data (`query_type` works in sqlite_tool.py)
- [ ] Agent instructions cover remediation for the new data type
- [ ] Multi-turn conversations work without errors
- [ ] **Restart ADK web after ANY code changes**

### Architecture Insights:

**CRITICAL INSIGHT: Agent Instructions = Remediation Engine**

For Vertex AI's single-tool limitation, the **agent instructions** serve as the comprehensive remediation knowledge base. This approach enables:

- **Data Analysis**: SQLite tool queries all GCP security data from cache
- **Intelligent Recommendations**: Instructions contain complete remediation guides for all database tables
- **Cloud Run Ready**: Single agent with embedded knowledge, no external API dependencies
- **Comprehensive Coverage**: Detailed remediation for storage_buckets, iam_policies, security_findings, assets, api_keys, org_policies, services, alert_policies, logs, recommendations

**Key Pattern:** When users ask "what should I do about it?", the agent uses its embedded instruction knowledge rather than external search tools, maintaining Vertex AI compliance while providing expert-level security guidance.

### Implementation Files:

- `/agents/gcp_security/vertex_sqlite_agent.py` - Contains comprehensive remediation instructions
- `/agents/gcp_security/sqlite_tool.py` - Single tool for all data queries  
- `/backend/services/data_fetcher.py` - Fetches from GCP APIs
- `/backend/cache/gcp_data.db` - SQLite cache database
- Must use `AGENT_MODE=sqlite` in `.env`

### Common Issues & Solutions:

- **"Database not found"** → Run `python populate_sqlite.py` to fetch GCP data
- **"Unexpected tool call"** → Remove search references from agent instructions
- **Changes not working** → Restart ADK web (`pkill -f "adk web" && adk web`)
- **Multi-turn failing** → Ensure only ONE tool in agent definition
- **Wrong agent in dropdown** → Start ADK web from agent directory: 
  ```bash
  cd /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/agents/gcp_security
  adk web
  ```

### Critical ADK Web Rules:

1. **ALWAYS restart ADK web after ANY code changes** - Changes won't take effect otherwise
2. **Start ADK web from the specific agent directory** - For dropdown to show correct agent:
   - Navigate to: `/agents/gcp_security/` 
   - Run: `adk web`
   - This ensures "vertex_sqlite" appears in dropdown, not "agents"

### 🧪 Evaluation Framework Troubleshooting:

**Before Debugging ANY Issue - Validate Evaluation Framework:**
```bash
cd evaluation && python simple_test.py
```

**Common Evaluation Issues:**
- **"Agent import failed"** → Check agent.py module exists with root_agent import
- **"SQLite tool error"** → Ensure database exists and has correct permissions  
- **"Evaluation failed"** → Verify .evalset.json files are valid JSON format
- **"Coverage gaps"** → Run `python test_coverage_verification.py` for detailed analysis

**Evaluation Framework Status Checks:**
```bash
# Quick validation (30 seconds)
python simple_test.py

# Full test run (5-10 minutes)  
python comprehensive_test_runner.py --agent vertex_sqlite_agent

# Coverage analysis
python test_coverage_verification.py
```

**Expected Results:**
- ✅ All 4 framework components passing
- ✅ Evaluator running successfully on all datasets
- ✅ SQLite tool returning security data
- ✅ Agent tools functioning correctly

---

## 🧹 Project Cleanup Requirements

**CRITICAL:** This project follows a SINGLE-AGENT, thin client-server architecture. Multi-agent/swarm code must be removed.

### Before Starting Any Work

1. **Review CLEANUP_RULES.md** located in `/docs/` for comprehensive cleanup guidelines.

2. **Remove These On Sight:**
   - Any multi-agent or swarm-related code
   - Files containing patterns: `swarm`, `RADAR`, `multi-agent`, `coordination`, `orchestration`
   - Duplicate files (e.g., `agent_enhanced.py` when `agent.py` exists)
   - Empty placeholder directories with only `__init__.py`
   - `/archive/`, `/old/`, `/deprecated/` directories
   - Files ending in `.bak`, `.backup`, `~`, `_old`, `_copy`

3. **Run These Cleanup Checks at Session Start:**
   ```bash
   # Find duplicate Python files
   find . -name "*.py" -exec basename {} \; | sort | uniq -d
   
   # Check for multi-agent code remnants
   grep -r "swarm\|RADAR\|multi-agent" --include="*.py" --include="*.md"
   
   # Remove Python cache
   find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
   
   # Find and remove empty directories
   find . -type d -empty -delete
   
   # Remove large log files
   find . -name "*.log" -size +10M -delete
   ```

4. **Maintain This Clean Structure:**
   ```
   security_agent/
   ├── agent.py             # Single agent implementation
   ├── backend/
   │   ├── main.py         # FastAPI server
   │   └── api/            # API endpoints (tools)
   ├── frontend/
   │   ├── main_app.py     # Streamlit app
   │   └── thin_client.py  # Minimal client
   ├── deploy/             # Deployment configs
   └── tests/              # Test files
   ```

5. **Alert the User Immediately If You Find:**
   - Hardcoded credentials or API keys in code
   - Service account JSON files tracked in git
   - Files larger than 100MB
   - Database files with real data
   - More than 3 similar documentation files

### Continuous Cleanup Philosophy

- **Don't create new files** if you can extend existing ones
- **Remove old code** when replacing with new implementations
- **Consolidate similar functions** into single modules
- **Never leave commented-out code blocks** in production files
- **Keep documentation minimal and authoritative** - one source of truth

---

## 🔧 Troubleshooting Guide - Known Issues & Fixes

### Critical ADK/Vertex AI Patterns

#### 1. **ModuleNotFoundError: No module named 'vertex_sqlite_agent'**
**Error:** Import fails when running streaming_client.py from frontend directory
**Root Cause:** Relative imports fail because sqlite_tool.py is in the same directory as vertex_sqlite_agent.py
**Fix:** 
```python
# Change to agent directory temporarily during import
original_cwd = os.getcwd()
os.chdir(agent_dir)
from vertex_sqlite_agent import root_agent
os.chdir(original_cwd)
```

#### 2. **TypeError: Runner.__init__() takes 1 positional argument but 2 were given**
**Error:** ADK Runner initialization fails with positional argument error
**Root Cause:** Runner requires keyword arguments, not positional
**Fix:**
```python
# WRONG
runner = Runner(root_agent)

# CORRECT
runner = Runner(
    app_name="gcp_security_agent",
    agent=root_agent,
    session_service=InMemorySessionService()
)
```

#### 3. **ValueError: Session not found**
**Error:** Runner.run() fails with session not found error
**Root Cause:** Using async create_session() instead of sync version
**Fix:**
```python
# WRONG - async method in sync context
session = session_service.create_session(...)

# CORRECT - use sync version
session = session_service.create_session_sync(
    app_name="test",
    user_id=user_id,
    session_id=session_id,
    state={}
)
```

#### 4. **TypeError: Runner.run() takes 1 positional argument but 2 were given**
**Error:** Cannot pass query string directly to runner.run()
**Root Cause:** Runner.run() expects specific parameters, not a query string
**Fix:**
```python
# WRONG
result = runner.run("What tables are available?")

# CORRECT
from google.genai import types

new_message = types.Content(
    role="user",
    parts=[types.Part(text="What tables are available?")]
)

for event in runner.run(
    user_id=user_id,
    session_id=session_id,
    new_message=new_message
):
    # Process events
```

#### 5. **DataFetcher.__init__() missing 1 required positional argument: 'project_id'**
**Error:** Background cache refresh fails in main.py
**Root Cause:** DataFetcher requires project_id but it wasn't being passed
**Fix:**
```python
# WRONG
fetcher = DataFetcher()

# CORRECT
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
if not project_id or project_id == "your-project-id":
    logger.warning("⚠️ GOOGLE_CLOUD_PROJECT not configured")
    continue
fetcher = DataFetcher(project_id=project_id)
```

#### 6. **ADK Web Agent Selection Issues**
**Error:** Wrong agent appears in ADK web dropdown
**Root Cause:** ADK web needs to be started from the agent's directory
**Fix:**
```bash
# WRONG - starts from wrong directory
cd /path/to/project && adk web

# CORRECT - start from agent directory
cd /path/to/project/agents/gcp_security && adk web
```
**Note:** Always restart ADK web after making agent changes!

#### 7. **Database Path Issues**
**Error:** "Database not found" when running queries
**Root Cause:** Relative paths fail when running from different directories
**Fix:**
```python
# WRONG - relative path in .env
DATABASE_PATH=backend/cache/gcp_data.db

# CORRECT - absolute path in .env
DATABASE_PATH=/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/backend/cache/gcp_data.db
```

#### 8. **Streaming=False and Follow-up Query Issues**
**Error:** Model logs show "stream: False" and interface becomes unresponsive after first query
**Root Cause:** Agent doesn't support streaming parameter, and session state corruption
**Fix:**
```python
# WRONG - Agent doesn't accept stream parameter
Agent(stream=True)  # ValidationError: Extra inputs not permitted

# CORRECT - Handle streaming in client with better session management
def stream_agent_response(query: str):
    for event in runner.run(...):
        # Process events properly
        if hasattr(event, 'content') and event.content:
            # Break text into words for streaming effect
            words = text.split(' ')
            for word in words:
                yield word
                
# Also ensure proper error handling in Streamlit
try:
    full_response = st.write_stream(stream_agent_response(query))
    if full_response and full_response.strip():
        st.session_state.messages.append(...)
except Exception as e:
    st.error(f"Error: {e}")
```

#### 9. **DataFetcher Background Refresh Errors**
**Errors:** Multiple data fetching failures in background cache refresh
**Root Causes:** API client library issues and invalid metric types
**Fixes:**

```python
# 1. Fix ZoneList.total_size error
# WRONG
zones_client.list(project=project_id).total_size  # Field doesn't exist

# CORRECT  
zones_list = list(zones_client.list(project=project_id))
return {"zones_checked": len(zones_list)}

# 2. Fix sql_v1 import error
# WRONG
from google.cloud import sql_v1  # Module not available

# CORRECT - with fallback
try:
    from google.cloud.sql_v1 import SqlInstancesServiceClient
except ImportError:
    logger.warning("Cloud SQL library not available")
    return {"count": 0, "skipped": "Library not installed"}

# 3. Fix invalid metric types
# WRONG - these don't exist
metric_types = ["gce_instance", "storage_bucket"]

# CORRECT - use valid metric names
metric_types = [
    "compute.googleapis.com/instance/cpu/utilization",
    "storage.googleapis.com/storage/total_bytes"
]

# 4. Fix missing 'summary' key error
# WRONG
logger.info(f"Complete: {result['summary']}")  # Key doesn't exist

# CORRECT - build summary from stats
total_records = sum(stat.get('count', 0) for stat in result.get('stats', {}).values())
summary = f"{total_records} records, {error_count} errors"
```

### Token Streaming Pattern

The application uses ADK's token streaming for real-time response display:
```python
# In streaming_client.py
for event in runner.run(...):
    if hasattr(event, 'content') and event.content:
        if hasattr(event.content, 'parts'):
            for part in event.content.parts:
                if hasattr(part, 'text'):
                    yield part.text  # Stream tokens
```

### Key Environment Variables

Always ensure these are set:
- `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account key
- `DATABASE_PATH` - Absolute path to SQLite database
- `GOOGLE_GENAI_USE_VERTEXAI=TRUE` - Use Vertex AI instead of GenAI

## 🎯 EVALUATION FRAMEWORK STANDARD (REQUIRED)

### ✅ Comprehensive ADK Evaluation Coverage
**ALL NEW FEATURES MUST INCLUDE CORRESPONDING EVALUATION TESTS**

The security agent uses a comprehensive ADK evaluation framework that ensures quality and reliability across all components. This framework is **MANDATORY** for all development work.

### 📊 Required Evaluation Components

**1. Core Test Suites (REQUIRED)**
- **Custom Roles Analyzer** - 6 test cases covering analysis, statistics, export, comparison
- **Knowledge Base** - 6 test cases covering CRUD operations, search, import/export  
- **API Integration** - 3 test cases covering all REST endpoint workflows
- **Edge Cases & Error Handling** - 3 test cases covering malformed input, API failures, extreme data
- **Performance & Scalability** - 3 test cases covering benchmarks, limits, optimization

**2. Security Analysis Test Suites (REQUIRED)**
- **IAM Security Analysis** - Identity and access management testing
- **Storage Security Analysis** - Cloud Storage security assessment
- **Network Security Analysis** - Firewall and VPC security validation
- **Compliance Assessment** - SOC2, GDPR, HIPAA compliance checking
- **Vulnerability Assessment** - Security finding analysis and remediation

### 🛠️ Evaluation Tools (STANDARD)

**1. `evaluation/comprehensive_test_runner.py`** - Enhanced test runner supporting:
   - Parallel and sequential execution
   - Performance benchmarking  
   - Comprehensive reporting
   - Timeout management
   - Error recovery

**2. `evaluation/test_coverage_verification.py`** - Coverage analysis tool providing:
   - 15 functional area coverage tracking
   - Priority-based gap analysis
   - Detailed recommendations
   - Quality metrics calculation

### 📋 Development Workflow (MANDATORY)

**Before ANY Feature Implementation:**
1. Create corresponding `.evalset.json` file in `/evaluation/datasets/`
2. Define comprehensive test cases covering all functionality
3. Include expected responses and tool calls
4. Test edge cases and error scenarios

**After Feature Development:**
1. Run evaluation tests: `python comprehensive_test_runner.py`
2. Achieve >85% test coverage using `python test_coverage_verification.py`
3. Fix any failing tests before code review
4. Update evaluation datasets for new functionality

**Validation Commands:**
```bash
# Test framework is working
cd evaluation && python simple_test.py

# Run comprehensive evaluation
python comprehensive_test_runner.py --agent vertex_sqlite_agent

# Check coverage
python test_coverage_verification.py
```

### 🚨 Quality Gates (ENFORCED)

**No code merges allowed without:**
- ✅ Corresponding evaluation test cases
- ✅ >85% evaluation coverage  
- ✅ All tests passing in CI/CD
- ✅ Performance benchmarks met
- ✅ Security validation completed

### 📈 Evaluation Standards

**Test Case Requirements:**
- **Functional Coverage**: Test all user-facing features
- **Security Coverage**: Validate all security controls
- **API Coverage**: Test all REST endpoints  
- **Error Coverage**: Handle all failure scenarios
- **Performance Coverage**: Validate scalability limits

**Quality Metrics:**
- **Response Times**: <2 seconds for 95% of operations
- **Accuracy**: >95% correct analysis results
- **Coverage**: >85% functional area coverage
- **Reliability**: <0.1% error rate
- **Security**: Zero vulnerabilities in scans

## 🚨 CRITICAL: Unified Streaming Client Architecture

### ⚠️ IMPORTANT: Avoid Sequential Agent Patterns
The security agent uses a **unified streaming client** (`unified_streaming_client.py`) that combines:
- Executive dashboard on the front page
- Token-by-token streaming chat interface
- Direct vertex_sqlite agent integration

### DO NOT Mix Agent Patterns
**NEVER** introduce sequential agent patterns or config_type fields that cause errors like:
```
UserWarning: Field name "config_type" in "SequentialAgent" shadows an attribute in parent "BaseAgent"
ERROR: can only concatenate str (not "NoneType") to str
ERROR: no such column: severity_score
```

### Correct Agent Initialization Pattern
Always use this exact pattern from the working streaming_client:
```python
# CORRECT - Working pattern
if "session_service" not in st.session_state:
    st.session_state.session_service = InMemorySessionService()
    
if "runner" not in st.session_state:
    st.session_state.runner = Runner(
        app_name="gcp_security_agent",
        agent=root_agent,
        session_service=st.session_state.session_service
    )
    
# Then create session with sync method
st.session_state.session = st.session_state.session_service.create_session_sync(...)
```

### Frontend Architecture Rules
- **Single unified app**: `frontend/unified_streaming_client.py`
- **Entry point**: `python run_frontend.py` (uses unified client)
- **Features**: Dashboard + streaming chat in ONE interface
- **NO duplicate pages**: Everything consolidated in one app
- **NO tabs for dashboard**: Dashboard on front page, chat below

### Key Implementation Rules
1. Use vertex_sqlite agent ONLY (no sequential agents)
2. Check for None/null values before string concatenation
3. Use proper event handling for streaming responses
4. Dashboard metrics on front page, NOT in tabs
5. Consolidate duplicate sections and metrics
6. Never mix main_app.py patterns with streaming_client.py patterns

### Streaming Response Pattern
```python
# CORRECT - Handle all event types and check for None
for event in runner.run(...):
    if hasattr(event, 'content') and event.content:
        if hasattr(event.content, 'parts'):
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:  # Check for None!
                    yield part.text
```

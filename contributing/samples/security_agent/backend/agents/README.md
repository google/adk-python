# RADAR Agent Architecture - SIMPLIFIED

## Super Clean Structure

### The Architecture
- **ONE LLM Agent**: RADAR Coordinator (understands intent, orchestrates flow)
- **FIVE Worker Agents**: Simple Python classes that call APIs directly
- **NO unnecessary inheritance or abstractions**

## Files

### 1. `radar_coordinator.py` 
**The Brain** - LLM agent + FastAPI endpoints
- Uses Gemini to understand user intent
- Decides which RADAR phases to run
- Orchestrates sequential flow
- Contains all API endpoints (/chat, /health, /ws)

### 2. Individual Worker Agents
Simple classes that directly call Google Cloud APIs, all defined within `radar_coordinator.py`:

- **`RecognitionAgent`** - Discovers resources
- **`AssessmentAgent`** - Evaluates security 
- **`DecisionAgent`** - Prioritizes issues
- **`ActionAgent`** - Executes fixes (limited write)
- **`ReviewAgent`** - Verifies and reports


### 4. Supporting Docs
- **`RADAR_ARCHITECTURE.md`** - Detailed explanation of RADAR pattern
- **`README.md`** - This file

## How It Works

```python
# User query comes in
query = "Check our security posture"

# RADAR Coordinator (LLM) decides what to do
coordinator = RADARCoordinator(project_id)
phases = coordinator.determine_phases(query)  # ["recognition", "assessment"]

# Coordinator executes the phases
recognition_agent = RecognitionAgent(project_id)
resources = await recognition_agent.discover_all_resources()

assessment_agent = AssessmentAgent(project_id)  
assessment = await assessment_agent.assess_security_posture(resources)

# Return aggregated results
return coordinator.generate_summary(assessment)
```

## Why This is Better

1. **One LLM** - Only the coordinator uses AI (expensive calls)
2. **Simple Workers** - Agents are just Python classes
3. **Direct API Calls** - No wrapper functions or decorators
4. **Clear Flow** - Recognition → Assessment → Decision → Action → Review
5. **No Magic** - Easy to understand and debug

## API Usage

```python
# In main.py
from agents.radar_coordinator import router as radar_router
app.include_router(radar_router, prefix="/api/v1/agent")
```

That's it! The coordinator handles all endpoints:
- `POST /api/v1/agent/chat` - Main chat endpoint
- `GET /api/v1/agent/health` - Health check
- `GET /api/v1/agent/capabilities` - List capabilities
- `WS /api/v1/agent/ws` - WebSocket for streaming
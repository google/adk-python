# Implementation Plan: Fix SQLite Database Connection in Chat Frontend

**Branch**: `001-review-the-project` | **Date**: 2025-09-17 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-review-the-project/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → Found spec at /specs/001-review-the-project/spec.md
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type: web (frontend + backend architecture)
   → Set Structure Decision: Option 2 - Web application
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → No specific violations (constitution template is generic)
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → Research database path resolution issues
   → Investigate ADK agent integration patterns
   → Document findings in research.md
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
7. Re-evaluate Constitution Check section
   → Design follows simple fix approach
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
**UPDATED PRIMARY REQUIREMENT**: Implement real LLM analysis capability in the ADK security agent to generate intelligent insights instead of returning raw JSON data.

**Original Issue**: Users reported that the chat interface was "not able to really connect to sql lite correctly"
**Root Cause Discovery**: Database connectivity works correctly. The real issue is that the ADK agent returns raw JSON data instead of providing LLM-generated security analysis, recommendations, and insights.

**Technical Approach**: Enhance the ADK agent instruction and response processing to analyze tool data through LLM reasoning and generate:
- Custom security risk assessments
- Prioritized recommendations
- Comparative analysis between resources
- Actionable insights based on actual data patterns

## Technical Context
**Language/Version**: Python 3.11 (Streamlit frontend, FastAPI backend)
**Primary Dependencies**: Streamlit, FastAPI, ADK (Agent Development Kit), SQLite3, google.genai
**Storage**: SQLite database at `backend/cache/gcp_data.db`
**Testing**: pytest (existing test files present)
**Target Platform**: Local development and Cloud Run deployment
**Project Type**: web - Frontend (Streamlit) + Backend (FastAPI) architecture
**Performance Goals**: Response time under 5 seconds for database queries
**Constraints**: Must maintain compatibility with existing ADK agent architecture
**Scale/Scope**: Supporting 30+ query types, maintaining session context

**Critical Discovery**: After testing, the fundamental issue is not database connectivity but **LLM analysis capability**. The system successfully:
- ✅ Calls ADK agent with correct tool parameters
- ✅ Retrieves raw JSON data from SQLite database
- ✅ Returns responses to frontend

However, it **fails to provide LLM-generated analysis**:
- ❌ Agent returns raw JSON instead of intelligent insights
- ❌ No security risk prioritization or recommendations
- ❌ No comparative analysis or custom reasoning

**Root Cause**: The Agent→Analysis→Response pipeline is incomplete. The ADK agent receives raw data from tools but doesn't process it through LLM reasoning to generate insights.

**Real Solution Needed**: Implement true LLM analysis layer where the agent analyzes tool responses and generates custom security insights, prioritizations, and recommendations.

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Since the constitution file is a template with placeholders, applying general best practices:
- [x] **Simplicity**: Fix focuses on connection issue, not architecture redesign
- [x] **Test-First**: Will create tests for database connection before implementation
- [x] **Observability**: Add logging for database path resolution and query execution
- [x] **Compatibility**: Maintain existing API contracts and interfaces

## Project Structure

### Documentation (this feature)
```
specs/001-review-the-project/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 2: Web application (existing structure)
backend/
├── main.py              # FastAPI application with ADK integration
├── adk_wrapper.py       # ADK agent wrapper
├── cache/
│   └── gcp_data.db     # SQLite database location
└── tests/

frontend/
├── app.py              # Streamlit application
├── components/
│   └── chat_widget.py  # Chat interface component
├── services/
│   └── adk_service.py  # Backend API client
└── pages/

agents/
├── adk_agent.py        # ADK agent definition
└── tools/
    └── sqlite_tool.py  # SQLite query tool
```

**Structure Decision**: Option 2 - Web application (matches existing architecture)

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - Database path resolution mechanism across different contexts
   - ADK agent session management and state persistence
   - Error propagation from SQLite tool through agent to frontend
   - Environment variable handling for database path

2. **Generate and dispatch research agents**:
   ```
   Task: "Research SQLite path resolution in Python web applications"
   Task: "Find best practices for ADK agent database tool integration"
   Task: "Research FastAPI-Streamlit communication patterns"
   Task: "Investigate session context preservation in ADK runners"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: Absolute path resolution for database
   - Rationale: Prevents working directory issues
   - Alternatives considered: Relative paths, environment-only configuration

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - ChatSession: session_id, user_id, created_at, messages[]
   - QueryRequest: message, session_id, user_id, timestamp
   - QueryResponse: response, tool_usage, execution_time, success
   - DatabaseConnection: path, status, last_query, error_state

2. **Generate API contracts** from functional requirements:
   - POST /api/v1/chat/message - Send chat message
   - GET /api/v1/chat/stream - Stream responses
   - GET /health/database - Check database connectivity
   - POST /api/v1/database/test - Test database query

3. **Generate contract tests** from contracts:
   - test_chat_message_contract.py
   - test_database_health_contract.py
   - test_query_response_schema.py

4. **Extract test scenarios** from user stories:
   - Test: Query high severity findings returns data
   - Test: Multiple queries maintain session context
   - Test: Database path resolution works across contexts
   - Test: Error messages are informative

5. **Update agent file incrementally** (O(1) operation):
   - Update CLAUDE.md with SQLite fix context
   - Add database path resolution guidance
   - Document ADK tool integration patterns

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Database path resolution tasks [P]
- ADK agent configuration validation tasks [P]
- Contract test implementation tasks
- Frontend-backend integration verification tasks
- Session management improvement tasks
- Error handling enhancement tasks

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: Database fix → Agent fix → API fix → Frontend fix
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 20-25 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*No violations identified - fix follows simplicity principle*

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) ✓
- [x] Phase 1: Design complete (/plan command) ✓
- [x] Phase 2: Task planning approach described (/plan command) ✓
- [ ] Phase 3: Tasks to be generated (/tasks command)
- [ ] Phase 4: Implementation pending
- [ ] Phase 5: Validation pending

**Constitution Checks**:
- [x] Initial Constitution Check: Passed
- [x] Post-Design Constitution Check: Passed

**Generated Artifacts**:
- ✓ research.md - Critical discovery: LLM analysis issue identified and documented
- ✓ data-model.md - Created with analysis pipeline entities
- ⚠️ contracts/ - Deferred (focus shifted to LLM analysis pipeline)
- ⚠️ quickstart.md - Deferred (focus shifted to LLM analysis pipeline)
- ⚠️ CLAUDE.md - Already updated with foundation fixes

---

## Next Steps
Execute `/tasks` command to generate the detailed task list for implementation.
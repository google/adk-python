# Frontend Enhancement Plan

This plan translates the earlier architectural observations into actionable work packages for the Flask front end, the agent, and supporting documentation.

## Objectives
1. Deliver a more dynamic chat experience that reflects the ADK agent's streaming responses.
2. Provide structured data pathways from tools to UI to reduce parsing complexity and enable richer visualizations.
3. Surface agent instructions, tool catalogue, and Cloud Function ingestion status in the frontend so operators understand capabilities at a glance.
4. Mirror critical behavioural documentation from code into human-readable artifacts.

## Phase 1 – Foundation
- [x] **Chat API streaming support**
  - Add a streaming-capable endpoint in the Flask app (Server-Sent Events or WebSocket) that proxies the ADK streaming interface.
  - Update the frontend chat component to append partial responses as they arrive.
  - Provide a graceful fallback path for browsers that cannot maintain the streaming connection.
- [x] **Refactor tool responses**
  - Audit existing tool functions to define JSON schemas for the data they return (counts, severities, timestamps, etc.).
  - Update `_tools` modules and `FunctionTool` wrappers to emit structured payloads alongside human-readable summaries.
  - Adjust Flask routes to consume structured data and remove regex-based parsing.
- [x] **Documentation sync**
  - Extract the agent instruction block from `agent.py` into a Markdown document under `docs/`.
  - Link the new document from the README and ensure the frontend can reference it for UI previews.

## Phase 2 – UX Enhancements
- [x] **Instruction & tool catalogue panel**
  - [x] Extend the `/agent-info` endpoint to return structured sections (instructions, tool list, usage examples).
  - [x] Implement a frontend panel that visualizes the instructions and tool descriptions, accessible before starting a chat.
- [ ] **Cloud Function status dashboard** _(in progress)_
  - [x] Document ingestion metadata requirements (last run timestamp, row counts, error state) for the aggregator job.
  - [ ] Create a backend aggregator that reads ingestion metadata (e.g., last run timestamp, row counts) from BigQuery.
  - [ ] Design a UI widget to display status per function and highlight stale data sources.
- [ ] **Data visualization upgrades**
  - Define reusable chart components (time series, severity breakdowns) that consume the structured tool payloads.
  - Prioritize visuals for security insights, exploration metrics, and release analysis outputs.

## Phase 3 – Observability & Polish
- [ ] **Health and latency metrics**
  - Instrument the streaming chat route and dashboard endpoints with timing and error metrics.
  - Report metrics to the existing logging/monitoring stack.
- [ ] **User feedback loops**
  - Add inline feedback controls (thumbs up/down) on agent responses and dashboards.
  - Persist feedback for review and potential model/tool tuning.
- [ ] **Documentation updates**
  - Refresh Cloud Functions README with direct links to function folders and dependency notes.
  - Update frontend README or user guide to describe the new features.

## Dependencies & Sequencing Notes
- Streaming support and structured tool responses unlock most downstream UX work; prioritize them first.
- Any schema changes to tool outputs should be versioned to keep the agent backwards compatible during rollout.
- BigQuery metadata access for Cloud Function status requires confirming available tables or adding lightweight tracking jobs if missing.

## Risk Mitigations
- Introduce feature flags for streaming chat and new dashboards to allow gradual rollout.
- Provide migration guides for existing consumers of tool responses (if any external clients rely on the current text format).
- Ensure adequate test coverage for the new structured responses and UI components.

## Success Criteria
- Users experience near-real-time chat updates without page refreshes.
- Dashboard views display structured insights without manual parsing.
- Operators can quickly assess available tools and data freshness from the UI.
- Documentation accurately reflects agent behaviour and ingestion architecture.

## Progress Log
- **2025-10-06**
  - Completed Phase 1 deliverables: streaming chat endpoint and frontend updates, structured tool responses, and shared instruction documentation.
  - Backend portion of the instruction/tool catalogue panel is live via the enhanced `/agent-info` endpoint returning structured sections for UI consumption.
- **2025-10-07**
  - Implemented the instruction & tool catalogue panel in the frontend, including search, bulk expand/collapse controls, and keyboard-accessible toggles tied to the structured `/agent-info` payload.
- **2025-10-08**
  - Captured ingestion metadata requirements for the upcoming Cloud Function status dashboard and identified existing BigQuery tables to reuse for the aggregator.

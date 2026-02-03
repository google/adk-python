# Design Review Comments and Responses

## Comment 1: Session Service as Durable Persistence

**From:** ADK Team
**Date:** 2026-02-02

**Comment:**
> "Session service is the durable session persistence. For local, user starts with InMemoryService, but they can opt-in storage-based session service: SQLite, DatabaseSessionService, BigQuerySessionService, etc."

---

### Response

Thank you for the feedback. You're correct that ADK already has a robust session service hierarchy. This comment raises an important architectural question: **Why introduce a separate CheckpointStore when SessionService already provides persistence?**

#### Key Distinction: Session State vs. Checkpoint State

| Aspect | Session Service | Checkpoint Store (Proposed) |
|--------|-----------------|----------------------------|
| **What it stores** | Conversation history (events, messages, tool calls) | Agent execution state (job ledgers, progress cursors, partial results) |
| **Granularity** | Per-message/event append | Per-checkpoint snapshot at logical boundaries |
| **Data model** | Event stream (append-only) | Point-in-time snapshots (two-phase commit) |
| **Primary use case** | Replay conversation context to LLM | Resume long-running task from failure point |
| **Recovery question** | "What did the agent say?" | "Where was the agent in a 6-hour BigQuery scan?" |
| **External job tracking** | Tool call events (but not reconciliation-ready) | Authoritative job ledger with status sync |

#### Why Session Service Alone May Be Insufficient

1. **Job Ledger with Authoritative Reconciliation**
   - Session events record that a tool was called, but don't maintain a ledger that can be reconciled against external job states (DONE/FAILED/RUNNING)
   - On resume, we need to query BigQuery: "Is job X still running?" and update our ledger accordingly
   - This reconciliation pattern doesn't fit the append-only event model

2. **Partial Results Persistence**
   - A 50-table PII scan may complete 30 tables before failure
   - Checkpoint stores: which tables done, their findings, which remain
   - Session stores: the conversation about starting the scan

3. **Two-Phase Commit Semantics**
   - Checkpoints require atomic visibility: GCS blob uploaded AND metadata pointer updated
   - Session services typically use simpler append semantics
   - Partial checkpoint writes must not be visible

4. **Workspace Snapshots**
   - Long-running coding agents may need `/workspace` file persistence
   - This is binary blob data, not conversation events
   - Doesn't fit session event model

5. **Different Query Patterns**
   - Session: "Give me all events for session X in order"
   - Checkpoint: "Give me the latest checkpoint for session X" (single row)
   - Fleet ops: "Show me all paused sessions with checkpoints > 1 hour old"

---

### Potential Approaches

#### Option A: Separate CheckpointStore (Current Design)

```
┌─────────────────────────────────────────────────────────────┐
│                    ADK Application                           │
├─────────────────────────────────────────────────────────────┤
│  SessionService (existing)     │  CheckpointStore (new)     │
│  - Conversation history        │  - Execution state         │
│  - Event replay for LLM        │  - Job ledgers             │
│  - Append-only events          │  - Two-phase commit        │
│  - SQLite/DB/BigQuery          │  - BigQuery + GCS          │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Clear separation of concerns
- Different consistency models for different needs
- No changes to existing SessionService implementations
- Checkpoint-specific optimizations (compression, GCS blob storage)

**Cons:**
- Two services to configure for durable agents
- Potential confusion about which stores what
- Additional infrastructure (though can share BigQuery dataset)

#### Option B: Extend SessionService with Checkpoint Capability

```python
class SessionService(ABC):
    # Existing methods...

    # New checkpoint methods
    async def write_checkpoint(
        self, session_id: str, checkpoint_seq: int, state: bytes, ...
    ) -> None: ...

    async def read_latest_checkpoint(
        self, session_id: str
    ) -> tuple[int, bytes] | None: ...
```

**Pros:**
- Single service to configure
- Unified persistence layer
- Familiar pattern for ADK users

**Cons:**
- Mixes conversation semantics with execution semantics
- May require significant changes to existing implementations
- Two-phase commit harder to add to existing append-only services
- Risk of breaking changes

#### Option C: Checkpoint as Special Event Type

```python
# Store checkpoint as a special event in the session
event = Event(
    author="system",
    type=EventType.CHECKPOINT,
    checkpoint_data=CheckpointData(
        seq=5,
        state_gcs_uri="gs://...",
        job_ledger={...},
    )
)
session_service.append_event(session_id, event)
```

**Pros:**
- Uses existing SessionService infrastructure
- Single storage location
- Events remain the universal abstraction

**Cons:**
- Checkpoint retrieval requires scanning events (inefficient)
- Two-phase commit semantics still needed for GCS blob
- Mixing large blobs with conversation events
- Query patterns still don't match (latest vs. stream)

---

### Recommendation

**Option A (Separate CheckpointStore)** is recommended for v1 because:

1. **Clean separation**: Conversation history and execution state serve different purposes
2. **No breaking changes**: Existing SessionService implementations unchanged
3. **Optimized for use case**: Checkpoint-specific features (GCS blobs, two-phase commit, lease management)
4. **Incremental adoption**: Users can add checkpointing without changing session config

However, we should:
- Document the relationship clearly
- Consider Option B for v2 if the pattern proves successful
- Ensure both can share the same BigQuery dataset for operational simplicity

---

## Suggested Updates to Design Doc

Based on this feedback, the following sections should be added/updated in `long_running_task_design.md`:

### 1. Add New Section: "Relationship to Existing Session Service"

**Location:** After Section 5 (Architecture Overview)

```markdown
## 5.4 Relationship to Existing Session Service

ADK provides a `SessionService` abstraction for conversation persistence:

| Implementation | Storage | Use Case |
|----------------|---------|----------|
| `InMemorySessionService` | RAM | Development/testing |
| `SQLiteSessionService` | Local SQLite | Single-machine persistence |
| `DatabaseSessionService` | PostgreSQL/MySQL | Production multi-instance |
| `BigQuerySessionService` | BigQuery | Enterprise scale |

**Why a separate CheckpointStore?**

The `SessionService` and `CheckpointStore` serve complementary purposes:

| SessionService | CheckpointStore |
|----------------|-----------------|
| Conversation history | Execution state snapshots |
| Append-only events | Point-in-time checkpoints |
| LLM context replay | Task resume from failure |
| Per-event granularity | Per-checkpoint granularity |

A durable long-horizon agent typically uses both:
- `SessionService` for conversation continuity
- `CheckpointStore` for execution state durability

**Shared Infrastructure**

Both services can share the same BigQuery dataset:
- `adk_metadata.sessions` (SessionService)
- `adk_metadata.events` (SessionService)
- `adk_metadata.durable_sessions` (CheckpointStore)
- `adk_metadata.checkpoints` (CheckpointStore)
```

### 2. Update Section 8.2 (Configuration)

Add clarity about the relationship:

```markdown
### 8.2 Configuration

```python
# A durable agent uses BOTH session service and checkpoint store
app = App(
    name="durable_scanner",
    root_agent=agent,

    # Session service for conversation history (existing)
    session_service=BigQuerySessionService(
        project="my-project",
        dataset="adk_metadata",
    ),

    # Checkpoint store for execution state (new)
    durable_session_config=DurableSessionConfig(
        is_durable=True,
        checkpoint_store=BigQueryCheckpointStore(
            project="my-project",
            dataset="adk_metadata",  # Can share dataset
            gcs_bucket="my-checkpoints",
        ),
    ),
)
```

**Note:** Both services can share the same BigQuery dataset. The checkpoint tables use a `durable_` prefix to avoid conflicts.
```

### 3. Add to Section 15 (Alternatives Considered)

```markdown
| Alternative | Why not (v1) |
|-------------|--------------|
| Extend SessionService with checkpoint methods | Different consistency models; risk of breaking changes to existing implementations |
| Checkpoint as special Event type | Inefficient retrieval (scan vs. point lookup); mixes blob storage with events |
```

### 4. Add FAQ Entry

```markdown
## Appendix F: FAQ

### Why not just use SessionService for checkpoints?

SessionService is optimized for conversation history (append-only event streams).
Checkpoints require:
- Point-in-time snapshots (not event streams)
- Two-phase commit (GCS blob + metadata atomicity)
- Different query patterns (latest-per-session, not full history)
- Large blob storage (workspace snapshots)

The separation ensures each service is optimized for its use case.

### Can I use CheckpointStore without SessionService?

Yes, but not recommended. SessionService provides conversation context for
the LLM on resume. Without it, the agent loses conversation history.

### Do they share the same BigQuery dataset?

Yes, recommended. Use the same dataset with different table prefixes:
- SessionService: `sessions`, `events`
- CheckpointStore: `durable_sessions`, `checkpoints`
```

---

## Action Items

- [ ] Add Section 5.4 to design doc
- [ ] Update Section 8.2 with dual-service example
- [ ] Add alternatives to Section 15
- [ ] Add FAQ appendix
- [ ] Consider renaming tables to avoid confusion (`durable_sessions` vs `sessions`)
- [ ] Document shared dataset configuration in README

---

## Open Questions for ADK Team

1. **Table naming**: Should checkpoint tables use a prefix (`durable_sessions`) or separate dataset?
2. **Unified service**: Is there interest in a `DurableSessionService` wrapper that manages both?
3. **Event integration**: Should checkpoint events be mirrored to SessionService for audit trail?
4. **BigQuerySessionService**: Does it already have any checkpoint-like capabilities we should leverage?

---

## Comment 2: GcsArtifactService for Large Blobs

**From:** ADK Team
**Date:** 2026-02-02

**Comment:**
> "In ADK, ArtifactService is designed for large blobs. Have you checked that? We have a GcsArtifactService in the core library."

---

### Response

Thank you for pointing this out. Yes, I've reviewed `GcsArtifactService` (`src/google/adk/artifacts/gcs_artifact_service.py`) and the `BaseArtifactService` interface. This is a valid consideration.

#### Current ArtifactService Capabilities

| Feature | GcsArtifactService |
|---------|-------------------|
| Storage backend | GCS bucket |
| Key structure | `{app_name}/{user_id}/{session_id}/{filename}/{version}` |
| Versioning | Monotonic integer versions (0, 1, 2, ...) |
| Data type | `types.Part` (inline_data, text, file_data) |
| Metadata | Custom metadata dict on blob |
| Operations | save, load, list, delete, list_versions |

#### Checkpoint Blob Requirements

| Requirement | ArtifactService Support | Gap |
|-------------|------------------------|-----|
| Store bytes/JSON blobs | Yes (`types.Part.from_bytes`) | None |
| Session-scoped storage | Yes | None |
| Version tracking | Yes (monotonic) | Checkpoint uses `checkpoint_seq` |
| Custom metadata | Yes | Need SHA-256, trigger, size_bytes |
| Two-phase commit | **No** | Critical gap |
| Atomic visibility with BQ | **No** | Critical gap |
| Workspace tar.gz bundles | Partially (as bytes) | None |
| Integrity verification | **No** | Need SHA-256 on read |

#### Key Gaps

**1. Two-Phase Commit Semantics**

The checkpoint pattern requires:
```
Phase 1: Upload blob to GCS (may fail, invisible)
Phase 2: Insert metadata to BigQuery (makes checkpoint visible)
```

`GcsArtifactService.save_artifact()` uploads and returns immediately. There's no coordination with an external metadata store. A partial upload becomes immediately "visible" via `load_artifact()`.

**2. Atomic Visibility with BigQuery Metadata**

Checkpoints must be invisible until both:
- GCS blob exists AND
- BigQuery metadata row exists

`GcsArtifactService` doesn't have this concept - artifacts are visible as soon as they're uploaded.

**3. SHA-256 Integrity Verification**

Checkpoints require integrity verification on read:
```python
# On read
blob = gcs.download(uri)
if sha256(blob) != metadata.sha256:
    raise CheckpointCorruptionError()
```

`GcsArtifactService` doesn't compute or verify checksums.

**4. Key Structure Mismatch**

| Service | Key Pattern |
|---------|-------------|
| ArtifactService | `{app}/{user}/{session}/{filename}/{version}` |
| CheckpointStore | `{session_id}/{checkpoint_seq}/state.json` |

Checkpoints don't have `app_name`, `user_id`, or `filename` - they're keyed purely by `session_id` + `checkpoint_seq`.

---

### Potential Approaches

#### Option A: Use GcsArtifactService as Underlying Storage (Adapt)

```python
class BigQueryCheckpointStore(DurableSessionStore):
    def __init__(self, artifact_service: GcsArtifactService, ...):
        self._artifact_service = artifact_service

    async def write_checkpoint(self, session_id, seq, state_blob, ...):
        # Phase 1: Use artifact service for GCS upload
        version = await self._artifact_service.save_artifact(
            app_name="checkpoints",
            user_id="system",
            session_id=session_id,
            filename=f"checkpoint_{seq}",
            artifact=types.Part.from_bytes(state_blob, mime_type="application/json"),
            custom_metadata={"sha256": sha256(state_blob)},
        )

        # Phase 2: Insert BQ metadata (makes checkpoint visible)
        await self._insert_bq_metadata(session_id, seq, ...)
```

**Pros:**
- Reuses existing GCS infrastructure
- Consistent with ADK patterns
- Less code duplication

**Cons:**
- Awkward key mapping (`app_name="checkpoints"`, `user_id="system"`)
- Still need custom two-phase commit logic
- Still need SHA-256 verification layer
- Version semantics don't match (artifact version vs checkpoint_seq)

#### Option B: Direct GCS Client (Current Design)

```python
class BigQueryCheckpointStore(DurableSessionStore):
    def __init__(self, gcs_bucket: str, ...):
        self._gcs_client = storage.Client()
        self._bucket = self._gcs_client.bucket(gcs_bucket)

    async def write_checkpoint(self, session_id, seq, state_blob, ...):
        # Phase 1: Direct GCS upload with preconditions
        blob = self._bucket.blob(f"{session_id}/{seq}/state.json")
        blob.upload_from_string(
            state_blob,
            if_generation_match=0,  # Fail if exists (idempotency)
        )

        # Phase 2: Insert BQ metadata
        await self._insert_bq_metadata(session_id, seq, ...)
```

**Pros:**
- Full control over GCS operations
- Clean key structure
- Native support for preconditions (`if_generation_match`)
- Simpler code path

**Cons:**
- Doesn't leverage existing ArtifactService
- Separate GCS client initialization

#### Option C: Extend ArtifactService Interface

Add checkpoint-specific methods to `BaseArtifactService`:

```python
class BaseArtifactService(ABC):
    # Existing methods...

    # New: Checkpoint-specific operations
    async def save_checkpoint_blob(
        self,
        *,
        session_id: str,
        checkpoint_seq: int,
        blob: bytes,
        sha256: str,
    ) -> str:
        """Save a checkpoint blob and return GCS URI."""
        ...

    async def load_checkpoint_blob(
        self,
        *,
        session_id: str,
        checkpoint_seq: int,
        expected_sha256: str,
    ) -> bytes:
        """Load and verify checkpoint blob."""
        ...
```

**Pros:**
- Unified artifact/checkpoint interface
- Extensible for future blob types

**Cons:**
- Modifies core ADK interface
- Checkpoint semantics may not fit all artifact backends
- Two-phase commit still external

---

### Recommendation

**Option B (Direct GCS Client)** is recommended for v1 because:

1. **Simpler implementation**: No adapter layer or key mapping
2. **Full control**: Native GCS preconditions for idempotency
3. **Clean semantics**: Checkpoint keys match checkpoint concepts
4. **No interface changes**: Doesn't require modifying BaseArtifactService

However, we should:
- Document the relationship with ArtifactService
- Consider Option A or C for v2 if there's desire for unification
- Ensure both can share the same GCS bucket if needed

---

### Suggested Design Doc Updates

Add to Section 15 (Alternatives Considered):

```markdown
| Alternative | Why not (v1) |
|-------------|--------------|
| Use GcsArtifactService for checkpoint blobs | Key structure mismatch; no two-phase commit support; no SHA-256 verification; would require adapter layer |
```

Add to Section 5.3 (Integration with Existing ADK Services):

```markdown
### Relationship to ArtifactService

ADK's `ArtifactService` (`GcsArtifactService`, `FileArtifactService`, etc.) is designed for
user/session-scoped file artifacts with versioning.

Checkpoints have different requirements:
- Two-phase commit with BigQuery metadata
- SHA-256 integrity verification
- Different key structure (session_id/checkpoint_seq)

For v1, `CheckpointStore` uses direct GCS client access. Future versions may consider
unifying with `ArtifactService` if the interface can be extended to support checkpoint
semantics.
```

---

---

## Comment 3: Leasing as General Requirement

**From:** ADK Team
**Date:** 2026-02-02

**Reference:** Section 7.3 - "We must ensure only one runner resumes a session at a time"

**Comment:**
> "This is not only applicable to resume. `Runner.run_async` also requires this. Leasing is a general requirement for app developers."

---

### Response

This is an important clarification. You're correct that session-level concurrency control is a **general requirement**, not specific to durable session resume.

#### Expanded Scope of Leasing

| Scenario | Concurrency Risk | Current ADK Handling |
|----------|------------------|---------------------|
| Multiple `run_async()` on same session | Race conditions, duplicate tool calls | App developer responsibility |
| Resume after pause | Duplicate resume attempts | App developer responsibility |
| Pub/Sub event redelivery | Multiple runners wake on same event | App developer responsibility |
| Horizontal scaling | Multiple instances claim same session | App developer responsibility |

The design doc incorrectly scoped leasing as a "durable session" concern. In reality:

```
Leasing requirement = ANY scenario where multiple runners might access the same session
```

#### Current State in ADK

Looking at `Runner.run_async()` in `src/google/adk/runners.py`:

```python
async def run_async(
    self,
    *,
    user_id: str,
    session_id: str,
    new_message: types.Content,
    ...
) -> AsyncGenerator[Event, None]:
    # No built-in lease acquisition
    # App developer must ensure single-runner-per-session
```

There's no built-in lease mechanism. App developers must implement their own concurrency control.

#### Implications for Design

**Option A: Leasing in Durable Layer Only (Current Design)**

```
┌─────────────────────────────────────────────────────────────┐
│                    ADK Application                           │
├─────────────────────────────────────────────────────────────┤
│  Runner.run_async()          │  CheckpointStore             │
│  - No built-in leasing       │  - Has lease management      │
│  - App manages concurrency   │  - Protects resume only      │
└─────────────────────────────────────────────────────────────┘
```

**Pros:** Non-breaking, durable sessions get protection
**Cons:** Inconsistent; regular sessions still unprotected

**Option B: Leasing in Runner (Framework-Level)**

```python
class Runner:
    def __init__(self, ..., lease_manager: Optional[LeaseManager] = None):
        self._lease_manager = lease_manager

    async def run_async(self, ..., session_id: str, ...):
        if self._lease_manager:
            lease = await self._lease_manager.acquire(session_id)
            if not lease:
                raise SessionLeaseDeniedError(session_id)
        try:
            # ... execute agent logic
        finally:
            if self._lease_manager:
                await self._lease_manager.release(session_id)
```

**Pros:** Consistent protection for all sessions
**Cons:** Breaking change; requires lease manager configuration

**Option C: Leasing in SessionService (Storage-Level)**

```python
class BaseSessionService(ABC):
    @abstractmethod
    async def acquire_session_lease(
        self, session_id: str, lease_id: str, ttl_seconds: int
    ) -> bool: ...

    @abstractmethod
    async def release_session_lease(
        self, session_id: str, lease_id: str
    ) -> None: ...
```

**Pros:** Unified with session storage; natural fit
**Cons:** Requires changes to all SessionService implementations

---

### Recommendation

**Short-term (v1):** Keep leasing in `CheckpointStore` for durable sessions, but:
- Update design doc to acknowledge this is a subset of a broader need
- Document that app developers need their own concurrency control for non-durable sessions

**Medium-term (v2):** Consider adding leasing to `SessionService` interface:
- `BigQuerySessionService` already has infrastructure for this
- `DatabaseSessionService` can use row-level locks
- `InMemorySessionService` can use asyncio locks

**Long-term:** Consider Runner-level lease integration as opt-in feature.

---

### Suggested Design Doc Updates

**Update Section 7.3 Title:**

From:
> "7.3 Leasing & optimistic concurrency"

To:
> "7.3 Leasing & optimistic concurrency (session-level)"

**Add Clarification Paragraph:**

```markdown
### 7.3 Leasing & Optimistic Concurrency

**Note:** Session-level concurrency control is a general ADK requirement, not
specific to durable sessions. Any scenario where multiple runners might access
the same session requires leasing:

- Multiple `run_async()` calls on the same session
- Resume after pause (durable or in-process)
- Event-driven wake-up with potential redelivery
- Horizontal scaling with shared session storage

Currently, ADK leaves session leasing to app developers. The durable session
layer provides lease management for checkpoint-protected sessions, but this
does not cover all concurrency scenarios.

**Future consideration:** Add optional `LeaseManager` to `Runner` or lease
methods to `SessionService` interface for framework-level protection.
```

**Add to Section 18 (Open Questions):**

```markdown
| Question | Risk Level | Notes |
|----------|------------|-------|
| Framework-level leasing | Medium | Should Runner have built-in lease support? Would require LeaseManager abstraction |
| SessionService lease methods | Medium | Natural fit but requires interface changes |
```

---

---

## Comment 4: Cross-Process Durability Clarification

**From:** ADK Team
**Date:** 2026-02-02

**Reference:** Section 1.2 - "Cross-process durability: state lost if the process dies"

**Comment:**
> "Could you elaborate on this? I think agent state is persisted in the event and the event will be persisted in the selected session service."

---

### Response

You're correct that session events are persisted in the SessionService. Let me clarify what "state lost" means in the context of long-running tasks.

#### What IS Preserved (SessionService Events)

| Data | Preserved? | Location |
|------|------------|----------|
| User messages | Yes | Session events |
| Agent responses | Yes | Session events |
| Tool call records | Yes | Session events (tool name, args, result) |
| LLM conversation context | Yes | Replayable from events |

#### What May NOT Be Preserved (or Not Usable)

| Data | Preserved? | Issue |
|------|------------|-------|
| In-flight tool execution | **No** | Process dies mid-tool-call |
| External job handles | **Partial** | Job ID in event, but no reconciliation structure |
| Multi-step operation progress | **No** | "I'm on step 3 of 7" not tracked |
| Agent's execution plan | **No** | Task graph, priorities, dependencies |
| Partial aggregated results | **No** | "Scanned 30 of 50 tables, found X so far" |
| Workspace files in progress | **No** | Draft reports, intermediate artifacts |

#### Concrete Example: 50-Table PII Scan

**Scenario:** Agent is scanning 50 BigQuery tables for PII. Process dies after completing 30 tables.

**With SessionService only:**

```
Events stored:
  - User: "Scan all tables for PII"
  - Agent: "I'll scan these 50 tables..."
  - ToolCall: scan_table("table_1") → {findings: [...]}
  - ToolCall: scan_table("table_2") → {findings: [...]}
  ...
  - ToolCall: scan_table("table_30") → {findings: [...]}
  - [PROCESS DIES HERE]
```

On restart:
- Events replay to LLM ✓
- LLM sees 30 tool calls completed ✓
- But: **LLM must re-deduce** which tables remain
- But: **No structured job ledger** for reconciliation
- But: **Aggregated findings** must be re-computed from events
- Risk: **LLM may miscount** or re-scan tables

**With Checkpoint + SessionService:**

```
Checkpoint stored:
  {
    "job_ledger": {
      "table_1": {"status": "complete", "findings": 3},
      "table_2": {"status": "complete", "findings": 0},
      ...
      "table_30": {"status": "complete", "findings": 5},
      "table_31": {"status": "pending"},
      ...
      "table_50": {"status": "pending"}
    },
    "aggregated_findings": {
      "total_tables_scanned": 30,
      "total_findings": 47,
      "findings_by_type": {"email": 20, "ssn": 15, "phone": 12}
    },
    "execution_plan": {
      "current_phase": "scanning",
      "next_table_index": 31
    }
  }
```

On restart:
- Load checkpoint ✓
- Know exactly which tables remain ✓
- Reconcile with BigQuery job states ✓
- Continue with aggregated state intact ✓
- No LLM re-deduction needed ✓

#### The Key Distinction

| Aspect | Session Events | Checkpoint State |
|--------|----------------|------------------|
| Purpose | LLM conversation context | Execution state recovery |
| Structure | Append-only event stream | Point-in-time snapshot |
| Recovery mode | Replay events to LLM | Load structured state |
| External jobs | Tool call records | Reconcilable job ledger |
| Aggregations | Must re-compute from events | Pre-computed, ready to use |
| Reliability | LLM must re-deduce state | Deterministic restoration |

#### When Session Events Are Sufficient

Session events alone work well for:
- Short conversations (< 5 min)
- Simple tool calls (no external async jobs)
- Stateless operations (each tool call independent)
- Human-in-the-loop flows (human provides continuity)

#### When Checkpoints Add Value

Checkpoints are valuable for:
- Long-running operations (hours/days)
- External async jobs (BigQuery, Cloud Build, ML training)
- Multi-step plans with dependencies
- Aggregated/computed state (partial results)
- Deterministic recovery (no LLM re-deduction)

---

### End-to-End Concrete Example: Enterprise PII Compliance Audit

Let me walk through a complete scenario showing what the checkpoint approach enables that event logging alone cannot.

#### Scenario Setup

**Task:** Scan 100 BigQuery tables across 5 datasets for PII (emails, SSNs, phone numbers) to generate a compliance report.

**Environment:**
- Cloud Run with 60-minute timeout
- Each table scan takes 2-10 minutes (BigQuery job)
- Total expected runtime: ~8 hours
- Multiple Cloud Run instances may be involved

**User Request:**
```
"Scan all tables in the customer_data, transactions, analytics,
logs, and marketing datasets for PII. Generate a compliance report
with findings by table and recommendations."
```

---

#### Timeline: What Happens

```
Hour 0:00 - Agent starts
  - Discovers 100 tables across 5 datasets
  - Creates execution plan: scan tables, aggregate findings, generate report
  - Begins scanning tables

Hour 2:30 - Progress checkpoint
  - 35 tables scanned
  - 127 PII findings so far
  - 15 BigQuery jobs completed, 2 running, 83 pending

Hour 3:15 - PROCESS DIES (Cloud Run timeout/crash)
  - 2 BigQuery jobs still running in the cloud
  - Agent process terminated
```

---

#### Path A: Event Logging Only (Current ADK)

**Events stored in SessionService:**
```json
[
  {"type": "user_message", "content": "Scan all tables..."},
  {"type": "agent_message", "content": "I'll scan 100 tables..."},
  {"type": "tool_call", "tool": "submit_bq_scan", "args": {"table": "customer_data.users"}, "result": {"job_id": "job_001", "status": "submitted"}},
  {"type": "tool_call", "tool": "get_job_result", "args": {"job_id": "job_001"}, "result": {"findings": [{"type": "email", "column": "contact_email", "count": 15000}]}},
  {"type": "tool_call", "tool": "submit_bq_scan", "args": {"table": "customer_data.orders"}, "result": {"job_id": "job_002", "status": "submitted"}},
  // ... 70 more tool call events ...
  {"type": "tool_call", "tool": "submit_bq_scan", "args": {"table": "analytics.events"}, "result": {"job_id": "job_037", "status": "submitted"}},
  // PROCESS DIES - no more events
]
```

**On Restart (New Cloud Run Instance):**

1. **Events replay to LLM** - LLM sees conversation history ✓

2. **LLM must re-deduce state:**
   ```
   LLM thinking: "Looking at these events... I see job_001 through job_037
   were submitted. Some have results, some don't. Let me figure out what's done..."
   ```

3. **Problems:**

   | Problem | Impact |
   |---------|--------|
   | **Job status unknown** | job_036, job_037 may have completed while process was dead - LLM doesn't know |
   | **No structured ledger** | LLM must parse 70+ events to determine table status |
   | **Aggregation lost** | "127 findings so far" must be re-counted from events |
   | **May re-submit jobs** | LLM might re-scan tables it already scanned |
   | **May miss completed jobs** | Jobs that finished during downtime have results waiting |
   | **Non-deterministic** | Different LLM calls may reach different conclusions |

4. **Likely LLM Response:**
   ```
   "I see we were scanning tables for PII. Let me check what's been done...
   [Spends tokens re-parsing events]
   I think tables 1-35 are done. Let me continue with table 36...

   Actually, I'm not sure if job_036 completed. Let me re-submit it to be safe."
   ```

5. **Result:**
   - Duplicate BigQuery jobs (wasted cost)
   - Inconsistent findings count
   - Report may have duplicates or gaps
   - ~30 minutes spent "figuring out" state

---

#### Path B: Checkpoint + Event Logging (Proposed)

**Checkpoint stored (in addition to events):**
```json
{
  "checkpoint_seq": 15,
  "created_at": "2026-02-02T05:30:00Z",

  "execution_plan": {
    "phase": "scanning",
    "total_tables": 100,
    "tables_completed": 35,
    "tables_in_progress": 2,
    "tables_pending": 63
  },

  "job_ledger": {
    "job_001": {"table": "customer_data.users", "status": "complete", "findings": 3},
    "job_002": {"table": "customer_data.orders", "status": "complete", "findings": 0},
    // ... jobs 3-35: complete ...
    "job_036": {"table": "analytics.sessions", "status": "running", "submitted_at": "2026-02-02T05:28:00Z"},
    "job_037": {"table": "analytics.events", "status": "running", "submitted_at": "2026-02-02T05:29:00Z"}
  },

  "aggregated_findings": {
    "total_findings": 127,
    "by_type": {"email": 45, "ssn": 32, "phone": 28, "address": 22},
    "by_dataset": {"customer_data": 67, "transactions": 35, "analytics": 25},
    "tables_with_pii": ["customer_data.users", "customer_data.profiles", "..."]
  },

  "pending_tables": [
    "analytics.pageviews",
    "logs.access_logs",
    // ... 63 more tables ...
  ]
}
```

**On Restart (New Cloud Run Instance):**

1. **Load checkpoint** - Deterministic state restoration ✓

2. **Reconcile with BigQuery:**
   ```python
   # Automatic reconciliation
   for job_id, job_meta in checkpoint["job_ledger"].items():
       if job_meta["status"] == "running":
           actual_status = bq_client.get_job(job_id).state
           if actual_status == "DONE":
               # Job completed while we were dead - fetch results
               results = fetch_results(job_id)
               update_findings(results)
               job_meta["status"] = "complete"
   ```

3. **Result of reconciliation:**
   ```
   Checkpoint loaded: 35 tables complete, 2 in-progress
   Reconciliation: job_036 DONE (found 5 PII), job_037 DONE (found 2 PII)
   Updated state: 37 tables complete, 134 total findings
   Remaining: 63 tables

   Resuming scan from table 38...
   ```

4. **Agent continues seamlessly:**
   - No duplicate jobs
   - No re-parsing events
   - Findings aggregation intact
   - Deterministic, reliable
   - Resume took ~5 seconds

---

#### Side-by-Side Comparison

| Aspect | Events Only | Checkpoint + Events |
|--------|-------------|---------------------|
| **Recovery time** | ~30 min (LLM re-parsing) | ~5 sec (load + reconcile) |
| **Duplicate jobs** | Likely (LLM uncertainty) | None (ledger prevents) |
| **Missed job results** | Possible | None (reconciliation catches) |
| **Findings accuracy** | May have errors | Exact (pre-aggregated) |
| **Token cost** | High (re-process events) | Low (structured state) |
| **Determinism** | No (LLM-dependent) | Yes (explicit state) |
| **Total runtime** | ~10 hours (retries, confusion) | ~8 hours (clean resume) |

---

#### What Checkpoint Enables That Events Cannot

1. **Authoritative Job Reconciliation**
   ```
   Events: "job_036 was submitted" (but is it done now?)
   Checkpoint: "job_036 status=running" → reconcile → "actually DONE, here are results"
   ```

2. **Pre-Aggregated State**
   ```
   Events: Count findings from 70 tool_call results
   Checkpoint: {"total_findings": 127, "by_type": {...}}
   ```

3. **Explicit Execution Plan**
   ```
   Events: LLM must re-deduce "what was I doing?"
   Checkpoint: {"phase": "scanning", "tables_completed": 35, "tables_pending": 63}
   ```

4. **Idempotent Resume**
   ```
   Events: May or may not re-submit jobs (LLM decides)
   Checkpoint: Never re-submits (ledger tracks all jobs)
   ```

5. **Multi-Instance Coordination**
   ```
   Events: Two instances might both try to continue
   Checkpoint: Lease ensures only one instance resumes
   ```

---

#### Cost Impact Example

| Metric | Events Only | Checkpoint |
|--------|-------------|------------|
| BigQuery jobs submitted | 115 (15 duplicates) | 100 (exact) |
| BQ job cost @ $5/job | $575 | $500 |
| Cloud Run time | 10 hours | 8 hours |
| Cloud Run cost @ $0.10/hr | $1.00 | $0.80 |
| LLM tokens for recovery | ~50,000 | ~1,000 |
| LLM cost @ $0.01/1K | $0.50 | $0.01 |
| **Total extra cost** | **$75.50** | **$0** |

For enterprise workloads running daily, this adds up significantly.

---

### Suggested Design Doc Update

Revise Section 1.2 limitation description:

**From:**
> "Cross-process durability: state lost if the process dies"

**To:**
> "Cross-process durability: While session events persist conversation history, structured execution state (job ledgers, aggregated results, execution plans) is not captured in a form that enables deterministic recovery. On restart, the LLM must re-deduce state from event history, which may be unreliable for complex multi-step operations."

Add clarification table to Section 1.2:

```markdown
**Clarification: Session Events vs. Checkpoint State**

| Recovery Need | Session Events | Checkpoint |
|---------------|----------------|------------|
| Conversation context | ✓ Sufficient | ✓ |
| External job reconciliation | ✗ Manual | ✓ Structured ledger |
| Multi-step progress tracking | ✗ LLM re-deduces | ✓ Explicit state |
| Aggregated partial results | ✗ Re-compute | ✓ Pre-computed |
| Deterministic recovery | ✗ LLM-dependent | ✓ Guaranteed |
```

---

## Updated Open Questions for ADK Team

1. **Table naming**: Should checkpoint tables use a prefix (`durable_sessions`) or separate dataset?
2. **Unified service**: Is there interest in a `DurableSessionService` wrapper that manages both SessionService and CheckpointStore?
3. **Event integration**: Should checkpoint events be mirrored to SessionService for audit trail?
4. **BigQuerySessionService**: Does it already have any checkpoint-like capabilities we should leverage?
5. **ArtifactService unification**: Should we extend `BaseArtifactService` with checkpoint-specific methods in v2?
6. **Shared bucket**: Can checkpoints share a GCS bucket with artifacts, or should they be separate?
7. **Framework-level leasing**: Should `Runner` have optional built-in lease management? Or should `SessionService` have lease methods?
8. **Lease backend standardization**: If leasing becomes a framework feature, what backends should be supported (BQ, Firestore, Redis, DB row locks)?
9. **Event-based recovery**: Is there interest in adding structured "execution state" events to SessionService as an alternative to separate checkpoints?

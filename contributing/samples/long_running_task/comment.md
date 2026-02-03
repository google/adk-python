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

## Updated Open Questions for ADK Team

1. **Table naming**: Should checkpoint tables use a prefix (`durable_sessions`) or separate dataset?
2. **Unified service**: Is there interest in a `DurableSessionService` wrapper that manages both SessionService and CheckpointStore?
3. **Event integration**: Should checkpoint events be mirrored to SessionService for audit trail?
4. **BigQuerySessionService**: Does it already have any checkpoint-like capabilities we should leverage?
5. **ArtifactService unification**: Should we extend `BaseArtifactService` with checkpoint-specific methods in v2?
6. **Shared bucket**: Can checkpoints share a GCS bucket with artifacts, or should they be separate?

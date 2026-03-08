# RFC: Phase 2 Extraction of BigQuery Analytics Event Writers

## Status

Draft

## Summary

This RFC proposes phase 2 of the internal refactor for
`BigQueryAgentAnalyticsPlugin`: extracting the write path behind a common
writer abstraction.

The immediate goal is to isolate the mechanics of event writing so the plugin
no longer knows whether a backend uses:

- `BatchProcessor` with `BigQueryWriteAsyncClient`
- `LegacyStreamingBatchProcessor`

This phase follows phase 1 backend extraction and keeps behavior unchanged.

## Motivation

After phase 1, table-model-specific behavior is isolated behind:

- `NativeBigQueryBackend`
- `BigLakeIcebergBackend`

However, the plugin still has direct knowledge of write-path internals such as:

- `BatchProcessor`
- `LegacyStreamingBatchProcessor`
- optional `write_client`
- native transport cleanup
- queue draining details

That leaves two remaining sources of coupling:

1. The plugin still knows the runtime shape of each backend's write resources.
2. Shutdown and flush logic still depend on native-vs-BigLake implementation
   details.

This phase removes that coupling by introducing a shared internal writer API.

## Goals

- Isolate event writing behind a common interface.
- Preserve existing native BigQuery write behavior.
- Preserve existing BigLake Iceberg write behavior.
- Move write-resource cleanup behind the writer abstraction.
- Reduce plugin knowledge of backend-specific runtime details.

## Non-Goals

This phase does not include:

- Adding new write backends
- Changing backend selection behavior
- Adding DML or load-job writers
- Changing retry behavior
- Changing schema behavior
- Changing public configuration
- Splitting code into new modules

## Current Behavior to Preserve

### Native BigQuery write path

Current native behavior includes:

- Writer resources created from `BigQueryWriteAsyncClient`
- Row batching through `BatchProcessor`
- Appends delegated to `BatchProcessor.append(...)`
- Flush delegated to `BatchProcessor.flush()`
- Shutdown delegated to `BatchProcessor.shutdown(...)`
- Transport cleanup through the write client transport

### BigLake Iceberg write path

Current BigLake behavior includes:

- Writer resources created from `LegacyStreamingBatchProcessor`
- Appends delegated to `LegacyStreamingBatchProcessor.append(...)`
- Flush delegated to `LegacyStreamingBatchProcessor.flush()`
- Shutdown delegated to `LegacyStreamingBatchProcessor.shutdown(...)`
- No write client transport to close

## Proposed Design

Introduce an internal writer abstraction:

- `EventWriter`
- `StorageWriteApiWriter`
- `LegacyStreamingWriter`

The writer owns:

- appending rows
- flushing queued rows
- shutdown and close behavior
- write-resource cleanup
- exposing an atexit-compatible cleanup target if needed

The plugin continues to own:

- callback lifecycle
- event parsing and formatting
- backend selection
- high-level startup orchestration
- high-level shutdown orchestration

## Plugin-Owned Responsibilities

The following remain in `BigQueryAgentAnalyticsPlugin` in phase 2:

- Callback lifecycle
- Event parsing and row construction
- Backend selection
- Loop-state lookup and caching
- High-level startup and shutdown orchestration

The plugin should stop knowing:

- which concrete processor is used
- whether a write client exists
- how transport cleanup works

## Writer Interface

The writer abstraction should remain internal in phase 2.

Suggested shape:

```python
class EventWriter(abc.ABC):
  """Internal interface for writing analytics events."""

  async def start(self) -> None:
    return None

  @abc.abstractmethod
  async def append(self, row: dict[str, Any]) -> None:
    ...

  @abc.abstractmethod
  async def flush(self) -> None:
    ...

  @abc.abstractmethod
  async def shutdown(self, timeout: float) -> None:
    ...

  @abc.abstractmethod
  async def close(self) -> None:
    ...

  @property
  def write_stream(self) -> Optional[str]:
    return None

  def atexit_processor(self) -> Any | None:
    return None
```

Notes:

- `start()` is a concrete no-op in phase 2. Startup ownership remains with the
  existing backend flow for this phase, and moving startup fully into the
  writer is deferred cleanup rather than a primary design goal.
- `close()` is distinct from `shutdown()` because native and BigLake resource
  cleanup differ.
- `write_stream` is optional and preserves the existing `write_stream`
  compatibility surface during the transition away from
  `_batch_processor_prop`.
- `atexit_processor()` is intentionally transitional in phase 2. It preserves
  the current `_atexit_cleanup(...)` behavior even though it still exposes the
  underlying processor as technical debt to remove later.

## Concrete Writers

### `StorageWriteApiWriter`

Responsibilities:

- Wrap `BigQueryWriteAsyncClient`
- Wrap `BatchProcessor`
- Delegate append/flush/shutdown to `BatchProcessor`
- Own transport cleanup for the native write client

Suggested implementation:

```python
class StorageWriteApiWriter(EventWriter):

  def __init__(
      self,
      write_client: BigQueryWriteAsyncClient,
      batch_processor: BatchProcessor,
  ):
    self._write_client = write_client
    self._batch_processor = batch_processor

  async def append(self, row: dict[str, Any]) -> None:
    await self._batch_processor.append(row)

  async def flush(self) -> None:
    await self._batch_processor.flush()

  async def shutdown(self, timeout: float) -> None:
    await self._batch_processor.shutdown(timeout=timeout)

  async def close(self) -> None:
    await self._batch_processor.close()
    if getattr(self._write_client, "transport", None):
      await self._write_client.transport.close()

  @property
  def write_stream(self) -> Optional[str]:
    return self._batch_processor.write_stream

  def atexit_processor(self) -> Any | None:
    return self._batch_processor
```

### `LegacyStreamingWriter`

Responsibilities:

- Wrap `LegacyStreamingBatchProcessor`
- Delegate append/flush/shutdown to the batch processor
- Own BigLake writer cleanup without any write-client assumptions

Suggested implementation:

```python
class LegacyStreamingWriter(EventWriter):

  def __init__(self, batch_processor: LegacyStreamingBatchProcessor):
    self._batch_processor = batch_processor

  async def append(self, row: dict[str, Any]) -> None:
    await self._batch_processor.append(row)

  async def flush(self) -> None:
    await self._batch_processor.flush()

  async def shutdown(self, timeout: float) -> None:
    await self._batch_processor.shutdown(timeout=timeout)

  async def close(self) -> None:
    await self._batch_processor.close()

  def atexit_processor(self) -> Any | None:
    return self._batch_processor
```

## Transitional `_LoopState`

Phase 2 should migrate incrementally. Do not immediately remove the old fields.

Recommended transitional shape:

```python
@dataclass(kw_only=True)
class _LoopState:
  writer: EventWriter
  write_client: Optional[BigQueryWriteAsyncClient] = None
  batch_processor: Optional[Any] = None
```

Rationale:

- The plugin can migrate to `writer` first.
- Existing tests can continue using `write_client` and `batch_processor`
  temporarily.
- Cleanup of old fields can happen at the end of phase 2 after behavior is
  stable.

## Backend Integration

Phase 2 does not change backend selection.

It changes what each backend returns from `create_loop_state(...)`.

### Native backend

Current native backend behavior should become:

- create `BigQueryWriteAsyncClient`
- create `BatchProcessor`
- create `StorageWriteApiWriter`
- return `_LoopState(writer=..., write_client=..., batch_processor=...)`

### BigLake backend

Current BigLake backend behavior should become:

- create `LegacyStreamingBatchProcessor`
- create `LegacyStreamingWriter`
- return `_LoopState(writer=..., write_client=None, batch_processor=...)`

This preserves current runtime behavior while moving the plugin onto the writer
interface.

## Detailed Implementation Plan

### Step 1: Add writer classes in the existing plugin file

File:

- `src/google/adk/plugins/bigquery_agent_analytics_plugin.py`

Actions:

- Add `EventWriter`
- Add `StorageWriteApiWriter`
- Add `LegacyStreamingWriter`
- Keep them in the existing file for phase 2

Rationale:

- Smaller review surface
- No import churn
- Easier comparison against current runtime logic

Acceptance criteria:

- Writer classes compile
- No plugin call sites use them yet

### Step 2: Extend `_LoopState` with `writer`

Actions:

- Add `writer: EventWriter`
- Keep `write_client` and `batch_processor` temporarily

Rationale:

- Enables incremental migration
- Reduces test churn

Acceptance criteria:

- `_LoopState` can hold a writer without changing behavior

### Step 3: Update backends to create writers

Actions:

- Update `NativeBigQueryBackend.create_loop_state()` to create
  `StorageWriteApiWriter`
- Update `BigLakeIcebergBackend.create_loop_state()` to create
  `LegacyStreamingWriter`
- Return `_LoopState(writer=..., ...)`

Guideline:

Preserve current processor start ordering. Do not move startup ownership unless
there is a specific need to do so.

Acceptance criteria:

- Backends return `_LoopState.writer`
- Existing runtime behavior remains unchanged

### Step 4: Add a writer property on the plugin

Add:

```python
@property
def _writer_prop(self) -> Optional[EventWriter]:
  ...
```

This should mirror the current loop-based lookup used by:

- `_batch_processor_prop`
- `_write_client_prop`

Acceptance criteria:

- The plugin can retrieve the current loop's writer

### Step 5: Migrate append path to use `writer`

Actions:

- Replace direct calls to `state.batch_processor.append(row)` with
  `state.writer.append(row)`

Acceptance criteria:

- Native writes still use `BatchProcessor.append(...)` under the wrapper
- BigLake writes still use `LegacyStreamingBatchProcessor.append(...)` under
  the wrapper

### Step 6: Migrate flush path to use `writer`

Actions:

- Replace direct calls to `state.batch_processor.flush()` with
  `state.writer.flush()`

Acceptance criteria:

- Flush behavior remains unchanged for both backends

### Step 7: Migrate shutdown path to use `writer`

Actions:

- Replace direct batch-processor shutdown with `state.writer.shutdown(timeout=t)`
- Replace direct transport cleanup with `state.writer.close()`

Important:

After this step, the writer becomes the single owner of write-resource cleanup.
The plugin should no longer close transports directly.

#### Multi-Loop Shutdown

The current plugin supports draining processors on foreign event loops via
`asyncio.run_coroutine_threadsafe(...)`. Phase 2 must preserve that behavior.

For the current loop:

```python
await state.writer.shutdown(timeout=t)
```

For a foreign loop:

```python
future = asyncio.run_coroutine_threadsafe(
    state.writer.shutdown(timeout=t),
    other_loop,
)
future.result(timeout=t)
```

The same cross-loop dispatch rule applies to `writer.close()` if close needs to
run on a foreign loop. Phase 2 must make this ownership explicit to avoid
double-close or partial-shutdown bugs.

Acceptance criteria:

- Native shutdown still closes the transport
- BigLake shutdown still drains and closes its processor
- No double-close or double-shutdown behavior is introduced

### Step 8: Move atexit registration behind the writer

Actions:

- Replace direct processor registration with:

```python
processor = state.writer.atexit_processor()
if processor is not None:
  atexit.register(self._atexit_cleanup, weakref.proxy(processor))
```

Rationale:

This preserves the existing `_atexit_cleanup(...)` shape while hiding concrete
processor types behind the writer.

Acceptance criteria:

- Existing atexit behavior remains unchanged

### Step 9: Remove direct plugin dependence on old fields

After all plugin call sites use `writer`:

- stop using `_batch_processor_prop` internally
- stop using `_write_client_prop` internally
- migrate `_write_stream_prop` to use `writer.write_stream`
- remove direct transport-close logic from plugin shutdown

This step is about eliminating internal plugin dependence on those properties,
not removing the compatibility surface yet.

#### Compatibility With `__getattribute__` Surface

The current plugin exposes compatibility properties through `__getattribute__`
for:

- `batch_processor`
- `write_client`
- `write_stream`

Phase 2 should preserve this surface during the transition rather than silently
breaking it.

Recommended approach:

- keep `batch_processor` and `write_client` compatibility properties while
  `_LoopState` still carries those fields
- route `write_stream` through `writer.write_stream`
- only consider removing any of these properties in a later cleanup phase with
  an explicit deprecation plan

Acceptance criteria:

- The plugin no longer depends on processor/client internals

### Step 10: Simplify `_LoopState`

Once tests are updated and no plugin logic depends on old fields, reduce
`_LoopState` to:

```python
@dataclass(kw_only=True)
class _LoopState:
  writer: EventWriter
```

This step should be the last change in phase 2.

## Suggested Code Skeleton

```python
class EventWriter(abc.ABC):
  """Internal interface for writing analytics events."""

  async def start(self) -> None:
    return None

  @abc.abstractmethod
  async def append(self, row: dict[str, Any]) -> None:
    ...

  @abc.abstractmethod
  async def flush(self) -> None:
    ...

  @abc.abstractmethod
  async def shutdown(self, timeout: float) -> None:
    ...

  @abc.abstractmethod
  async def close(self) -> None:
    ...

  @property
  def write_stream(self) -> Optional[str]:
    return None

  def atexit_processor(self) -> Any | None:
    return None


class StorageWriteApiWriter(EventWriter):

  def __init__(
      self,
      write_client: BigQueryWriteAsyncClient,
      batch_processor: BatchProcessor,
  ):
    self._write_client = write_client
    self._batch_processor = batch_processor

  async def append(self, row: dict[str, Any]) -> None:
    await self._batch_processor.append(row)

  async def flush(self) -> None:
    await self._batch_processor.flush()

  async def shutdown(self, timeout: float) -> None:
    await self._batch_processor.shutdown(timeout=timeout)

  async def close(self) -> None:
    await self._batch_processor.close()
    if getattr(self._write_client, "transport", None):
      await self._write_client.transport.close()

  @property
  def write_stream(self) -> Optional[str]:
    return self._batch_processor.write_stream

  def atexit_processor(self) -> Any | None:
    return self._batch_processor


class LegacyStreamingWriter(EventWriter):

  def __init__(self, batch_processor: LegacyStreamingBatchProcessor):
    self._batch_processor = batch_processor

  async def append(self, row: dict[str, Any]) -> None:
    await self._batch_processor.append(row)

  async def flush(self) -> None:
    await self._batch_processor.flush()

  async def shutdown(self, timeout: float) -> None:
    await self._batch_processor.shutdown(timeout=timeout)

  async def close(self) -> None:
    await self._batch_processor.close()

  def atexit_processor(self) -> Any | None:
    return self._batch_processor
```

## Suggested Implementation Order

1. Add writer classes with copied delegation logic
2. Extend `_LoopState` with `writer`
3. Update backends to create writers
4. Add `_writer_prop`
5. Migrate append path
6. Migrate flush path
7. Migrate shutdown path
8. Migrate atexit registration
9. Remove old plugin dependencies
10. Simplify `_LoopState`

This order minimizes risk and makes regressions easier to isolate.

## Pickle / Unpickle Compatibility

Phase 2 changes runtime state again by adding writer-owned objects. The plugin
already has explicit pickle compatibility handling, so this phase must preserve
that contract.

Requirements:

- `__getstate__()` must clear any new writer-related non-picklable runtime
  state
- `__setstate__()` must backfill any new fields with `setdefault(...)`
- add a regression test that simulates unpickling a pre-phase-2 state

Phase 2 should assume that writer instances, processors, and async resources
are all runtime-only and must be recreated lazily after unpickle.

## Risks

### Risk 1: Double-close or double-shutdown

If the plugin and writer both try to clean up the same resource, phase 2 can
introduce flaky shutdown behavior.

Mitigation:

- Make the writer the single owner of write-resource cleanup
- Remove direct transport closing from the plugin once shutdown is migrated

### Risk 2: Starting processors twice

If processor startup remains in the backend and is also added to `writer.start()`,
phase 2 can accidentally double-start processing tasks.

Mitigation:

- Preserve current start ordering
- Make `writer.start()` a no-op unless startup ownership is intentionally moved

This is a cleanup concern rather than a primary design blocker for phase 2.

### Risk 3: Atexit cleanup breakage

If processor access is removed too early, `_atexit_cleanup(...)` can no longer
drain or inspect the underlying queue.

Mitigation:

- Keep `atexit_processor()` as a transitional hook

### Risk 4: Test drift toward implementation details

Tests that assert exact `_LoopState` fields can become brittle during the
migration.

Mitigation:

- Prefer tests that assert writer type and observed behavior
- Keep old fields temporarily during the transition

### Risk 5: Fork safety regression

The plugin already has `_reset_runtime_state()` to handle fork safety by
clearing loop-bound runtime state. If `_LoopState` changes shape in phase 2,
that behavior must remain valid.

Mitigation:

- ensure `_reset_runtime_state()` still clears all loop-bound writer state by
  resetting `_loop_state_by_loop`
- treat writer instances as runtime-only resources that are always recreated
  after fork

## Testing Plan

Run at minimum:

```bash
pytest tests/unittests/plugins/test_bigquery_agent_analytics_plugin.py -q
```

Recommended validation areas:

- Existing plugin tests pass unchanged as regression coverage
- New tests for writer selection through backends
- New tests for append/flush/shutdown through the writer interface
- New tests for multi-loop shutdown using `writer.shutdown(...)`
- New tests for native transport cleanup inside `StorageWriteApiWriter`
- New tests for BigLake cleanup without any write-client assumption
- New tests for atexit registration through `writer.atexit_processor()`
- New tests for `write_stream` compatibility through the writer interface
- New tests for pickle/unpickle compatibility with missing phase-2 fields

Recommended new tests:

1. `test_native_backend_returns_storage_write_api_writer`
2. `test_biglake_backend_returns_legacy_streaming_writer`
3. `test_plugin_flush_uses_writer_interface`
4. `test_plugin_shutdown_uses_writer_interface_for_native`
5. `test_plugin_shutdown_uses_writer_interface_for_biglake`
6. `test_storage_write_api_writer_closes_transport`
7. `test_legacy_streaming_writer_close_does_not_require_write_client`
8. `test_atexit_cleanup_registration_uses_writer_processor`
9. `test_multi_loop_shutdown_uses_writer_shutdown`
10. `test_write_stream_property_preserved_via_writer`
11. `test_unpickle_legacy_state_missing_writer_fields`

## Acceptance Criteria

This phase is complete when all of the following are true:

- No public API changes
- No behavior change for native BigQuery tables
- No behavior change for BigLake Iceberg tables
- Existing plugin tests pass unchanged
- Plugin append path uses `writer.append(...)`
- Plugin flush path uses `writer.flush(...)`
- Plugin shutdown path uses `writer.shutdown(...)` and `writer.close(...)`
- Foreign-loop shutdown still uses `asyncio.run_coroutine_threadsafe(...)`
  with `writer.shutdown(...)`
- The plugin no longer closes native transports directly
- `write_stream` remains available through the compatibility surface
- Atexit registration is done through the writer hook
- Pickle/unpickle compatibility is preserved for pre-phase-2 state
- New writer-focused tests pass

## Follow-up Phases

### Phase 3

After phase 2, the plugin no longer depends on concrete write mechanics. That
creates a safe seam for future optional writers if they are justified, such as:

- DML writer
- load-job writer

This phase intentionally does not introduce them.

## Recommendation

Proceed with phase 2 only after phase 1 backend extraction is complete and
stable.

The end-state of phase 2 should be:

- backends decide table behavior
- writers decide write mechanics
- the plugin orchestrates lifecycle without knowing concrete write internals

That is the right long-term structure for keeping BigLake support isolated from
native BigQuery support as the plugin evolves.

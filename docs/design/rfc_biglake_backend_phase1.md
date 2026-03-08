# RFC: Phase 1 Extraction of BigQuery Analytics Table Backends

## Status

Draft

## Summary

This RFC proposes a no-behavior-change refactor of
`BigQueryAgentAnalyticsPlugin` so that native BigQuery table logic and
BigLake Iceberg table logic are isolated behind backend classes.

The immediate goal is to extract the current implementation into:

- `NativeBigQueryBackend`
- `BigLakeIcebergBackend`

This phase does not introduce any new user-visible feature, API, or behavior.
It only restructures the implementation so future BigLake work remains isolated
from the existing native BigQuery path.

## Motivation

The current implementation supports two materially different storage models:

- Native BigQuery tables
- BigLake Iceberg managed tables

They differ across several axes:

- Schema shape
- Table creation behavior
- Writer implementation
- Arrow schema usage
- View compatibility

In the current implementation, those differences are represented with
conditional branching inside `BigQueryAgentAnalyticsPlugin`, especially through
checks like `if self.is_biglake`.

That works for the current MVP, but it will create increasing coupling as more
BigLake-specific behavior is added. Without an isolation boundary, future work
will continue to spread across the plugin class in startup, schema creation,
table creation, shutdown, docs, and testing.

This refactor introduces that isolation boundary now, while the BigLake support
surface is still small.

## Goals

- Isolate native BigQuery and BigLake Iceberg implementation details.
- Preserve all existing behavior for native tables.
- Preserve all existing behavior for BigLake Iceberg tables.
- Keep the refactor incremental and reviewable.
- Establish a clean seam for future backend-specific changes.

## Non-Goals

This phase does not include:

- Adding new write backends
- Changing backend selection behavior
- Switching BigLake to DML or load jobs
- Changing schema semantics
- Splitting code into new modules
- Introducing public configuration changes
- Introducing a generic writer abstraction

Those are valid follow-up phases, but they are intentionally out of scope for
phase 1.

## Current Behavior to Preserve

### Native BigQuery tables

Current native behavior includes:

- Schema built from `_get_events_schema(biglake=False)`
- Arrow schema built through `to_arrow_schema(...)`
- Event writes through `BatchProcessor`
- Use of `BigQueryWriteAsyncClient`
- Use of the `_default` Storage Write API stream
- JSON and RECORD fields preserved in schema
- Existing native table creation behavior preserved
- Existing view behavior preserved

### BigLake Iceberg tables

Current BigLake behavior includes:

- Schema built from `_get_events_schema(biglake=True)`
- BigLake-compatible schema flattening through `_replace_json_with_string(...)`
- Table creation using `BigLakeConfiguration`
- No Arrow schema generation
- Event writes through `LegacyStreamingBatchProcessor`
- Use of `insert_rows_json(...)`
- Existing BigLake partitioning and connection normalization behavior preserved
- Existing plugin behavior around views preserved exactly as-is in this phase

## Proposed Design

Introduce an internal backend abstraction:

- `AnalyticsTableBackend`
- `NativeBigQueryBackend`
- `BigLakeIcebergBackend`

`BigQueryAgentAnalyticsPlugin` remains the public entry point and continues to
own:

- Callback lifecycle
- Event parsing and formatting
- Trace/span handling
- Session/invocation handling
- Shutdown orchestration
- High-level startup orchestration

The backend owns storage-model-specific behavior:

- Schema construction
- Table creation customization
- Arrow schema creation
- Loop-state creation
- Backend capability flags such as whether views are supported

## Backend Interface

The backend abstraction should remain internal in phase 1.

Suggested shape:

```python
class AnalyticsTableBackend(abc.ABC):
  """Storage-model-specific behavior for BQ analytics tables."""

  def __init__(self, plugin: "BigQueryAgentAnalyticsPlugin"):
    self._plugin = plugin

  @property
  @abc.abstractmethod
  def is_biglake(self) -> bool:
    ...

  @abc.abstractmethod
  def build_schema(self) -> list[bigquery.SchemaField]:
    ...

  @abc.abstractmethod
  def maybe_build_arrow_schema(
      self, schema: list[bigquery.SchemaField]
  ) -> Optional[pa.Schema]:
    ...

  @abc.abstractmethod
  def prepare_table_for_create(
      self, table: bigquery.Table
  ) -> bigquery.Table:
    ...

  @abc.abstractmethod
  async def create_loop_state(
      self, loop: asyncio.AbstractEventLoop
  ) -> _LoopState:
    ...

  @abc.abstractmethod
  def supports_views(self) -> bool:
    ...
```

## Concrete Backends

### `NativeBigQueryBackend`

Responsibilities:

- Build native schema
- Build Arrow schema
- Create native write client and `BatchProcessor`
- Apply native table creation settings
- Preserve current native behavior exactly

Suggested implementation:

```python
class NativeBigQueryBackend(AnalyticsTableBackend):

  @property
  def is_biglake(self) -> bool:
    return False

  def build_schema(self) -> list[bigquery.SchemaField]:
    return _get_events_schema(biglake=False)

  def maybe_build_arrow_schema(
      self, schema: list[bigquery.SchemaField]
  ) -> Optional[pa.Schema]:
    arrow_schema = to_arrow_schema(schema)
    if not arrow_schema:
      raise RuntimeError(
          "Failed to convert BigQuery schema to Arrow schema."
      )
    return arrow_schema

  def prepare_table_for_create(
      self, table: bigquery.Table
  ) -> bigquery.Table:
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp",
    )
    table.clustering_fields = self._plugin.config.clustering_fields
    return table

  async def create_loop_state(
      self, loop: asyncio.AbstractEventLoop
  ) -> _LoopState:
    ...

  def supports_views(self) -> bool:
    return True
```

### `BigLakeIcebergBackend`

Responsibilities:

- Build BigLake-compatible schema
- Skip Arrow schema creation
- Create BigLake legacy streaming processor
- Apply BigLake table creation settings
- Preserve current BigLake behavior exactly

Suggested implementation:

```python
class BigLakeIcebergBackend(AnalyticsTableBackend):

  @property
  def is_biglake(self) -> bool:
    return True

  def build_schema(self) -> list[bigquery.SchemaField]:
    return _get_events_schema(biglake=True)

  def maybe_build_arrow_schema(
      self, schema: list[bigquery.SchemaField]
  ) -> Optional[pa.Schema]:
    return None

  def prepare_table_for_create(
      self, table: bigquery.Table
  ) -> bigquery.Table:
    if self._plugin.config.biglake_time_partitioning:
      table.time_partitioning = bigquery.TimePartitioning(
          type_=bigquery.TimePartitioningType.DAY,
          field="timestamp",
      )
    table.clustering_fields = self._plugin.config.clustering_fields
    conn_id = _normalize_biglake_connection_id(
        self._plugin.config.connection_id,
        self._plugin.project_id,
    )
    table.biglake_configuration = BigLakeConfiguration(
        connection_id=conn_id,
        storage_uri=self._plugin.config.biglake_storage_uri,
        file_format="PARQUET",
        table_format="ICEBERG",
    )
    return table

  async def create_loop_state(
      self, loop: asyncio.AbstractEventLoop
  ) -> _LoopState:
    ...

  def supports_views(self) -> bool:
    return False
```

Note: `supports_views()` is included now because it is part of the natural
backend contract, but this phase does not require any behavior change from the
current implementation.

## Plugin Integration

### New plugin field

Add:

```python
self._backend: Optional[AnalyticsTableBackend] = None
```

### Backend selection

Add:

```python
def _make_backend(self) -> AnalyticsTableBackend:
  if self.config.biglake_storage_uri:
    return BigLakeIcebergBackend(self)
  return NativeBigQueryBackend(self)
```

### Backend property

Add:

```python
@property
def backend(self) -> AnalyticsTableBackend:
  if self._backend is None:
    self._backend = self._make_backend()
  return self._backend
```

### Keep `is_biglake`

Preserve the public/internal property but delegate to the backend:

```python
@property
def is_biglake(self) -> bool:
  return self.backend.is_biglake
```

This minimizes diff size and avoids unnecessary churn in the rest of the class.

## Detailed Implementation Plan

### Step 1: Add backend classes in the existing plugin file

File:

- `src/google/adk/plugins/bigquery_agent_analytics_plugin.py`

Actions:

- Add `AnalyticsTableBackend`
- Add `NativeBigQueryBackend`
- Add `BigLakeIcebergBackend`
- Keep these classes in the existing file for phase 1

Rationale:

- Smaller review surface
- No import churn
- Easier to compare old and new logic side by side

Acceptance criteria:

- Backend classes compile
- No code is switched to use them yet

### Step 2: Move schema construction into backends

Actions:

- Move native schema selection to `NativeBigQueryBackend.build_schema()`
- Move BigLake schema selection to `BigLakeIcebergBackend.build_schema()`

Plugin change:

Replace:

```python
self._schema = _get_events_schema(biglake=self.is_biglake)
```

With:

```python
self._schema = self.backend.build_schema()
```

Acceptance criteria:

- Existing schema tests still pass
- No behavior change in the resulting schema

### Step 3: Move Arrow schema creation into backends

Actions:

- Native backend returns `to_arrow_schema(schema)`
- BigLake backend returns `None`

Plugin change:

Replace current `if not self.is_biglake` Arrow logic in `_lazy_setup()` with:

```python
self.arrow_schema = self.backend.maybe_build_arrow_schema(self._schema)
```

Acceptance criteria:

- Native path still creates Arrow schema
- BigLake path still skips Arrow schema
- Existing Arrow-related tests still pass

### Step 4: Move loop-state creation into backends

Actions:

- Copy native `BigQueryWriteAsyncClient` and `BatchProcessor` creation into
  `NativeBigQueryBackend.create_loop_state()`
- Copy BigLake `LegacyStreamingBatchProcessor` creation into
  `BigLakeIcebergBackend.create_loop_state()`

Plugin change:

Replace the backend-specific logic in `_get_loop_state()` with:

```python
async def _get_loop_state(self) -> _LoopState:
  loop = asyncio.get_running_loop()
  self._cleanup_stale_loop_states()
  if loop in self._loop_state_by_loop:
    return self._loop_state_by_loop[loop]

  state = await self.backend.create_loop_state(loop)
  self._loop_state_by_loop[loop] = state
  atexit.register(self._atexit_cleanup, weakref.proxy(state.batch_processor))
  return state
```

Acceptance criteria:

- `_get_loop_state()` no longer branches on `self.is_biglake`
- Existing loop-state behavior is preserved
- Existing write-path tests still pass

### Step 5: Move table creation customization into backends

Actions:

- Move native partitioning and clustering behavior into
  `NativeBigQueryBackend.prepare_table_for_create()`
- Move BigLake partitioning, connection normalization, and
  `BigLakeConfiguration` setup into
  `BigLakeIcebergBackend.prepare_table_for_create()`

Plugin change in `_ensure_schema_exists()`:

Replace inlined table customization with:

```python
tbl = bigquery.Table(self.full_table_id, schema=self._schema)
tbl = self.backend.prepare_table_for_create(tbl)
tbl.labels = {_SCHEMA_VERSION_LABEL_KEY: _SCHEMA_VERSION}
```

Guideline:

For phase 1, prefer explicit duplication inside each backend over a partially
shared contract. The goal is isolation, not deduplication.

Acceptance criteria:

- Table creation behavior remains identical
- BigLake configuration remains identical
- Existing table-creation tests still pass

### Step 6: Keep shutdown logic unchanged

Actions:

- Keep `_LoopState` shape unchanged
- Keep `write_client` optional
- Keep generic shutdown logic unchanged

Rationale:

Shutdown already supports heterogeneous loop state as long as each backend
returns `_LoopState(write_client, batch_processor)` consistently.

Acceptance criteria:

- No change in shutdown behavior
- Existing shutdown tests still pass

### Step 7: Keep helper functions unchanged

Keep the following helpers as-is in phase 1:

- `_get_events_schema(...)`
- `_replace_json_with_string(...)`
- `_normalize_biglake_connection_id(...)`
- `BatchProcessor`
- `LegacyStreamingBatchProcessor`

Only change who calls them.

Rationale:

- Low risk
- Preserves existing tests
- Keeps phase 1 focused on structure

### Step 8: Add focused backend tests

Add tests for the new internal abstraction.

Recommended tests:

1. `test_make_backend_native`
   - Plugin without `biglake_storage_uri`
   - Assert `NativeBigQueryBackend`

2. `test_make_backend_biglake`
   - Plugin with `biglake_storage_uri`
   - Assert `BigLakeIcebergBackend`

3. `test_native_backend_builds_native_schema`
   - Assert native schema preserves JSON/RECORD behavior

4. `test_biglake_backend_builds_biglake_schema`
   - Assert BigLake schema preserves flattened behavior

5. `test_native_backend_builds_arrow_schema`
   - Assert non-`None`

6. `test_biglake_backend_skips_arrow_schema`
   - Assert `None`

7. `test_native_backend_creates_batch_processor`
   - Assert `BatchProcessor`

8. `test_biglake_backend_creates_legacy_processor`
   - Assert `LegacyStreamingBatchProcessor`

Goal:

- Add minimal, targeted coverage for the new abstraction
- Keep existing tests as the primary regression safety net

## Suggested Code Skeleton

```python
class AnalyticsTableBackend(abc.ABC):
  """Storage-model-specific behavior for BQ analytics tables."""

  def __init__(self, plugin: "BigQueryAgentAnalyticsPlugin"):
    self._plugin = plugin

  @property
  @abc.abstractmethod
  def is_biglake(self) -> bool:
    ...

  @abc.abstractmethod
  def build_schema(self) -> list[bigquery.SchemaField]:
    ...

  @abc.abstractmethod
  def maybe_build_arrow_schema(
      self, schema: list[bigquery.SchemaField]
  ) -> Optional[pa.Schema]:
    ...

  @abc.abstractmethod
  def prepare_table_for_create(
      self, table: bigquery.Table
  ) -> bigquery.Table:
    ...

  @abc.abstractmethod
  async def create_loop_state(
      self, loop: asyncio.AbstractEventLoop
  ) -> _LoopState:
    ...

  @abc.abstractmethod
  def supports_views(self) -> bool:
    ...


class NativeBigQueryBackend(AnalyticsTableBackend):

  @property
  def is_biglake(self) -> bool:
    return False

  def build_schema(self) -> list[bigquery.SchemaField]:
    return _get_events_schema(biglake=False)

  def maybe_build_arrow_schema(
      self, schema: list[bigquery.SchemaField]
  ) -> Optional[pa.Schema]:
    arrow_schema = to_arrow_schema(schema)
    if not arrow_schema:
      raise RuntimeError(
          "Failed to convert BigQuery schema to Arrow schema."
      )
    return arrow_schema

  def prepare_table_for_create(
      self, table: bigquery.Table
  ) -> bigquery.Table:
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp",
    )
    table.clustering_fields = self._plugin.config.clustering_fields
    return table

  async def create_loop_state(
      self, loop: asyncio.AbstractEventLoop
  ) -> _LoopState:
    ...

  def supports_views(self) -> bool:
    return True


class BigLakeIcebergBackend(AnalyticsTableBackend):

  @property
  def is_biglake(self) -> bool:
    return True

  def build_schema(self) -> list[bigquery.SchemaField]:
    return _get_events_schema(biglake=True)

  def maybe_build_arrow_schema(
      self, schema: list[bigquery.SchemaField]
  ) -> Optional[pa.Schema]:
    return None

  def prepare_table_for_create(
      self, table: bigquery.Table
  ) -> bigquery.Table:
    if self._plugin.config.biglake_time_partitioning:
      table.time_partitioning = bigquery.TimePartitioning(
          type_=bigquery.TimePartitioningType.DAY,
          field="timestamp",
      )
    table.clustering_fields = self._plugin.config.clustering_fields
    conn_id = _normalize_biglake_connection_id(
        self._plugin.config.connection_id,
        self._plugin.project_id,
    )
    table.biglake_configuration = BigLakeConfiguration(
        connection_id=conn_id,
        storage_uri=self._plugin.config.biglake_storage_uri,
        file_format="PARQUET",
        table_format="ICEBERG",
    )
    return table

  async def create_loop_state(
      self, loop: asyncio.AbstractEventLoop
  ) -> _LoopState:
    ...

  def supports_views(self) -> bool:
    return False
```

## Suggested Implementation Order

1. Add backend classes with copied logic from the plugin
2. Add plugin backend selection helpers
3. Switch schema creation to `backend.build_schema()`
4. Switch Arrow schema creation to `backend.maybe_build_arrow_schema()`
5. Switch `_get_loop_state()` to `backend.create_loop_state()`
6. Switch table creation customization to
   `backend.prepare_table_for_create()`
7. Add focused backend tests
8. Run plugin unit tests
9. Run formatting

This order minimizes behavior drift and makes regressions easier to isolate.

## Migration Notes

Phase 1 should keep everything in:

- `src/google/adk/plugins/bigquery_agent_analytics_plugin.py`

Do not split code into new files yet.

Rationale:

- Smaller review surface
- Easier diff against the current implementation
- Avoids import-cycle and packaging churn

File splitting can happen later once the abstraction is stable.

## Risks

### Risk 1: Behavior drift during extraction

Mitigation:

- Copy code first
- Refactor call sites second
- Preserve startup order and helper usage

### Risk 2: Startup order changes

Mitigation:

- Preserve `_lazy_setup()` sequence exactly
- Only replace backend-specific branches with backend calls

### Risk 3: Table creation mismatch

Mitigation:

- Add focused table-preparation tests
- Keep backend-specific table customization explicit

### Risk 4: Shutdown assumptions

Mitigation:

- Keep `_LoopState` unchanged in phase 1
- Keep optional `write_client` handling unchanged

## Testing Plan

Run at minimum:

```bash
pytest tests/unittests/plugins/test_bigquery_agent_analytics_plugin.py -q
```

Recommended validation areas:

- Existing BigLake schema tests
- Existing BigLake connection normalization tests
- Existing processor selection tests
- Existing lifecycle/shutdown tests
- New backend selection tests
- New backend behavior tests

## Acceptance Criteria

This phase is complete when all of the following are true:

- No public API changes
- No behavior change for native BigQuery tables
- No behavior change for BigLake Iceberg tables
- `_get_loop_state()` no longer contains native/BigLake branching
- `_lazy_setup()` no longer contains direct schema/Arrow branching by table type
- Table creation customization is delegated to backends
- Existing plugin unit tests pass
- New backend-focused tests pass

## Follow-up Phases

### Phase 2

Introduce a writer abstraction behind the backend, for example:

- `StorageWriteApiWriter`
- `LegacyStreamingWriter`

### Phase 3

Add optional backend override only if there is a real use case, for example:

- `write_backend="auto"`
- `biglake_write_backend="legacy_streaming"`

### Phase 4

Add alternative BigLake write implementations only if justified:

- DML writer
- Load-job writer

## Recommendation

Proceed with phase 1 before adding any additional BigLake-specific behavior.

This creates the cleanest implementation boundary with the lowest coupling to
the existing native BigQuery path, while preserving the current MVP behavior in
PR `#4750`.

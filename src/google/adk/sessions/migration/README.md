# Database Schema Migrations for Contributors

This document describes how to add new database schema versions for
`DatabaseSessionService` using Alembic.

For user-facing migration documentation, see the
[Migration Guide](../../../../docs/migration_guide.md). For design rationale,
see the [Alembic Adoption RFC](../../../../docs/rfcs/alembic-adoption.md).

## Overview

ADK uses [Alembic](https://alembic.sqlalchemy.org/) with a strictly linear
revision chain. The chain starts at `001_baseline_v1` and each new migration
appends to it. Migration scripts live in `sessions/migration/versions/` and
ship with the `google-adk` package.

ADK maintains two version tracking mechanisms that migration scripts must keep
in sync:

- `alembic_version` table — Alembic's revision tracking.
- `adk_internal_metadata.schema_version` — ADK's higher-level compatibility
  layer used by `DatabaseSessionService` for schema detection.

## Adding a New Schema Version

The following steps assume you are adding schema version `2`, migrating from
`1`. (Existing versions use bare integers: `"0"`, `"1"`.)

### 1. Update SQLAlchemy Models

Fork from the latest schema version in `google/adk/sessions/schemas/` and
modify the SQLAlchemy model classes (`StorageSession`, `StorageEvent`,
`StorageAppState`, `StorageUserState`, `StorageMetadata`) to reflect the new
schema. Call the new file `v2.py`.

Changes might include adding new `mapped_column` definitions, changing types,
or adding new classes for new tables.

### 2. Generate the Migration

```bash
adk migrate generate --db_url "sqlite:///dev.db" --message "add_session_tags"
```

This uses Alembic's autogenerate to compare the SQLAlchemy models against the
database and produce `upgrade()` and `downgrade()` functions. The output file
is placed in `sessions/migration/versions/`.

To create an empty template instead of auto-detecting changes, add
`--no_autogenerate`.

**Autogenerate limitations**: Alembic cannot detect column renames (appears as
drop + add), data type changes requiring transformation, or some index/constraint
changes. Always review the generated code.

### 3. Review and Customize

- Verify both `upgrade()` and `downgrade()` functions are correct.
- Add data migration logic if needed (e.g., backfilling new columns).
- Update `adk_internal_metadata.schema_version` in `upgrade()`:

    ```python
    op.execute(
        "UPDATE adk_internal_metadata SET value = '2' "
        "WHERE key = 'schema_version'"
    )
    ```

- Add metadata to the migration file header:
    - Database schema version (e.g., `2`)
    - Compatible ADK versions (e.g., `>=1.26.0`)
    - Rollback notes (any downgrade limitations)

### 4. Update Schema Version Constants

Add the new version and update `LATEST_SCHEMA_VERSION` in
`google/adk/sessions/migration/_schema_check_utils.py`:

```python
SCHEMA_VERSION_2 = "2"
LATEST_SCHEMA_VERSION = SCHEMA_VERSION_2
```

### 5. Update `DatabaseSessionService` Business Logic

If the schema change affects how data is read or written during normal
operation (e.g., a new column that needs to be populated on session creation),
update the methods in `DatabaseSessionService` (`create_session`,
`get_session`, `append_event`, etc.) accordingly.

`DatabaseSessionService` is designed to be backward-compatible with the
previous schema for at least 2 releases. It detects the current database schema
and branches based on `_db_schema_version`. Modify `_prepare_tables` and the
CRUD methods to support both the old and new schema.

### 6. Test

Run unit tests:

```bash
pytest tests/unittests/sessions/migration/
```

Run integration tests against real databases:

```bash
docker compose -f tests/integration/sessions/docker-compose.yml up -d
TEST_POSTGRES_URL="postgresql://testuser:testpass@localhost:5432/test_adk" \
  pytest tests/integration/sessions/
```

Test the full cycle: upgrade, downgrade, upgrade again.

### 7. Commit

Use conventional commit format:

```
feat(migration): add session tags column
```

No CLI changes are needed — the `adk migrate upgrade` command automatically
discovers and applies new migrations.

### 8. Deprecate the Previous Schema

After at least 2 releases, remove backward-compatibility logic for the
previous schema. Only use the latest schema in `DatabaseSessionService` and
raise an exception if detecting legacy schema versions. Keep old schema files
(`schemas/v1.py`) and migration scripts for reference.

## Migration Reversibility Policy

- All migrations must implement `downgrade()`, even if limited.
- Non-reversible migrations (dropping columns, destructive data
  transformations) must be tied to a **MAJOR** release of the ADK SDK.
- Document any downgrade limitations in the migration file header.

---

## Legacy Manual Workflow (Deprecated)

> **Deprecated**: The manual copy-based migration workflow below is deprecated
> and will be removed in a future major release. Use the Alembic workflow above
> for all new schema versions.

The following steps describe the legacy process for adding a new schema version
without Alembic. This workflow uses the `adk migrate session` copy-based
migration command.

### 1. Update SQLAlchemy Models

Fork from the latest schema version in `google/adk/sessions/schemas/` folder and
modify the SQLAlchemy model classes (`StorageSession`, `StorageEvent`,
`StorageAppState`, `StorageUserState`, `StorageMetadata`) to reflect the new
`2.0` schema, call it `v2.py`. Changes might be adding new `mapped_column`
definitions, changing types, or adding new classes for new tables.

### 2. Create a New Migration Script

You need to create a script that migrates data from schema `1.0` to `2.0`.

*   Create a new file, for example:
    `google/adk/sessions/migration/migrate_from_1_0_to_2_0.py`.
*   This script must contain a `migrate(source_db_url: str, dest_db_url: str)`
    function, similar to `migrate_from_sqlalchemy_pickle.py`.
*   Inside this function:
    *   Connect to the `source_db_url` (which has schema 1.0) and `dest_db_url`
        engines using SQLAlchemy.
    *   **Important**: Create the tables in the destination database using the
        new 2.0 schema definition by calling
        `v2.Base.metadata.create_all(dest_engine)`.
    *   Read data from the source tables (schema 1.0). The recommended way to do
        this without relying on outdated models is to use `sqlalchemy.text`,
        like:

        ```python
        from sqlalchemy import text
        ...
        rows = source_session.execute(text("SELECT * FROM sessions")).mappings().all()
        ```

    *   For each row read from the source, transform the data as necessary to
        fit the `2.0` schema, and create an instance of the corresponding new
        SQLAlchemy model (e.g., `v2.StorageSession(...)`).
    *   Add these new `2.0` objects to the destination session, ideally using
        `dest_session.merge()` to upsert.
    *   After migrating data for all tables, ensure the destination database is
        marked with the new schema version using the `adk_internal_metadata`
        table:

        ```python
        from google.adk.sessions.migration import _schema_check_utils
        ...
        dest_session.merge(
            v2.StorageMetadata(
                key=_schema_check_utils.SCHEMA_VERSION_KEY,
                value="2.0",
            )
        )
        dest_session.commit()
        ```

### 3. Update Schema Version Constant

You need to add the new version and update `LATEST_SCHEMA_VERSION` in
`google/adk/sessions/migration/_schema_check_utils.py` to reflect the new version:

```python
SCHEMA_VERSION_2_0 = "2.0"
LATEST_SCHEMA_VERSION = SCHEMA_VERSION_2_0
```

This will also update `LATEST_VERSION` in `migration_runner.py`, as it uses this
constant.

### 4. Register the New Migration Script in Migration Runner

In `google/adk/sessions/migration/migration_runner.py`, import your new
migration script and add it to the `MIGRATIONS` dictionary. This tells the
runner how to get from version `1.0` to `2.0`. For example:

```python
from google.adk.sessions.migration import _schema_check_utils
from google.adk.sessions.migration import migrate_from_sqlalchemy_pickle
from google.adk.sessions.migration import migrate_from_1_0_to_2_0
...
MIGRATIONS = {
    # Previous migrations
    _schema_check_utils.SCHEMA_VERSION_0_PICKLE: (
        _schema_check_utils.SCHEMA_VERSION_1_JSON,
        migrate_from_sqlalchemy_pickle.migrate,
    ),
    # Your new migration
    _schema_check_utils.SCHEMA_VERSION_1_JSON: (
        _schema_check_utils.SCHEMA_VERSION_2_0,
        migrate_from_1_0_to_2_0.migrate,
    ),
}
```

### 5. Update `DatabaseSessionService` Business Logic

If your schema change affects how data should be read or written during normal
operation (e.g., you added a new column that needs to be populated on session
creation), update the methods within `DatabaseSessionService` (`create_session`,
`get_session`, `append_event`, etc.) in `database_session_service.py`
accordingly.

The `DatabaseSessionService` is designed to be backward-compatible with the
previous schema for a few releases (at least 2). It detects the current database
schema, and if it's using the previous version of schema, it will still function
correctly. But for new databases, it will create tables using the latest schema.
Therefore, you should modify `_prepare_tables` method and the
DatabaseSessionService's methods (`create_session`, `get_session`,
`append_event`, etc.) to branch based on the `_db_schema_version` variable
accordingly.

### 6. CLI Command Changes

No changes are needed for the Click command definition in `cli_tools_click.py`.
The `adk migrate session` command calls `migration_runner.upgrade()`, which will
now automatically detect the source database version and apply the necessary
migration steps (e.g., `0.1 -> 1.0 -> 2.0`, or `1.0 -> 2.0`) to reach
`LATEST_VERSION`.

### 7. Deprecate the Previous Schema

After a few releases (at least 2), remove the logic for the previous schema.
Only use the latest schema in the `DatabaseSessionService`, and raise an
Exception if detecting legacy schema versions. Keep the schema files like
`schemas/v1.py` and the migration scripts for documentation and not-yet-migrated
users.
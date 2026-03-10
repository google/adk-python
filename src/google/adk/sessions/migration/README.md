# Process for Adding a New Schema Version

This document outlines the steps required to introduce a new database schema
version for `DatabaseSessionService`. Let's assume you are introducing schema
version `2.0`, migrating from `1.0`.

## 1. Update SQLAlchemy Models

Fork from the latest schema version in `google/adk/sessions/schemas/` folder and
modify the SQLAlchemy model classes (`StorageSession`, `StorageEvent`,
`StorageAppState`, `StorageUserState`, `StorageMetadata`) to reflect the new
`2.0` schema, call it `v2.py`. Changes might be adding new `mapped_column`
definitions, changing types, or adding new classes for new tables.

## 2. Create a New Migration Script

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

## 3. Update Schema Version Constant

You need to add the new version and update `LATEST_SCHEMA_VERSION` in
`google/adk/sessions/migration/_schema_check_utils.py` to reflect the new version:

```python
SCHEMA_VERSION_2_0 = "2.0"
LATEST_SCHEMA_VERSION = SCHEMA_VERSION_2_0
```

This will also update `LATEST_VERSION` in `migration_runner.py`, as it uses this
constant.

## 4. Register the New Migration Script in Migration Runner

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

## 5. Update `DatabaseSessionService` Business Logic

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

## 6. CLI Command Changes

No changes are needed for the Click command definition in `cli_tools_click.py`.
The `adk migrate session` command calls `migration_runner.upgrade()`, which will
now automatically detect the source database version and apply the necessary
migration steps (e.g., `0.1 -> 1.0 -> 2.0`, or `1.0 -> 2.0`) to reach
`LATEST_VERSION`.

## 7. Deprecate the Previous Schema

After a few releases (at least 2), remove the logic for the previous schema.
Only use the latest schema in the `DatabaseSessionService`, and raise an
Exception if detecting legacy schema versions. Keep the schema files like
`schemas/v1.py` and the migration scripts for documentation and not-yet-migrated
users.
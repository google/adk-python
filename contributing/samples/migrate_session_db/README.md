# Loading and Upgrading Old Session Databases

This example demonstrates how to upgrade a session database created with an older version of ADK to be compatible with the current version.

## Sample Database

This sample includes `dnd_sessions.db`, a database created with ADK v1.15.0. The following steps show how to run into a schema error and then resolve it using the migration script.

## 1. Reproduce the Error

First, copy the old database to `sessions.db`, which is the file the sample application expects.

```bash
cp dnd_sessions.db sessions.db
python main.py
```

Running the application against the old database will fail with a schema mismatch error, as the `events` table is missing a column required by newer ADK versions:

```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such column: events.usage_metadata
```

## 2. Upgrade the Database Schema

ADK provides a migration script to update the database schema. Run the following command to download and execute it.

```bash
# Clean up the previous run before executing the migration
cp dnd_sessions.db sessions.db

# Download and run the migration script
curl -fsSL https://raw.githubusercontent.com/google/adk-python/main/scripts/db_migration.sh | sh -s -- "sqlite:///%(here)s/sessions.db" "google.adk.sessions.database_session_service"
```

This script uses `alembic` to compare the existing schema against the current model definition and automatically generates and applies the necessary migrations.

**Note on generated files:**
*   The script will create an `alembic.ini` file and an `alembic/` directory. You must delete these before re-running the script.
*   The `sample-output` directory in this example contains a reference of the generated files for your inspection.
*   The `%(here)s` variable in the database URL is an `alembic` placeholder that refers to the current directory.

## 3. Run the Agent Successfully

With the database schema updated, the application can now load the session correctly.

```bash
python main.py
```

You should see output indicating that the old session was successfully loaded.

## Limitations

The migration script is designed to add new columns that have been introduced in newer ADK versions. It does not handle more complex schema changes, such as modifying a column's data type (e.g., from `int` to `string`) or altering the internal structure of stored data.
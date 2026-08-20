# Loading and Upgrading Old Session Databases

This example demonstrates how to upgrade a session database created with an older version of ADK to be compatible with the current version.

## Sample Database

This sample includes `dnd_sessions.db`, a database created with ADK v1.15.0. The following steps show how to run into a schema error and then resolve it using the migration command.

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

ADK ships an `adk migrate session` command that reads the old database and writes a new one on the current schema.

```bash
# The migration writes a new database, so remove the copy made above
rm sessions.db

adk migrate session \
  --source_db_url "sqlite:///./dnd_sessions.db" \
  --dest_db_url "sqlite:///./sessions.db" \
  --allow-unsafe-unpickling
```

The command copies every app state, user state, session and event into the new database, converting each one to the current schema, and records the schema version it wrote.

**Notes:**

- `--allow-unsafe-unpickling` is required for this database. The old schema stores event actions as a Python pickle, so unpickling them runs code from the file; only pass this flag for a database you trust.
- The destination must be a new file. Delete `sessions.db` before re-running the command, or the old tables left behind will shadow the new schema and no events will be copied.

## 3. Run the Agent Successfully

With the database schema updated, the application can now load the session correctly.

```bash
python main.py
```

You should see output indicating that the old session was successfully loaded.

## Limitations

The command never writes to the source database, so `--source_db_url` and `--dest_db_url` must differ. It upgrades a database by its recorded schema version, so a database written by a newer ADK than the one you are running has no upgrade path: the command reports a failure rather than downgrading it.

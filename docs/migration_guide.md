# Database Migration Guide

ADK uses [Alembic](https://alembic.sqlalchemy.org/) to manage database schema
changes for `DatabaseSessionService`. When you upgrade ADK to a version that
includes schema changes, Alembic applies the necessary migrations to bring your
database up to date.

Migrations can run automatically on application startup or manually via the ADK
CLI.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ADK_AUTO_MIGRATE_DB` | `false` | Enable automatic migration on startup |

### Auto-migration (development and simple deployments)

Set the environment variable before starting your application:

```bash
export ADK_AUTO_MIGRATE_DB=true
```

When enabled, `DatabaseSessionService` automatically detects the database schema
version and applies any pending migrations during initialization. This includes
bootstrapping Alembic tracking for databases that predate Alembic support.

### Manual migration (production)

Run migrations explicitly using the CLI before deploying your application:

```bash
adk migrate upgrade --db_url "postgresql://user:pass@host/db"
```

### Kubernetes deployments

For Kubernetes, use Helm hooks to run migrations before application pods start.
See the [Helm Migration Guide](helm_migration_guide.md).

## Upgrading from Earlier ADK Versions

### Existing databases without Alembic tracking

ADK versions up to and including 1.24.0 do not use Alembic for schema tracking.
If you are upgrading from any of these versions, the migration system
automatically detects your database schema version and bootstraps Alembic
tracking.

**Using the CLI:**

```bash
adk migrate upgrade --db_url "postgresql://user:pass@host/db"
```

This command auto-bootstraps: it detects the current schema version, performs any
necessary data migration (e.g., V0 pickle-to-JSON conversion), stamps the
Alembic baseline revision, and applies any pending migrations.

**Using auto-migration:**

When `ADK_AUTO_MIGRATE_DB=true`, `DatabaseSessionService` handles bootstrapping
transparently on startup, including V0-to-V1 migration.

### New databases

No action needed. `DatabaseSessionService` creates tables using the latest
schema and stamps the Alembic baseline automatically.

### Legacy copy-based migration

The existing copy-based migration command remains available:

```bash
adk migrate session \
  --source_db_url "postgresql://localhost:5432/v0" \
  --dest_db_url "postgresql://localhost:5432/v1"
```

This copies data from a source database to a destination database, converting
the schema in the process. It is an alternative to the in-place migration
performed by `adk migrate upgrade`.

## CLI Reference

| Command | Description |
|---------|-------------|
| `adk migrate upgrade --db_url URL` | Apply pending migrations (auto-bootstraps existing databases) |
| `adk migrate downgrade --db_url URL --revision "-1"` | Rollback one migration step |
| `adk migrate check --db_url URL` | Check if migrations are pending (exit 0 = up-to-date, exit 1 = pending) |
| `adk migrate stamp --db_url URL` | Bootstrap Alembic tracking for an existing database |
| `adk migrate generate --db_url URL --message MSG` | Generate a new migration script (contributors) |
| `adk migrate session --source_db_url URL --dest_db_url URL` | Legacy copy-based migration |

All commands accept an optional `--log_level` flag (`DEBUG`, `INFO`, `WARNING`,
`ERROR`, `CRITICAL`). The default is `INFO`.

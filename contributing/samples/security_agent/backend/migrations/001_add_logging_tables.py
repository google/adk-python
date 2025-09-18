"""
Migration to add query logging and session state tables.
"""

import sqlite3
import sys
from pathlib import Path
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.database import get_database_path, get_db_connection


def migrate():
    """Add logging tables to the database."""
    db_path = get_database_path()

    print(f"Running migration on database: {db_path}")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Create query_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                query_type TEXT,
                execution_time REAL,
                success BOOLEAN,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✓ Created query_logs table")

        # Create index for query_logs
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_logs_session
            ON query_logs(session_id)
        """)
        print("✓ Created index on query_logs")

        # Create session_state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✓ Created session_state table")

        # Create index for session_state
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_state_user
            ON session_state(user_id)
        """)
        print("✓ Created index on session_state")

        # Create cache_status table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_status (
                id INTEGER PRIMARY KEY,
                table_name TEXT UNIQUE,
                last_updated TIMESTAMP,
                record_count INTEGER,
                status TEXT
            )
        """)
        print("✓ Ensured cache_status table exists")

        conn.commit()
        print("\n✅ Migration completed successfully")

        # Verify tables were created
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\nTables in database: {', '.join(tables)}")


def rollback():
    """Rollback the migration (drop the tables)."""
    db_path = get_database_path()

    print(f"Rolling back migration on database: {db_path}")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Drop tables in reverse order
        cursor.execute("DROP TABLE IF EXISTS query_logs")
        cursor.execute("DROP TABLE IF EXISTS session_state")

        conn.commit()
        print("✅ Rollback completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database migration for logging tables")
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback the migration"
    )
    args = parser.parse_args()

    if args.rollback:
        rollback()
    else:
        migrate()
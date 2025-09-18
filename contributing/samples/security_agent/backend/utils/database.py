"""
Centralized database path resolution and connection utilities.
"""

import os
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_database_path() -> Path:
    """
    Get the absolute path to the SQLite database file.

    Returns:
        Path: Absolute path to the database file
    """
    # Priority: ENV var > relative to project root > default
    db_path_str = os.getenv("DATABASE_PATH")

    if not db_path_str:
        # Fallback to default path relative to project root
        project_root = Path(__file__).parent.parent.parent  # backend/utils/../.. = project root
        db_path_str = "backend/cache/gcp_data.db"
        db_path = project_root / db_path_str
    else:
        db_path = Path(db_path_str)

        # If relative path, make it absolute from project root
        if not db_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / db_path_str

    # Resolve to absolute path
    db_path = db_path.resolve()

    logger.info(f"Database path resolved to: {db_path}")
    return db_path


def validate_database() -> tuple[bool, str]:
    """
    Validate that the database file exists and is readable.

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        db_path = get_database_path()

        # Check if file exists
        if not db_path.exists():
            error_msg = f"Database file does not exist: {db_path}"
            logger.warning(error_msg)
            return False, error_msg

        # Check if file is readable
        if not db_path.is_file():
            error_msg = f"Database path is not a file: {db_path}"
            logger.error(error_msg)
            return False, error_msg

        # Try to open the database
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            conn.close()

            logger.info(f"Database validation successful: {db_path}")
            return True, "Database is valid and accessible"

        except sqlite3.Error as e:
            error_msg = f"Cannot connect to database: {e}"
            logger.error(error_msg)
            return False, error_msg

    except Exception as e:
        error_msg = f"Database validation error: {e}"
        logger.error(error_msg)
        return False, error_msg


@contextmanager
def get_db_connection(db_path: Optional[Path] = None):
    """
    Context manager for safe database connections.

    Args:
        db_path: Optional database path. If None, uses get_database_path()

    Yields:
        sqlite3.Connection: Database connection object
    """
    if db_path is None:
        db_path = get_database_path()

    conn = None
    try:
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect with row factory for dict-like access
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        logger.debug(f"Database connection opened: {db_path}")
        yield conn

    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.debug(f"Database connection closed: {db_path}")


def create_database_if_missing() -> bool:
    """
    Create an empty database with basic schema if it doesn't exist.

    Returns:
        bool: True if database was created, False if it already existed
    """
    db_path = get_database_path()

    if db_path.exists():
        logger.info(f"Database already exists: {db_path}")
        return False

    logger.warning(f"Database not found, creating empty database: {db_path}")

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Create basic security_findings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                category TEXT,
                severity TEXT,
                state TEXT,
                resource_name TEXT,
                description TEXT,
                recommendation TEXT,
                event_time TEXT,
                data TEXT
            )
        """)

        # Create assets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                asset_type TEXT,
                display_name TEXT,
                location TEXT,
                state TEXT,
                create_time TEXT,
                update_time TEXT,
                data TEXT
            )
        """)

        # Create query_logs table for tracking
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

        conn.commit()
        logger.info(f"Empty database created with basic schema: {db_path}")

    return True


def get_database_info() -> dict:
    """
    Get information about the database.

    Returns:
        dict: Database information including path, status, table count, etc.
    """
    db_path = get_database_path()
    is_valid, message = validate_database()

    info = {
        "database_path": str(db_path),
        "exists": db_path.exists(),
        "readable": is_valid,
        "status_message": message,
        "table_count": 0,
        "total_records": 0,
        "tables": []
    }

    if is_valid:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Get table count and names
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table'
                    ORDER BY name
                """)
                tables = cursor.fetchall()
                info["tables"] = [row["name"] for row in tables]
                info["table_count"] = len(info["tables"])

                # Get total record count
                total_records = 0
                for table_name in info["tables"]:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                    count = cursor.fetchone()["count"]
                    total_records += count

                info["total_records"] = total_records

        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            info["error"] = str(e)

    return info


if __name__ == "__main__":
    # Test the utilities
    print("Testing database utilities...")

    # Get database path
    db_path = get_database_path()
    print(f"Database path: {db_path}")

    # Validate database
    is_valid, message = validate_database()
    print(f"Database valid: {is_valid}")
    print(f"Message: {message}")

    # Get database info
    info = get_database_info()
    print(f"Database info: {info}")

    # Create if missing
    if not is_valid:
        created = create_database_if_missing()
        print(f"Database created: {created}")
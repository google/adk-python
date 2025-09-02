"""
Database configuration module for consistent path handling.
Provides centralized database path management across all components.
"""
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Centralized database configuration management."""
    
    @staticmethod
    def get_database_path() -> str:
        """
        Get the absolute path to the SQLite database.
        
        Returns:
            str: Absolute path to the database file
            
        Raises:
            ValueError: If database path cannot be determined
        """
        # Try environment variable first (absolute path preferred)
        env_path = os.getenv("DATABASE_PATH")
        if env_path:
            if os.path.isabs(env_path):
                return env_path
            else:
                # Convert relative path to absolute based on project root
                project_root = DatabaseConfig._get_project_root()
                return str(project_root / env_path)
        
        # Default to backend/cache/gcp_data.db relative to project root
        project_root = DatabaseConfig._get_project_root()
        default_path = project_root / "backend" / "cache" / "gcp_data.db"
        
        # Ensure the cache directory exists
        default_path.parent.mkdir(parents=True, exist_ok=True)
        
        return str(default_path)
    
    @staticmethod
    def _get_project_root() -> Path:
        """
        Get the project root directory (security_agent).
        
        Returns:
            Path: Path to the security_agent directory
        """
        # Start from this file's location and work up to find security_agent
        current = Path(__file__).resolve()
        
        # Walk up the directory tree to find security_agent
        for parent in current.parents:
            if parent.name == "security_agent":
                return parent
        
        # Fallback: assume we're in config/ and go up two levels
        return current.parent.parent
    
    @staticmethod
    def ensure_database_exists(database_path: Optional[str] = None) -> bool:
        """
        Ensure the database file and its directory exist.
        
        Args:
            database_path: Optional path to database file
            
        Returns:
            bool: True if database exists or was created successfully
        """
        if not database_path:
            database_path = DatabaseConfig.get_database_path()
        
        try:
            db_path = Path(database_path)
            
            # Create directory if it doesn't exist
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if database file exists
            if db_path.exists():
                logger.info(f"✅ Database found at: {database_path}")
                return True
            else:
                logger.warning(f"⚠️ Database not found at: {database_path}")
                logger.info("Run 'python populate_sqlite.py' to create and populate the database")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking database: {e}")
            return False
    
    @staticmethod
    def get_database_status() -> dict:
        """
        Get comprehensive database status information.
        
        Returns:
            dict: Database status details
        """
        database_path = DatabaseConfig.get_database_path()
        
        status = {
            "path": database_path,
            "exists": False,
            "readable": False,
            "writable": False,
            "size_bytes": 0,
            "table_count": 0,
            "last_modified": None,
            "error": None
        }
        
        try:
            db_path = Path(database_path)
            
            if db_path.exists():
                status["exists"] = True
                status["readable"] = os.access(database_path, os.R_OK)
                status["writable"] = os.access(database_path, os.W_OK)
                status["size_bytes"] = db_path.stat().st_size
                status["last_modified"] = db_path.stat().st_mtime
                
                # Try to count tables
                try:
                    import sqlite3
                    with sqlite3.connect(database_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                        status["table_count"] = cursor.fetchone()[0]
                except Exception as e:
                    status["error"] = f"Database access error: {str(e)}"
            else:
                status["error"] = "Database file does not exist"
                
        except Exception as e:
            status["error"] = f"Status check error: {str(e)}"
        
        return status

# Convenience function for backward compatibility
def get_database_path() -> str:
    """Get the absolute path to the SQLite database."""
    return DatabaseConfig.get_database_path()

def ensure_database_exists() -> bool:
    """Ensure the database file and its directory exist."""
    return DatabaseConfig.ensure_database_exists()

def get_database_status() -> dict:
    """Get comprehensive database status information."""
    return DatabaseConfig.get_database_status()
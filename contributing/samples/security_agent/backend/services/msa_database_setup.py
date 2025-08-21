#!/usr/bin/env python3
"""
MSA Database Setup - Creates tables for storing MSA analysis results
"""

import sqlite3
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_msa_tables(db_path: str):
    """Create tables for storing MSA analysis results."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Table for storing MSA emails
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS msa_emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_content TEXT NOT NULL,
                received_date TIMESTAMP,
                analyzed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                project_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for storing extracted changes with structured fields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS msa_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                msa_email_id INTEGER,
                service TEXT NOT NULL,
                change_type TEXT NOT NULL,
                description TEXT,
                effective_date TEXT,
                required_action TEXT,
                impact_level TEXT,
                affected_resources TEXT,  -- JSON array
                old_permission TEXT,       -- Original permission being changed
                new_permissions TEXT,      -- JSON array of new permissions
                api_parameters TEXT,       -- JSON object of API parameters
                affects_predefined_roles BOOLEAN,
                testing_available BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (msa_email_id) REFERENCES msa_emails(id)
            )
        """)
        
        # Table for storing impact assessments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS msa_impact_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                msa_change_id INTEGER,
                project_id TEXT NOT NULL,
                resource_type TEXT,
                resource_count INTEGER,
                impact_level TEXT,
                recommended_actions TEXT,  -- JSON array
                affected_resources TEXT,    -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (msa_change_id) REFERENCES msa_changes(id)
            )
        """)
        
        # Table for storing overall recommendations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS msa_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                msa_email_id INTEGER,
                recommendation TEXT NOT NULL,
                priority TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (msa_email_id) REFERENCES msa_emails(id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msa_changes_service ON msa_changes(service)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msa_changes_impact ON msa_changes(impact_level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msa_impact_project ON msa_impact_assessments(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msa_emails_project ON msa_emails(project_id)")
        
        conn.commit()
        logger.info("✅ MSA tables created successfully")
        
        # Log table information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'msa_%'")
        tables = cursor.fetchall()
        logger.info(f"📊 MSA tables in database: {[t[0] for t in tables]}")
        
    except Exception as e:
        logger.error(f"Error creating MSA tables: {e}")
        conn.rollback()
    finally:
        conn.close()

def store_msa_analysis(db_path: str, analysis_results: dict):
    """Store MSA analysis results in the database."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Store the email
        cursor.execute("""
            INSERT INTO msa_emails (email_content, project_id, analyzed_date)
            VALUES (?, ?, ?)
        """, (
            analysis_results.get('email_content', ''),
            analysis_results.get('project_id'),
            datetime.now()
        ))
        
        msa_email_id = cursor.lastrowid
        
        # Store extracted changes with structured fields
        for change in analysis_results.get('extracted_changes', []):
            import json
            cursor.execute("""
                INSERT INTO msa_changes (
                    msa_email_id, service, change_type, description,
                    effective_date, required_action, impact_level, affected_resources,
                    old_permission, new_permissions, api_parameters,
                    affects_predefined_roles, testing_available
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                msa_email_id,
                change.get('service'),
                change.get('change_type'),
                change.get('description'),
                change.get('effective_date'),
                change.get('required_action'),
                change.get('impact_level'),
                json.dumps(change.get('affected_resources', [])),
                change.get('old_permission'),
                json.dumps(change.get('new_permissions', [])) if change.get('new_permissions') else None,
                json.dumps(change.get('api_parameters', {})) if change.get('api_parameters') else None,
                change.get('affects_predefined_roles', False),
                change.get('testing_available', False)
            ))
            
            change_id = cursor.lastrowid
            
            # Store impact assessments for this change
            for assessment in analysis_results.get('impact_assessments', []):
                cursor.execute("""
                    INSERT INTO msa_impact_assessments (
                        msa_change_id, project_id, resource_type, resource_count,
                        impact_level, recommended_actions, affected_resources
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    change_id,
                    assessment.get('project_id'),
                    assessment.get('resource_type'),
                    assessment.get('resource_count'),
                    assessment.get('impact_level'),
                    str(assessment.get('recommended_actions', [])),
                    str(assessment.get('affected_resources', []))
                ))
        
        # Store overall recommendations
        for rec in analysis_results.get('recommendations', []):
            cursor.execute("""
                INSERT INTO msa_recommendations (msa_email_id, recommendation)
                VALUES (?, ?)
            """, (msa_email_id, rec))
        
        conn.commit()
        logger.info(f"✅ Stored MSA analysis results (ID: {msa_email_id})")
        return msa_email_id
        
    except Exception as e:
        logger.error(f"Error storing MSA analysis: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

if __name__ == "__main__":
    # Setup MSA tables
    db_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    print(f"📂 Setting up MSA tables in: {db_path}")
    create_msa_tables(db_path)
    
    print("\n📊 Verifying table creation...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'msa_%'")
    tables = cursor.fetchall()
    
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        print(f"\n✅ Table: {table[0]}")
        for col in columns:
            print(f"   - {col[1]} ({col[2]})")
    
    conn.close()
    print("\n✅ MSA database setup complete!")
"""
Feedback Database Management for STORY-005
==========================================

Manages feedback collection, storage, and analytics for the human-in-the-loop
feedback system with ADK evaluation integration.
"""

import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class FeedbackDatabase:
    """Manages feedback database operations and schema."""
    
    def __init__(self, database_path: str = None):
        """Initialize feedback database."""
        if database_path is None:
            database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._create_schema()
        logger.info(f"✅ Feedback database initialized at {self.database_path}")
    
    def _create_schema(self):
        """Create feedback database schema."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Main feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    corrected_response TEXT,
                    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
                    thumbs_vote TEXT CHECK (thumbs_vote IN ('up', 'down', NULL)),
                    categories TEXT, -- JSON array of category tags
                    user_comments TEXT,
                    user_id TEXT DEFAULT 'anonymous',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, message_id)
                )
            """)
            
            # Feedback metrics for analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_responses INTEGER DEFAULT 0,
                    feedback_count INTEGER DEFAULT 0,
                    feedback_rate REAL DEFAULT 0.0,
                    avg_rating REAL DEFAULT 0.0,
                    thumbs_up INTEGER DEFAULT 0,
                    thumbs_down INTEGER DEFAULT 0,
                    response_accuracy REAL DEFAULT 0.0,
                    helpfulness_score REAL DEFAULT 0.0,
                    completeness_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            # ADK evalset generation tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evalset_generation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evalset_id TEXT NOT NULL UNIQUE,
                    feedback_count INTEGER NOT NULL,
                    min_feedback_id INTEGER NOT NULL,
                    max_feedback_id INTEGER NOT NULL,
                    evalset_content TEXT NOT NULL, -- JSON content
                    file_path TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    validation_status TEXT DEFAULT 'pending'
                )
            """)
            
            # Improvement tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS improvement_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_start DATE NOT NULL,
                    period_end DATE NOT NULL,
                    baseline_accuracy REAL,
                    current_accuracy REAL,
                    improvement_percentage REAL,
                    feedback_samples_used INTEGER,
                    recommendations TEXT, -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session_id ON feedback(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_thumbs ON feedback(thumbs_vote)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_date ON feedback_metrics(date)")
            
            conn.commit()
            logger.info("✅ Feedback database schema created successfully")
    
    def save_feedback(self, feedback_data: Dict[str, Any]) -> int:
        """Save feedback to database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Convert categories to JSON if it's a list
                categories = feedback_data.get('categories')
                if isinstance(categories, list):
                    categories = json.dumps(categories)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO feedback (
                        session_id, message_id, user_query, assistant_response,
                        corrected_response, rating, thumbs_vote, categories,
                        user_comments, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_data['session_id'],
                    feedback_data['message_id'],
                    feedback_data['user_query'],
                    feedback_data['assistant_response'],
                    feedback_data.get('corrected_response'),
                    feedback_data.get('rating'),
                    feedback_data.get('thumbs_vote'),
                    categories,
                    feedback_data.get('user_comments'),
                    feedback_data.get('user_id', 'anonymous')
                ))
                
                feedback_id = cursor.lastrowid
                conn.commit()
                
                # Update daily metrics
                self._update_daily_metrics()
                
                logger.info(f"✅ Feedback saved with ID: {feedback_id}")
                return feedback_id
                
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            raise
    
    def get_feedback(self, session_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve feedback from database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute("""
                        SELECT * FROM feedback 
                        WHERE session_id = ? 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM feedback 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (limit,))
                
                rows = cursor.fetchall()
                
                # Convert to dictionaries and parse JSON fields
                feedback_list = []
                for row in rows:
                    feedback = dict(row)
                    # Parse categories JSON
                    if feedback['categories']:
                        try:
                            feedback['categories'] = json.loads(feedback['categories'])
                        except json.JSONDecodeError:
                            feedback['categories'] = []
                    else:
                        feedback['categories'] = []
                    feedback_list.append(feedback)
                
                return feedback_list
                
        except Exception as e:
            logger.error(f"Failed to retrieve feedback: {e}")
            return []
    
    def get_feedback_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback analytics and metrics."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        AVG(rating) as avg_rating,
                        SUM(CASE WHEN thumbs_vote = 'up' THEN 1 ELSE 0 END) as thumbs_up,
                        SUM(CASE WHEN thumbs_vote = 'down' THEN 1 ELSE 0 END) as thumbs_down,
                        COUNT(DISTINCT session_id) as unique_sessions
                    FROM feedback
                    WHERE created_at >= datetime('now', '-{} days')
                """.format(days))
                
                stats = dict(cursor.fetchone())
                
                # Daily trends
                cursor.execute("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as daily_feedback,
                        AVG(rating) as daily_avg_rating,
                        SUM(CASE WHEN thumbs_vote = 'up' THEN 1 ELSE 0 END) as daily_thumbs_up,
                        SUM(CASE WHEN thumbs_vote = 'down' THEN 1 ELSE 0 END) as daily_thumbs_down
                    FROM feedback
                    WHERE created_at >= datetime('now', '-{} days')
                    GROUP BY DATE(created_at)
                    ORDER BY DATE(created_at)
                """.format(days))
                
                daily_trends = [dict(row) for row in cursor.fetchall()]
                
                # Category analysis
                cursor.execute("""
                    SELECT categories, COUNT(*) as count
                    FROM feedback
                    WHERE categories IS NOT NULL 
                    AND created_at >= datetime('now', '-{} days')
                    GROUP BY categories
                    ORDER BY count DESC
                    LIMIT 10
                """.format(days))
                
                category_stats = []
                for row in cursor.fetchall():
                    try:
                        categories = json.loads(row[0]) if row[0] else []
                        category_stats.append({
                            'categories': categories,
                            'count': row[1]
                        })
                    except json.JSONDecodeError:
                        continue
                
                return {
                    'overview': stats,
                    'daily_trends': daily_trends,
                    'category_analysis': category_stats,
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"Failed to get feedback metrics: {e}")
            return {'overview': {}, 'daily_trends': [], 'category_analysis': []}
    
    def _update_daily_metrics(self):
        """Update daily aggregated metrics."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                today = datetime.now().date()
                
                # Calculate daily metrics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_feedback,
                        AVG(rating) as avg_rating,
                        SUM(CASE WHEN thumbs_vote = 'up' THEN 1 ELSE 0 END) as thumbs_up,
                        SUM(CASE WHEN thumbs_vote = 'down' THEN 1 ELSE 0 END) as thumbs_down
                    FROM feedback
                    WHERE DATE(created_at) = ?
                """, (today,))
                
                result = cursor.fetchone()
                
                if result and result[0] > 0:
                    # Insert or update daily metrics
                    cursor.execute("""
                        INSERT OR REPLACE INTO feedback_metrics (
                            date, feedback_count, avg_rating, thumbs_up, thumbs_down,
                            feedback_rate, response_accuracy, helpfulness_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        today,
                        result[0],
                        result[1] or 0,
                        result[2] or 0,
                        result[3] or 0,
                        min(100.0, (result[0] / max(1, result[0])) * 100),  # feedback_rate placeholder
                        (result[1] or 0) / 5.0 * 100 if result[1] else 0,  # response_accuracy placeholder
                        (result[1] or 0) / 5.0 * 100 if result[1] else 0   # helpfulness_score placeholder
                    ))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to update daily metrics: {e}")
    
    def generate_evalset(self, min_feedback_count: int = 10) -> Optional[Dict[str, Any]]:
        """Generate ADK evalset from feedback data."""
        try:
            # Get feedback suitable for evalset generation
            feedback_data = self.get_feedback(limit=min_feedback_count * 2)
            
            # Filter feedback with corrections or high ratings
            suitable_feedback = [
                f for f in feedback_data 
                if f.get('corrected_response') or (f.get('rating') and f['rating'] >= 4)
            ]
            
            if len(suitable_feedback) < min_feedback_count:
                logger.warning(f"Not enough suitable feedback ({len(suitable_feedback)} < {min_feedback_count})")
                return None
            
            # Generate evalset ID
            evalset_id = f"feedback_evalset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create evalset structure
            evalset = {
                "eval_set_id": evalset_id,
                "name": evalset_id,
                "description": f"Generated from {len(suitable_feedback)} feedback items",
                "eval_cases": [],
                "creation_timestamp": 1755714844.265905  # Static timestamp for compatibility
            }
            
            # Convert feedback to eval cases in ADK format
            for i, feedback in enumerate(suitable_feedback[:min_feedback_count]):
                # Use corrected response if available, otherwise use original with high rating
                expected_response = feedback.get('corrected_response') or feedback['assistant_response']
                
                eval_case = {
                    "eval_id": f"feedback_case_{i+1}_{feedback['id']}",
                    "conversation": [
                        {
                            "invocation_id": f"feedback-{feedback['id']}-user",
                            "user_content": {
                                "parts": [
                                    {
                                        "text": feedback['user_query'],
                                        "video_metadata": None,
                                        "thought": None,
                                        "inline_data": None,
                                        "file_data": None,
                                        "thought_signature": None,
                                        "code_execution_result": None,
                                        "executable_code": None,
                                        "function_call": None,
                                        "function_response": None
                                    }
                                ],
                                "role": "user"
                            },
                            "final_response": {
                                "parts": [
                                    {
                                        "text": expected_response,
                                        "video_metadata": None,
                                        "thought": None,
                                        "inline_data": None,
                                        "file_data": None,
                                        "thought_signature": None,
                                        "code_execution_result": None,
                                        "executable_code": None,
                                        "function_call": None,
                                        "function_response": None
                                    }
                                ],
                                "role": None
                            },
                            "intermediate_data": {
                                "tool_uses": [],
                                "intermediate_responses": []
                            },
                            "creation_timestamp": 1755714844.265905  # Static timestamp for compatibility
                        }
                    ],
                    "session_input": {
                        "app_name": "gcp_security_agent",
                        "user_id": feedback.get('user_id', 'anonymous'),
                        "state": {}
                    },
                    "creation_timestamp": 1755714844.265905,
                    "metadata": {
                        "feedback_id": feedback['id'],
                        "session_id": feedback['session_id'],
                        "human_rating": feedback.get('rating'),
                        "thumbs_vote": feedback.get('thumbs_vote'),
                        "feedback_categories": feedback.get('categories', []),
                        "has_correction": bool(feedback.get('corrected_response')),
                        "created_at": feedback['created_at']
                    }
                }
                
                evalset["eval_cases"].append(eval_case)
            
            # Save evalset to database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO evalset_generation (
                        evalset_id, feedback_count, min_feedback_id, max_feedback_id,
                        evalset_content, validation_status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    evalset_id,
                    len(suitable_feedback),
                    min(f['id'] for f in suitable_feedback),
                    max(f['id'] for f in suitable_feedback),
                    json.dumps(evalset, indent=2),
                    'generated'
                ))
                
                conn.commit()
            
            logger.info(f"✅ Generated evalset {evalset_id} with {len(evalset['eval_cases'])} cases")
            return evalset
            
        except Exception as e:
            logger.error(f"Failed to generate evalset: {e}")
            return None
    
    def save_evalset_to_file(self, evalset: Dict[str, Any], directory: str = None) -> str:
        """Save evalset to .evalset.json file."""
        try:
            if directory is None:
                directory = Path(self.database_path).parent / "evalsets"
            
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            
            filename = f"{evalset['eval_set_id']}.evalset.json"
            file_path = directory / filename
            
            with open(file_path, 'w') as f:
                json.dump(evalset, f, indent=2)
            
            # Update database with file path
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE evalset_generation 
                    SET file_path = ?, validation_status = 'saved'
                    WHERE evalset_id = ?
                """, (str(file_path), evalset['eval_set_id']))
                conn.commit()
            
            logger.info(f"✅ Evalset saved to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save evalset to file: {e}")
            raise

# Initialize global feedback database instance
feedback_db = FeedbackDatabase()
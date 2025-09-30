"""
Confluence Documentation Tools for ADK Security Agent

These tools enable the agent to search, retrieve, and analyze Confluence documentation
for security policies, procedures, and compliance materials.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib
import sqlite3
from pathlib import Path

# Try to import the Confluence service
try:
    import sys
    # Add backend services to path
    backend_path = Path(__file__).parent.parent.parent.parent / "backend" / "services"
    if backend_path.exists():
        sys.path.insert(0, str(backend_path))
    from confluence_service import ConfluenceService
    CONFLUENCE_AVAILABLE = True
except ImportError:
    CONFLUENCE_AVAILABLE = False
    logging.warning("Confluence service not available. Tools will use cached data only.")

logger = logging.getLogger(__name__)

# Configuration from environment
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "")
CONFLUENCE_SPACES = os.getenv("CONFLUENCE_SPACES", "SEC,POLICY,GCP").split(",")
CACHE_DB_PATH = os.getenv("CONFLUENCE_CACHE_DB", "backend/cache/confluence_cache.db")
CACHE_TTL_HOURS = int(os.getenv("CONFLUENCE_CACHE_TTL_HOURS", "6"))

# Initialize cache database
def init_cache_db():
    """Initialize the SQLite cache database for Confluence data."""
    db_path = Path(CACHE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS confluence_documents (
            document_id TEXT PRIMARY KEY,
            space_key TEXT,
            title TEXT,
            content TEXT,
            url TEXT,
            created_date TIMESTAMP,
            modified_date TIMESTAMP,
            created_by TEXT,
            modified_by TEXT,
            parent_id TEXT,
            labels TEXT,
            content_hash TEXT,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS confluence_search_cache (
            query_hash TEXT PRIMARY KEY,
            query TEXT,
            spaces TEXT,
            results TEXT,
            result_count INTEGER,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS confluence_metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_space_key ON confluence_documents(space_key)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_modified_date ON confluence_documents(modified_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cached_at ON confluence_documents(cached_at)")

    conn.commit()
    conn.close()

# Initialize database on module load
init_cache_db()

def search_confluence_documentation(
    query: str,
    spaces: Optional[List[str]] = None,
    limit: int = 10,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Search Confluence documentation for security policies and procedures.

    Args:
        query: Search query string (supports CQL syntax)
        spaces: List of Confluence spaces to search (default: configured spaces)
        limit: Maximum number of results to return (default: 10)
        use_cache: Whether to check cache first (default: True)

    Returns:
        Dictionary containing search results with document metadata

    Example:
        >>> search_confluence_documentation("GCP security policies")
        >>> search_confluence_documentation("IAM best practices", spaces=["SEC"])
    """
    try:
        spaces = spaces or CONFLUENCE_SPACES
        query_hash = hashlib.md5(f"{query}:{','.join(spaces)}:{limit}".encode()).hexdigest()

        # Check cache first if enabled
        if use_cache:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cursor = conn.cursor()

            # Check if we have cached results that are still fresh
            cache_cutoff = datetime.now() - timedelta(hours=CACHE_TTL_HOURS)
            cursor.execute("""
                SELECT results, result_count, cached_at
                FROM confluence_search_cache
                WHERE query_hash = ? AND cached_at > ?
            """, (query_hash, cache_cutoff))

            cached = cursor.fetchone()
            if cached:
                logger.info(f"📚 Cache hit for Confluence search: {query}")
                conn.close()
                return {
                    "success": True,
                    "query": query,
                    "spaces": spaces,
                    "results": json.loads(cached[0]),
                    "count": cached[1],
                    "source": "cache",
                    "cached_at": cached[2]
                }
            conn.close()

        # Try to fetch from Confluence API if available
        if CONFLUENCE_AVAILABLE and CONFLUENCE_URL:
            try:
                service = ConfluenceService()
                results = service.search_documents(
                    query=query,
                    spaces=spaces,
                    limit=limit
                )

                # Cache the results
                if results.get("success"):
                    conn = sqlite3.connect(CACHE_DB_PATH)
                    cursor = conn.cursor()

                    # Store search results in cache
                    cursor.execute("""
                        INSERT OR REPLACE INTO confluence_search_cache
                        (query_hash, query, spaces, results, result_count, cached_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        query_hash,
                        query,
                        ",".join(spaces),
                        json.dumps(results.get("results", [])),
                        len(results.get("results", [])),
                        datetime.now()
                    ))

                    # Also cache individual documents
                    for doc in results.get("results", []):
                        cursor.execute("""
                            INSERT OR REPLACE INTO confluence_documents
                            (document_id, space_key, title, content, url,
                             created_date, modified_date, created_by, modified_by,
                             parent_id, labels, content_hash, cached_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            doc.get("id"),
                            doc.get("space_key"),
                            doc.get("title"),
                            doc.get("content", ""),
                            doc.get("url"),
                            doc.get("created_date"),
                            doc.get("modified_date"),
                            doc.get("created_by"),
                            doc.get("modified_by"),
                            doc.get("parent_id"),
                            json.dumps(doc.get("labels", [])),
                            hashlib.md5(doc.get("content", "").encode()).hexdigest(),
                            datetime.now()
                        ))

                    conn.commit()
                    conn.close()

                    results["source"] = "live"
                    return results

            except Exception as e:
                logger.warning(f"Failed to fetch from Confluence API: {str(e)}")

        # Fall back to cached data if available
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        # Search in cached documents
        search_pattern = f"%{query}%"
        cursor.execute("""
            SELECT document_id, space_key, title, content, url,
                   created_date, modified_date, cached_at
            FROM confluence_documents
            WHERE (title LIKE ? OR content LIKE ?)
            AND space_key IN ({})
            ORDER BY modified_date DESC
            LIMIT ?
        """.format(",".join(["?" for _ in spaces])),
            (search_pattern, search_pattern, *spaces, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "space_key": row[1],
                "title": row[2],
                "excerpt": row[3][:500] if row[3] else "",
                "url": row[4],
                "created_date": row[5],
                "modified_date": row[6],
                "cached_at": row[7]
            })

        conn.close()

        return {
            "success": True,
            "query": query,
            "spaces": spaces,
            "results": results,
            "count": len(results),
            "source": "cache_fallback",
            "message": "Using cached data only"
        }

    except Exception as e:
        logger.error(f"Error searching Confluence: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "spaces": spaces
        }


def get_confluence_document(
    document_id: str,
    use_cache: bool = True,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Retrieve a specific Confluence document by ID.

    Args:
        document_id: Confluence document/page ID
        use_cache: Whether to check cache first (default: True)
        include_content: Whether to include full content (default: True)

    Returns:
        Dictionary containing document details

    Example:
        >>> get_confluence_document("123456789")
    """
    try:
        # Check cache first
        if use_cache:
            conn = sqlite3.connect(CACHE_DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT document_id, space_key, title, content, url,
                       created_date, modified_date, created_by, modified_by,
                       labels, cached_at
                FROM confluence_documents
                WHERE document_id = ?
            """, (document_id,))

            cached = cursor.fetchone()
            if cached:
                conn.close()
                return {
                    "success": True,
                    "document": {
                        "id": cached[0],
                        "space_key": cached[1],
                        "title": cached[2],
                        "content": cached[3] if include_content else None,
                        "url": cached[4],
                        "created_date": cached[5],
                        "modified_date": cached[6],
                        "created_by": cached[7],
                        "modified_by": cached[8],
                        "labels": json.loads(cached[9]) if cached[9] else [],
                        "cached_at": cached[10]
                    },
                    "source": "cache"
                }
            conn.close()

        # Try to fetch from Confluence API
        if CONFLUENCE_AVAILABLE and CONFLUENCE_URL:
            try:
                service = ConfluenceService()
                result = service.get_document(document_id)

                if result.get("success"):
                    # Cache the document
                    doc = result.get("document", {})
                    conn = sqlite3.connect(CACHE_DB_PATH)
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT OR REPLACE INTO confluence_documents
                        (document_id, space_key, title, content, url,
                         created_date, modified_date, created_by, modified_by,
                         labels, content_hash, cached_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        doc.get("id"),
                        doc.get("space_key"),
                        doc.get("title"),
                        doc.get("content", ""),
                        doc.get("url"),
                        doc.get("created_date"),
                        doc.get("modified_date"),
                        doc.get("created_by"),
                        doc.get("modified_by"),
                        json.dumps(doc.get("labels", [])),
                        hashlib.md5(doc.get("content", "").encode()).hexdigest(),
                        datetime.now()
                    ))

                    conn.commit()
                    conn.close()

                    result["source"] = "live"
                    return result

            except Exception as e:
                logger.warning(f"Failed to fetch document from Confluence: {str(e)}")

        return {
            "success": False,
            "error": "Document not found",
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"Error retrieving Confluence document: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "document_id": document_id
        }


def analyze_confluence_coverage(
    topics: List[str],
    spaces: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze documentation coverage for specified security topics.

    Args:
        topics: List of topics to check coverage for
        spaces: Confluence spaces to analyze (default: configured spaces)

    Returns:
        Dictionary containing coverage analysis

    Example:
        >>> analyze_confluence_coverage(["IAM", "Network Security", "Encryption"])
    """
    try:
        spaces = spaces or CONFLUENCE_SPACES
        coverage_results = {}

        for topic in topics:
            # Search for documentation on each topic
            search_result = search_confluence_documentation(
                query=topic,
                spaces=spaces,
                limit=5
            )

            if search_result.get("success"):
                coverage_results[topic] = {
                    "documented": search_result.get("count", 0) > 0,
                    "document_count": search_result.get("count", 0),
                    "documents": [
                        {
                            "title": doc.get("title"),
                            "space": doc.get("space_key"),
                            "modified": doc.get("modified_date")
                        }
                        for doc in search_result.get("results", [])[:3]
                    ]
                }
            else:
                coverage_results[topic] = {
                    "documented": False,
                    "document_count": 0,
                    "documents": []
                }

        # Calculate coverage statistics
        total_topics = len(topics)
        documented_topics = sum(1 for r in coverage_results.values() if r["documented"])
        coverage_percentage = (documented_topics / total_topics * 100) if total_topics > 0 else 0

        return {
            "success": True,
            "topics_analyzed": total_topics,
            "topics_documented": documented_topics,
            "coverage_percentage": round(coverage_percentage, 1),
            "coverage_details": coverage_results,
            "spaces_analyzed": spaces,
            "recommendations": _generate_coverage_recommendations(coverage_results)
        }

    except Exception as e:
        logger.error(f"Error analyzing Confluence coverage: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "topics": topics
        }


def get_confluence_statistics() -> Dict[str, Any]:
    """
    Get statistics about cached Confluence documentation.

    Returns:
        Dictionary containing cache statistics

    Example:
        >>> get_confluence_statistics()
    """
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        # Get document statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total_documents,
                COUNT(DISTINCT space_key) as unique_spaces,
                MIN(cached_at) as oldest_cache,
                MAX(cached_at) as newest_cache,
                AVG(LENGTH(content)) as avg_content_length
            FROM confluence_documents
        """)

        doc_stats = cursor.fetchone()

        # Get space breakdown
        cursor.execute("""
            SELECT space_key, COUNT(*) as count
            FROM confluence_documents
            GROUP BY space_key
            ORDER BY count DESC
        """)

        space_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

        # Get search cache statistics
        cursor.execute("""
            SELECT
                COUNT(*) as cached_searches,
                AVG(result_count) as avg_results_per_search
            FROM confluence_search_cache
        """)

        search_stats = cursor.fetchone()

        # Get recent documents
        cursor.execute("""
            SELECT title, space_key, modified_date
            FROM confluence_documents
            ORDER BY modified_date DESC
            LIMIT 5
        """)

        recent_docs = [
            {"title": row[0], "space": row[1], "modified": row[2]}
            for row in cursor.fetchall()
        ]

        conn.close()

        # Calculate cache freshness
        if doc_stats[3]:  # newest_cache
            newest_cache = datetime.fromisoformat(doc_stats[3])
            cache_age_hours = (datetime.now() - newest_cache).total_seconds() / 3600
            cache_status = "fresh" if cache_age_hours < CACHE_TTL_HOURS else "stale"
        else:
            cache_age_hours = None
            cache_status = "empty"

        return {
            "success": True,
            "cache_statistics": {
                "total_documents": doc_stats[0] or 0,
                "unique_spaces": doc_stats[1] or 0,
                "space_breakdown": space_breakdown,
                "oldest_cache": doc_stats[2],
                "newest_cache": doc_stats[3],
                "cache_age_hours": round(cache_age_hours, 1) if cache_age_hours else None,
                "cache_status": cache_status,
                "avg_document_size": round(doc_stats[4] or 0)
            },
            "search_statistics": {
                "cached_searches": search_stats[0] or 0,
                "avg_results_per_search": round(search_stats[1] or 0, 1)
            },
            "recent_documents": recent_docs,
            "configuration": {
                "confluence_url": CONFLUENCE_URL if CONFLUENCE_URL else "Not configured",
                "monitored_spaces": CONFLUENCE_SPACES,
                "cache_ttl_hours": CACHE_TTL_HOURS,
                "api_available": CONFLUENCE_AVAILABLE
            }
        }

    except Exception as e:
        logger.error(f"Error getting Confluence statistics: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def refresh_confluence_cache(
    spaces: Optional[List[str]] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Refresh the Confluence cache by fetching latest documents.

    Args:
        spaces: Specific spaces to refresh (default: all configured)
        force: Force refresh even if cache is fresh (default: False)

    Returns:
        Dictionary containing refresh status

    Example:
        >>> refresh_confluence_cache(spaces=["SEC"], force=True)
    """
    try:
        spaces = spaces or CONFLUENCE_SPACES

        if not CONFLUENCE_AVAILABLE or not CONFLUENCE_URL:
            return {
                "success": False,
                "error": "Confluence API not available",
                "message": "Cannot refresh cache without API access"
            }

        # Check if refresh is needed
        if not force:
            stats = get_confluence_statistics()
            if stats.get("cache_statistics", {}).get("cache_status") == "fresh":
                return {
                    "success": True,
                    "message": "Cache is still fresh",
                    "cache_age_hours": stats["cache_statistics"]["cache_age_hours"]
                }

        service = ConfluenceService()
        documents_fetched = 0
        errors = []

        for space in spaces:
            try:
                # Search for all documents in space
                results = service.search_documents(
                    query=f"space = {space}",
                    spaces=[space],
                    limit=100  # Fetch more documents
                )

                if results.get("success"):
                    # Cache all documents
                    conn = sqlite3.connect(CACHE_DB_PATH)
                    cursor = conn.cursor()

                    for doc in results.get("results", []):
                        cursor.execute("""
                            INSERT OR REPLACE INTO confluence_documents
                            (document_id, space_key, title, content, url,
                             created_date, modified_date, created_by, modified_by,
                             labels, content_hash, cached_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            doc.get("id"),
                            doc.get("space_key", space),
                            doc.get("title"),
                            doc.get("content", ""),
                            doc.get("url"),
                            doc.get("created_date"),
                            doc.get("modified_date"),
                            doc.get("created_by"),
                            doc.get("modified_by"),
                            json.dumps(doc.get("labels", [])),
                            hashlib.md5(doc.get("content", "").encode()).hexdigest(),
                            datetime.now()
                        ))
                        documents_fetched += 1

                    conn.commit()
                    conn.close()

            except Exception as e:
                errors.append(f"Error refreshing space {space}: {str(e)}")
                logger.error(f"Error refreshing space {space}: {str(e)}")

        return {
            "success": len(errors) == 0,
            "documents_fetched": documents_fetched,
            "spaces_refreshed": spaces,
            "errors": errors if errors else None,
            "message": f"Refreshed {documents_fetched} documents from {len(spaces)} spaces"
        }

    except Exception as e:
        logger.error(f"Error refreshing Confluence cache: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def _generate_coverage_recommendations(coverage_results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on coverage analysis."""
    recommendations = []

    for topic, coverage in coverage_results.items():
        if not coverage["documented"]:
            recommendations.append(f"📝 Create documentation for '{topic}' - No existing documents found")
        elif coverage["document_count"] < 3:
            recommendations.append(f"📚 Expand documentation for '{topic}' - Only {coverage['document_count']} document(s) found")

    if not recommendations:
        recommendations.append("✅ Good documentation coverage for all analyzed topics")

    return recommendations


# Export all tools for agent usage
__all__ = [
    'search_confluence_documentation',
    'get_confluence_document',
    'analyze_confluence_coverage',
    'get_confluence_statistics',
    'refresh_confluence_cache'
]
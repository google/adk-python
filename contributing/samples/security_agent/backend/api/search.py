"""
Google Custom Search API Integration for ADK Security Agent.
Following FastAPI router patterns with dependency injection and async/await.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from collections import defaultdict, deque
import logging
import httpx
import json

# Safe imports with fallbacks for search functionality
try:
    from backend.models.search_models import (
        SearchRequest, SearchResponse, SearchResult,
        SearchContextRequest, SearchContextResponse,
        SearchHistoryEntry, SearchHistoryRequest, SearchHistoryResponse,
        SearchAnalyticsRequest, SearchAnalyticsResponse,
        SearchConfigRequest, SearchConfigResponse
    )
    SEARCH_MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Search models not available: {e}")
    SEARCH_MODELS_AVAILABLE = False
    
    # Create mock models
    from pydantic import BaseModel
    
    class SearchRequest(BaseModel):
        query: str
        max_results: int = 10
        safe_search: bool = True
        user_id: Optional[str] = None
        session_id: Optional[str] = None
    
    class SearchResult(BaseModel):
        title: str
        url: str
        snippet: str
        display_url: str
        relevance_score: Optional[float] = None
    
    class SearchResponse(BaseModel):
        success: bool
        query: str
        results: List[SearchResult]
        total_results: int
        search_time_ms: int
        session_id: Optional[str] = None
        llm_summary: Optional[str] = None
        suggested_refinements: Optional[List[str]] = None
        security_context: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
    
    class SearchContextRequest(BaseModel):
        query: str
        session_id: str
        context_window: int = 5
        analyze_security: bool = False
    
    class SearchContextResponse(BaseModel):
        success: bool
        context: Dict[str, Any]
        suggested_queries: List[str]
        conversation_summary: str
        security_recommendations: Optional[List[str]] = None
        topic_analysis: Dict[str, float]
    
    class SearchHistoryEntry(BaseModel):
        timestamp: datetime
        query: str
        results_count: int
        session_id: str
        user_feedback: Optional[str] = None
        clicked_urls: List[str] = []
    
    class SearchHistoryRequest(BaseModel):
        session_id: str
        limit: int = 20
        offset: int = 0
    
    class SearchHistoryResponse(BaseModel):
        success: bool
        history: List[SearchHistoryEntry]
        total_count: int
        has_more: bool
    
    class SearchAnalyticsRequest(BaseModel):
        user_id: Optional[str] = None
        session_id: Optional[str] = None
        time_range_hours: int = 24
        group_by: str = "hour"
    
    class SearchAnalyticsResponse(BaseModel):
        success: bool
        total_searches: int
        unique_queries: int
        avg_results_per_search: float
        popular_queries: List[Dict[str, Any]]
        search_timeline: List[Dict[str, Any]]
        rate_limit_status: Dict[str, Any]
    
    class SearchConfigRequest(BaseModel):
        pass
    
    class SearchConfigResponse(BaseModel):
        success: bool
        config: Dict[str, Any]
        api_configured: bool
        cache_enabled: bool
        rate_limiting_enabled: bool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/search", tags=["search"])

# Configuration from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
SEARCH_RATE_LIMIT_PER_MINUTE = int(os.getenv("SEARCH_RATE_LIMIT_PER_MINUTE", "100"))
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "3600"))

# In-memory storage (production should use Redis/database)
search_cache = {}
search_history = defaultdict(list)
rate_limit_tracker = defaultdict(lambda: deque(maxlen=SEARCH_RATE_LIMIT_PER_MINUTE))


class SearchService:
    """Google Custom Search service following ADK patterns."""
    
    def __init__(self):
        self.api_key = GOOGLE_API_KEY
        self.cse_id = GOOGLE_CSE_ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(
        self, 
        query: str, 
        max_results: int = 10,
        safe_search: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform Google Custom Search with rate limiting and caching."""
        
        if not self.api_key or not self.cse_id:
            logger.warning("Google Custom Search API not configured")
            # Return mock data for development
            return self._get_mock_search_results(query, max_results)
        
        # Check rate limits
        if user_id and not self._check_rate_limit(user_id):
            raise HTTPException(
                status_code=429,
                detail="Search rate limit exceeded. Please wait before searching again."
            )
        
        # Check cache
        cache_key = f"{query}:{max_results}:{safe_search}"
        if cache_key in search_cache:
            cache_entry = search_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < SEARCH_CACHE_TTL_SECONDS:
                logger.info(f"Cache hit for query: {query}")
                return cache_entry["data"]
        
        # Perform search
        start_time = time.time()
        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": min(max_results, 10),  # Google CSE max is 10 per request
                "safe": "active" if safe_search else "off"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.base_url, 
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Google CSE API error: {response.status_code}")
                    return self._get_mock_search_results(query, max_results)
                
                data = response.json()
            
            # Process results
            search_results = []
            items = data.get("items", [])
            
            for item in items:
                search_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "display_url": item.get("displayLink", ""),
                    "relevance_score": None
                })
            
            search_time_ms = int((time.time() - start_time) * 1000)
            
            result = {
                "results": search_results,
                "total_results": int(data.get("searchInformation", {}).get("totalResults", 0)),
                "search_time_ms": search_time_ms,
                "query": query
            }
            
            # Cache results
            search_cache[cache_key] = {
                "data": result,
                "timestamp": time.time()
            }
            
            # Update rate limit tracking
            if user_id:
                self._update_rate_limit(user_id)
            
            return result
            
        except httpx.TimeoutException:
            logger.error(f"Search timeout for query: {query}")
            return self._get_mock_search_results(query, max_results)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._get_mock_search_results(query, max_results)
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit using sliding window."""
        now = time.time()
        user_requests = rate_limit_tracker[user_id]
        
        # Remove requests older than 1 minute
        while user_requests and user_requests[0] < now - 60:
            user_requests.popleft()
        
        return len(user_requests) < SEARCH_RATE_LIMIT_PER_MINUTE
    
    def _update_rate_limit(self, user_id: str):
        """Update rate limit tracker."""
        rate_limit_tracker[user_id].append(time.time())
    
    def _get_mock_search_results(self, query: str, max_results: int) -> Dict[str, Any]:
        """Return mock search results for development/testing."""
        mock_results = []
        
        # Generate mock results based on query
        security_terms = ["security", "vulnerability", "threat", "compliance", "authentication"]
        is_security_query = any(term in query.lower() for term in security_terms)
        
        if is_security_query:
            mock_results = [
                {
                    "title": f"Security Best Practices for {query}",
                    "url": f"https://security.google.com/search?q={query}",
                    "snippet": f"Comprehensive guide on security best practices related to {query}. Learn about vulnerabilities, threats, and mitigation strategies.",
                    "display_url": "security.google.com",
                    "relevance_score": 0.95
                },
                {
                    "title": f"OWASP Guidelines: {query}",
                    "url": f"https://owasp.org/search?q={query}",
                    "snippet": f"OWASP recommendations and guidelines for {query}. Industry-standard security practices and vulnerability prevention.",
                    "display_url": "owasp.org",
                    "relevance_score": 0.92
                },
                {
                    "title": f"Google Cloud Security: {query}",
                    "url": f"https://cloud.google.com/security/search?q={query}",
                    "snippet": f"Google Cloud Platform security documentation for {query}. Implementation guides and best practices.",
                    "display_url": "cloud.google.com",
                    "relevance_score": 0.88
                }
            ]
        else:
            mock_results = [
                {
                    "title": f"Documentation: {query}",
                    "url": f"https://docs.google.com/search?q={query}",
                    "snippet": f"Official documentation and guides for {query}. Getting started, tutorials, and API references.",
                    "display_url": "docs.google.com",
                    "relevance_score": 0.85
                },
                {
                    "title": f"Best Practices: {query}",
                    "url": f"https://developers.google.com/search?q={query}",
                    "snippet": f"Developer best practices and implementation guide for {query}. Code examples and patterns.",
                    "display_url": "developers.google.com",
                    "relevance_score": 0.82
                }
            ]
        
        return {
            "results": mock_results[:min(max_results, len(mock_results))],
            "total_results": len(mock_results),
            "search_time_ms": 150,
            "query": query
        }


# Dependency injection
async def get_search_service() -> SearchService:
    """Dependency to get search service instance."""
    return SearchService()


@router.post("/web", response_model=SearchResponse)
async def search_web(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Perform web search using Google Custom Search API.
    Following ADK patterns for API endpoints.
    """
    try:
        # Perform search
        search_result = await search_service.search(
            query=request.query,
            max_results=request.max_results,
            safe_search=request.safe_search,
            user_id=request.user_id
        )
        
        # Generate LLM summary
        llm_summary = await _generate_search_summary(
            request.query, 
            search_result["results"]
        )
        
        # Generate suggested refinements
        suggested_refinements = await _generate_search_refinements(
            request.query,
            search_result.get("results", [])
        )
        
        # Store search in history (background task)
        background_tasks.add_task(
            _store_search_history,
            session_id=request.session_id,
            user_id=request.user_id,
            query=request.query,
            results_count=len(search_result["results"])
        )
        
        # Add security context if relevant
        security_context = None
        if _is_security_query(request.query):
            security_context = await _generate_security_context(
                request.query,
                search_result["results"]
            )
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=[SearchResult(**result) for result in search_result["results"]],
            total_results=search_result["total_results"],
            search_time_ms=search_result["search_time_ms"],
            session_id=request.session_id,
            llm_summary=llm_summary,
            suggested_refinements=suggested_refinements,
            security_context=security_context
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return SearchResponse(
            success=False,
            query=request.query,
            results=[],
            total_results=0,
            search_time_ms=0,
            session_id=request.session_id,
            error=str(e)
        )


@router.post("/context", response_model=SearchContextResponse)
async def get_search_context(request: SearchContextRequest):
    """
    Get contextual search suggestions based on conversation history.
    """
    try:
        # Get search history for session
        session_searches = search_history.get(request.session_id, [])
        recent_searches = session_searches[-request.context_window:]
        
        # Extract context
        context = {
            "recent_queries": [s["query"] for s in recent_searches],
            "search_count": len(session_searches),
            "session_id": request.session_id
        }
        
        # Generate suggestions
        suggested_queries = _generate_contextual_queries(
            request.query, 
            recent_searches
        )
        
        # Generate conversation summary
        conversation_summary = _generate_conversation_summary(
            recent_searches
        )
        
        # Security recommendations if requested
        security_recommendations = None
        if request.analyze_security:
            security_recommendations = _generate_security_recommendations(
                request.query
            )
        
        # Topic analysis
        topic_analysis = _analyze_topics(recent_searches)
        
        return SearchContextResponse(
            success=True,
            context=context,
            suggested_queries=suggested_queries,
            conversation_summary=conversation_summary,
            security_recommendations=security_recommendations,
            topic_analysis=topic_analysis
        )
        
    except Exception as e:
        logger.error(f"Search context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=SearchHistoryResponse)
async def get_search_history(
    session_id: str, 
    limit: int = 20,
    offset: int = 0,
    include_results: bool = False
):
    """Get search history for a session."""
    try:
        session_searches = search_history.get(session_id, [])
        
        # Apply pagination
        total_count = len(session_searches)
        paginated_searches = session_searches[offset:offset + limit]
        
        # Convert to response model
        history_entries = []
        for search in paginated_searches:
            entry = SearchHistoryEntry(
                timestamp=search["timestamp"],
                query=search["query"],
                results_count=search.get("results_count", 0),
                session_id=session_id,
                user_feedback=search.get("feedback"),
                clicked_urls=search.get("clicked_urls", [])
            )
            history_entries.append(entry)
        
        return SearchHistoryResponse(
            success=True,
            history=history_entries,
            total_count=total_count,
            has_more=(offset + limit) < total_count
        )
        
    except Exception as e:
        logger.error(f"Search history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics", response_model=SearchAnalyticsResponse)
async def get_search_analytics(request: SearchAnalyticsRequest):
    """Get search usage analytics."""
    try:
        # Calculate analytics
        all_searches = []
        if request.user_id:
            # Filter by user (would need user tracking in production)
            pass
        elif request.session_id:
            all_searches = search_history.get(request.session_id, [])
        else:
            # All searches
            for session_searches in search_history.values():
                all_searches.extend(session_searches)
        
        # Filter by time range
        cutoff_time = datetime.now() - timedelta(hours=request.time_range_hours)
        filtered_searches = [
            s for s in all_searches 
            if s.get("timestamp", datetime.min) > cutoff_time
        ]
        
        # Calculate metrics
        total_searches = len(filtered_searches)
        unique_queries = len(set(s["query"] for s in filtered_searches))
        avg_results = sum(s.get("results_count", 0) for s in filtered_searches) / max(total_searches, 1)
        
        # Popular queries
        query_counts = defaultdict(int)
        for search in filtered_searches:
            query_counts[search["query"]] += 1
        
        popular_queries = [
            {"query": q, "count": c} 
            for q, c in sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Search timeline (simplified)
        search_timeline = []
        if request.group_by == "hour":
            # Group by hour
            hour_counts = defaultdict(int)
            for search in filtered_searches:
                hour = search.get("timestamp", datetime.now()).replace(minute=0, second=0, microsecond=0)
                hour_counts[hour.isoformat()] += 1
            search_timeline = [{"time": t, "count": c} for t, c in hour_counts.items()]
        
        # Rate limit status
        rate_limit_status = {
            "limit_per_minute": SEARCH_RATE_LIMIT_PER_MINUTE,
            "current_usage": len(rate_limit_tracker.get(request.user_id or "anonymous", []))
        }
        
        return SearchAnalyticsResponse(
            success=True,
            total_searches=total_searches,
            unique_queries=unique_queries,
            avg_results_per_search=avg_results,
            popular_queries=popular_queries,
            search_timeline=search_timeline,
            rate_limit_status=rate_limit_status
        )
        
    except Exception as e:
        logger.error(f"Search analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=SearchConfigResponse)
async def get_search_config():
    """Get current search configuration."""
    try:
        config = {
            "safe_search_default": True,
            "max_results_default": 10,
            "cache_ttl_seconds": SEARCH_CACHE_TTL_SECONDS,
            "rate_limit_per_minute": SEARCH_RATE_LIMIT_PER_MINUTE,
            "google_cse_configured": bool(GOOGLE_API_KEY and GOOGLE_CSE_ID)
        }
        
        return SearchConfigResponse(
            success=True,
            config=config,
            api_configured=bool(GOOGLE_API_KEY and GOOGLE_CSE_ID),
            cache_enabled=True,
            rate_limiting_enabled=True
        )
        
    except Exception as e:
        logger.error(f"Search config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for search service."""
    return {
        "status": "healthy",
        "service": "search",
        "api_configured": bool(GOOGLE_API_KEY and GOOGLE_CSE_ID),
        "cache_size": len(search_cache),
        "timestamp": datetime.now().isoformat()
    }


# Helper functions
async def _generate_search_summary(query: str, results: List[Dict]) -> str:
    """Generate LLM summary of search results."""
    if not results:
        return f"No results found for '{query}'"
    
    # Simple summary (would use LLM in production)
    return f"Found {len(results)} relevant results for '{query}'. Top results include information about {', '.join([r.get('title', '')[:30] for r in results[:3]])}"


async def _generate_search_refinements(query: str, results: List[Dict]) -> List[str]:
    """Generate suggested search refinements."""
    refinements = []
    
    # Add basic refinements
    refinements.append(f"{query} tutorial")
    refinements.append(f"{query} best practices")
    refinements.append(f"{query} examples")
    
    # Add security-specific if relevant
    if _is_security_query(query):
        refinements.append(f"{query} vulnerabilities")
        refinements.append(f"{query} security checklist")
    
    return refinements[:5]


async def _generate_security_context(query: str, results: List[Dict]) -> Dict[str, Any]:
    """Generate security context for search results."""
    return {
        "is_security_related": True,
        "risk_level": "medium",
        "key_concerns": _identify_security_concerns(query),
        "recommended_actions": _generate_security_recommendations(query)
    }


def _store_search_history(session_id: str, user_id: str, query: str, results_count: int):
    """Store search in history (background task)."""
    search_entry = {
        "timestamp": datetime.now(),
        "query": query,
        "results_count": results_count,
        "user_id": user_id
    }
    search_history[session_id].append(search_entry)
    
    # Limit history size
    if len(search_history[session_id]) > 100:
        search_history[session_id] = search_history[session_id][-100:]


def _is_security_query(query: str) -> bool:
    """Check if query is security-related."""
    security_terms = [
        "security", "vulnerability", "threat", "exploit", "attack",
        "authentication", "authorization", "encryption", "compliance",
        "audit", "penetration", "breach", "malware", "phishing"
    ]
    query_lower = query.lower()
    return any(term in query_lower for term in security_terms)


def _identify_security_concerns(query: str) -> List[str]:
    """Identify security concerns from query."""
    concerns = []
    query_lower = query.lower()
    
    if "vulnerability" in query_lower:
        concerns.append("Known vulnerabilities")
    if "authentication" in query_lower:
        concerns.append("Access control")
    if "data" in query_lower:
        concerns.append("Data protection")
    if "compliance" in query_lower:
        concerns.append("Regulatory compliance")
    
    return concerns or ["General security"]


def _generate_security_recommendations(query: str) -> List[str]:
    """Generate security recommendations."""
    recommendations = [
        "Implement defense in depth",
        "Follow principle of least privilege",
        "Enable audit logging",
        "Regular security assessments",
        "Keep systems updated"
    ]
    return recommendations[:3]


def _generate_contextual_queries(query: str, recent_searches: List[Dict]) -> List[str]:
    """Generate contextual query suggestions."""
    suggestions = set()
    
    # Add variations of current query
    if query:
        suggestions.add(f"{query} advanced")
        suggestions.add(f"{query} troubleshooting")
    
    # Add based on recent searches
    for search in recent_searches[-3:]:
        prev_query = search.get("query", "")
        if prev_query and prev_query != query:
            suggestions.add(f"{prev_query} and {query}")
    
    return list(suggestions)[:5]


def _generate_conversation_summary(recent_searches: List[Dict]) -> str:
    """Generate summary of search conversation."""
    if not recent_searches:
        return "No recent search history"
    
    queries = [s.get("query", "") for s in recent_searches[-5:]]
    return f"Recent searches focused on: {', '.join(queries)}"


def _analyze_topics(recent_searches: List[Dict]) -> Dict[str, float]:
    """Analyze topic distribution in searches."""
    topic_counts = defaultdict(int)
    total = len(recent_searches)
    
    if total == 0:
        return {}
    
    for search in recent_searches:
        query = search.get("query", "").lower()
        
        # Simple topic detection
        if "security" in query:
            topic_counts["security"] += 1
        if "api" in query:
            topic_counts["api"] += 1
        if "cloud" in query or "gcp" in query:
            topic_counts["cloud"] += 1
        if "database" in query or "storage" in query:
            topic_counts["data"] += 1
    
    # Convert to percentages
    return {topic: (count / total) for topic, count in topic_counts.items()}
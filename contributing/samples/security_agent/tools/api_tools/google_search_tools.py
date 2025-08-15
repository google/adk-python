"""
Google Custom Search Tools for ADK Security Agent.
Following established ADK patterns for tool function implementation.
"""

from typing import Dict, Any, Optional, List
import requests
import json
import os
import logging

# ADK tool context import (adjust path as needed)
# from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

# Backend API base URL (configurable via environment)
API_BASE_URL = os.getenv("ADK_BACKEND_URL", "http://localhost:8000")


def search_web(
    query: str,
    max_results: int = 10,
    safe_search: bool = True,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tool_context: Optional[Any] = None  # ToolContext type
) -> str:
    """
    Perform web search using Google Custom Search API following ADK patterns.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (1-20)
        safe_search: Enable safe search filtering
        session_id: Session identifier for context
        user_id: User identifier for rate limiting
        tool_context: ADK tool execution context
        
    Returns:
        JSON string with search results or error message
    """
    try:
        # Extract session/user from tool context if provided
        if tool_context:
            session_id = session_id or getattr(tool_context, 'session_id', None)
            user_id = user_id or getattr(tool_context, 'user_id', None)
        
        request_data = {
            "query": query,
            "max_results": max_results,
            "safe_search": safe_search,
            "session_id": session_id or "default",
            "user_id": user_id or "anonymous",
            "context": {}
        }
        
        logger.info(f"Performing web search for query: {query}")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/search/web", 
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        
        response_data = response.json()
        
        if response_data.get("success"):
            # Format results for LLM consumption
            formatted_results = _format_search_results(response_data)
            return formatted_results
        else:
            error_msg = response_data.get('error', 'Unknown error')
            logger.error(f"Search error: {error_msg}")
            return f"Search error: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "Search timeout: Request took too long to complete"
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error during search: {e}")
        return f"Network error performing web search: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        return f"Error performing web search: {e}"


def get_search_context(
    session_id: str,
    query: str = "",
    include_history: bool = True,
    context_window: int = 5,
    tool_context: Optional[Any] = None
) -> str:
    """
    Get contextual search suggestions based on conversation history.
    
    Args:
        session_id: Session identifier
        query: Optional query for context analysis
        include_history: Include conversation history
        context_window: Number of recent messages to analyze
        tool_context: ADK tool execution context
        
    Returns:
        JSON string with context and suggestions
    """
    try:
        request_data = {
            "session_id": session_id,
            "query": query,
            "include_history": include_history,
            "context_window": context_window,
            "analyze_security": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/search/context",
            json=request_data,
            timeout=15
        )
        response.raise_for_status()
        
        response_data = response.json()
        
        if response_data.get("success"):
            return json.dumps(response_data, indent=2)
        else:
            return f"Context error: {response_data.get('error', 'Unknown error')}"
        
    except Exception as e:
        logger.error(f"Error getting search context: {e}")
        return f"Error getting search context: {e}"


def search_security_topics(
    query: str,
    session_id: Optional[str] = None,
    tool_context: Optional[Any] = None
) -> str:
    """
    Perform security-focused web search with enhanced context.
    
    Args:
        query: Security-related search query
        session_id: Session identifier
        tool_context: ADK tool execution context
        
    Returns:
        JSON string with security-focused search results
    """
    try:
        # Enhance query with security context
        security_query = f"{query} security vulnerability threat intelligence best practices"
        
        # Perform search
        search_result = search_web(
            query=security_query,
            max_results=15,
            safe_search=True,
            session_id=session_id,
            tool_context=tool_context
        )
        
        # Parse and enhance results with security analysis
        try:
            results_data = json.loads(search_result)
            if isinstance(results_data, dict) and "results" in results_data:
                # Add security context analysis
                results_data["security_analysis"] = {
                    "query_type": "security",
                    "focus_areas": _identify_security_focus(query),
                    "recommended_actions": _generate_security_recommendations(query)
                }
                return json.dumps(results_data, indent=2)
        except json.JSONDecodeError:
            pass
        
        return search_result
        
    except Exception as e:
        logger.error(f"Error in security search: {e}")
        return f"Error performing security search: {e}"


def get_search_history(
    session_id: str,
    limit: int = 20,
    tool_context: Optional[Any] = None
) -> str:
    """
    Retrieve search history for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of history entries
        tool_context: ADK tool execution context
        
    Returns:
        JSON string with search history
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/search/history/{session_id}",
            params={"limit": limit},
            timeout=10
        )
        response.raise_for_status()
        
        return json.dumps(response.json(), indent=2)
        
    except Exception as e:
        logger.error(f"Error getting search history: {e}")
        return f"Error getting search history: {e}"


def get_search_suggestions(
    session_id: str,
    current_topic: str = "",
    tool_context: Optional[Any] = None
) -> str:
    """
    Generate contextual search suggestions based on conversation.
    
    Args:
        session_id: Session identifier
        current_topic: Current conversation topic
        tool_context: ADK tool execution context
        
    Returns:
        JSON string with search suggestions
    """
    try:
        # Get context first
        context_response = get_search_context(
            session_id=session_id,
            query=current_topic,
            include_history=True,
            context_window=10,
            tool_context=tool_context
        )
        
        try:
            context_data = json.loads(context_response)
            if context_data.get("success"):
                suggestions = context_data.get("suggested_queries", [])
                
                # Add topic-specific suggestions
                if current_topic:
                    suggestions.extend(_generate_topic_suggestions(current_topic))
                
                return json.dumps({
                    "success": True,
                    "suggestions": list(set(suggestions))[:10],
                    "topic": current_topic
                }, indent=2)
        except json.JSONDecodeError:
            pass
        
        # Fallback to basic suggestions
        return json.dumps({
            "success": True,
            "suggestions": _generate_topic_suggestions(current_topic)[:5],
            "topic": current_topic
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        return f"Error generating search suggestions: {e}"


# Helper functions
def _format_search_results(response_data: Dict) -> str:
    """Format search results for LLM consumption."""
    try:
        results = response_data.get("results", [])
        query = response_data.get("query", "")
        total = response_data.get("total_results", 0)
        summary = response_data.get("llm_summary", "")
        
        formatted = {
            "query": query,
            "total_results": total,
            "summary": summary,
            "results": []
        }
        
        for idx, result in enumerate(results, 1):
            formatted["results"].append({
                "position": idx,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "relevance": result.get("relevance_score")
            })
        
        # Add suggested refinements if available
        if response_data.get("suggested_refinements"):
            formatted["suggested_refinements"] = response_data["suggested_refinements"]
        
        # Add security context if available
        if response_data.get("security_context"):
            formatted["security_context"] = response_data["security_context"]
        
        return json.dumps(formatted, indent=2)
        
    except Exception as e:
        logger.error(f"Error formatting results: {e}")
        return json.dumps(response_data, indent=2)


def _identify_security_focus(query: str) -> List[str]:
    """Identify security focus areas from query."""
    query_lower = query.lower()
    focus_areas = []
    
    security_keywords = {
        "vulnerability": ["vulnerability", "cve", "exploit", "weakness"],
        "threat": ["threat", "attack", "malware", "ransomware"],
        "compliance": ["compliance", "regulation", "gdpr", "hipaa", "pci"],
        "authentication": ["auth", "login", "password", "mfa", "2fa"],
        "encryption": ["encrypt", "crypto", "tls", "ssl", "certificate"],
        "access_control": ["permission", "iam", "rbac", "access", "privilege"],
        "monitoring": ["monitor", "log", "audit", "siem", "alert"],
        "incident": ["incident", "breach", "response", "forensics"]
    }
    
    for area, keywords in security_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            focus_areas.append(area)
    
    return focus_areas or ["general_security"]


def _generate_security_recommendations(query: str) -> List[str]:
    """Generate security recommendations based on query."""
    recommendations = []
    query_lower = query.lower()
    
    if "vulnerability" in query_lower:
        recommendations.extend([
            "Run vulnerability scans regularly",
            "Apply security patches promptly",
            "Monitor CVE databases for updates"
        ])
    
    if "compliance" in query_lower:
        recommendations.extend([
            "Review compliance requirements",
            "Document security controls",
            "Conduct regular audits"
        ])
    
    if "authentication" in query_lower:
        recommendations.extend([
            "Implement multi-factor authentication",
            "Use strong password policies",
            "Enable account lockout policies"
        ])
    
    if not recommendations:
        recommendations = [
            "Follow security best practices",
            "Keep systems updated",
            "Monitor for security events"
        ]
    
    return recommendations[:5]


def _generate_topic_suggestions(topic: str) -> List[str]:
    """Generate topic-based search suggestions."""
    if not topic:
        return [
            "GCP security best practices",
            "Cloud security monitoring",
            "IAM configuration guide",
            "Security compliance checklist",
            "Threat detection strategies"
        ]
    
    topic_lower = topic.lower()
    suggestions = []
    
    # Add variations of the topic
    suggestions.append(f"{topic} best practices")
    suggestions.append(f"{topic} security guide")
    suggestions.append(f"{topic} implementation")
    suggestions.append(f"{topic} troubleshooting")
    suggestions.append(f"latest {topic} updates")
    
    # Add security-specific suggestions
    if "security" not in topic_lower:
        suggestions.append(f"{topic} security considerations")
    
    return suggestions
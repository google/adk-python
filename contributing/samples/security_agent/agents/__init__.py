# ADK Agent exports with fallback support
try:
    from google.adk import Agent
    from .agent import root_agent, query_security_data
    ADK_AVAILABLE = True
    __all__ = ['Agent', 'root_agent', 'query_security_data']
except ImportError:
    Agent = None
    root_agent = None
    ADK_AVAILABLE = False
    # Fallback query function
    try:
        from ._tools.sqlite_tool import query_security_data
    except ImportError:
        def query_security_data(*args, **kwargs):
            return {"success": False, "error": "SQLite tool not available"}
    __all__ = ['query_security_data']
"""
Input Sanitization Middleware
=============================

Comprehensive input sanitization to prevent injection attacks.
"""

import re
import html
import urllib.parse
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Comprehensive input sanitization for security."""
    
    # Dangerous patterns for SQL injection
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE|OR|AND|HAVING|GROUP BY|ORDER BY)\b)",
        r"(--|#|\/\*|\*\/|;)",
        r"('|\"|`)",
        r"(\bOR\b\s*\d+\s*=\s*\d+)",
        r"(\bOR\b\s*'[^']*'\s*=\s*'[^']*')",
    ]
    
    # Dangerous patterns for NoSQL injection
    NOSQL_PATTERNS = [
        r"(\$ne|\$eq|\$gt|\$gte|\$lt|\$lte|\$in|\$nin)",
        r"(\$or|\$and|\$not|\$nor)",
        r"(\$regex|\$where|\$text)",
        r"(\{|\}|\[|\])",
    ]
    
    # Dangerous patterns for command injection
    COMMAND_PATTERNS = [
        r"(;|\||&|`|\$\(|\))",
        r"(>|<|>>|<<)",
        r"(\bsh\b|\bbash\b|\bcmd\b|\bpowershell\b)",
        r"(\/bin\/|\/usr\/bin\/|\/etc\/)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"(<script[^>]*>.*?</script>)",
        r"(<iframe[^>]*>.*?</iframe>)",
        r"(javascript:|data:text/html)",
        r"(on\w+\s*=)",
        r"(<img[^>]*onerror[^>]*>)",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"(\.\.\/|\.\.\\)",
        r"(%2e%2e%2f|%2e%2e%5c)",
        r"(\/etc\/passwd|\/windows\/system32)",
    ]
    
    @classmethod
    def sanitize_string(cls, value: str, context: str = "general") -> str:
        """Sanitize a string value based on context."""
        if not value:
            return value
            
        # HTML escape for XSS prevention
        value = html.escape(value)
        
        # URL decode to catch encoded attacks
        try:
            decoded_value = urllib.parse.unquote(value)
        except:
            decoded_value = value
        
        # Check for dangerous patterns based on context
        if context in ["sql", "database", "query"]:
            for pattern in cls.SQL_PATTERNS:
                if re.search(pattern, decoded_value, re.IGNORECASE):
                    logger.warning(f"SQL injection pattern detected: {pattern}")
                    # Remove dangerous characters
                    value = re.sub(r"[;'\"`\-#/*]", "", value)
                    break
        
        elif context in ["nosql", "json", "mongo"]:
            for pattern in cls.NOSQL_PATTERNS:
                if re.search(pattern, decoded_value, re.IGNORECASE):
                    logger.warning(f"NoSQL injection pattern detected: {pattern}")
                    # Remove dangerous characters
                    value = re.sub(r"[${}[\]]", "", value)
                    break
        
        elif context in ["command", "shell", "exec"]:
            for pattern in cls.COMMAND_PATTERNS:
                if re.search(pattern, decoded_value, re.IGNORECASE):
                    logger.warning(f"Command injection pattern detected: {pattern}")
                    # Remove dangerous characters
                    value = re.sub(r"[;|&`$()<>]", "", value)
                    break
        
        elif context in ["path", "file", "directory"]:
            for pattern in cls.PATH_TRAVERSAL_PATTERNS:
                if re.search(pattern, decoded_value, re.IGNORECASE):
                    logger.warning(f"Path traversal pattern detected: {pattern}")
                    # Remove path traversal sequences
                    value = value.replace("..", "").replace("//", "/").replace("\\\\", "\\")
                    break
        
        # General XSS prevention
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, decoded_value, re.IGNORECASE):
                logger.warning(f"XSS pattern detected: {pattern}")
                # Strip tags
                value = re.sub(r"<[^>]+>", "", value)
                break
        
        # Limit string length to prevent buffer overflow
        max_length = 1000
        if len(value) > max_length:
            value = value[:max_length]
        
        return value
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], context: str = "general") -> Dict[str, Any]:
        """Recursively sanitize dictionary values."""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize the key itself
            safe_key = cls.sanitize_string(str(key), context)
            
            # Sanitize the value based on type
            if isinstance(value, str):
                sanitized[safe_key] = cls.sanitize_string(value, context)
            elif isinstance(value, dict):
                sanitized[safe_key] = cls.sanitize_dict(value, context)
            elif isinstance(value, list):
                sanitized[safe_key] = cls.sanitize_list(value, context)
            else:
                sanitized[safe_key] = value
        
        return sanitized
    
    @classmethod
    def sanitize_list(cls, data: List[Any], context: str = "general") -> List[Any]:
        """Sanitize list values."""
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                sanitized.append(cls.sanitize_string(item, context))
            elif isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item, context))
            elif isinstance(item, list):
                sanitized.append(cls.sanitize_list(item, context))
            else:
                sanitized.append(item)
        
        return sanitized
    
    @classmethod
    def validate_and_sanitize_query_params(cls, params: Dict[str, str]) -> Dict[str, str]:
        """Validate and sanitize query parameters."""
        sanitized = {}
        
        for key, value in params.items():
            # Determine context based on parameter name
            context = "general"
            if key in ["query", "search", "filter", "where"]:
                context = "sql"
            elif key in ["path", "file", "dir", "directory"]:
                context = "path"
            elif key in ["cmd", "command", "exec"]:
                context = "command"
            
            # Sanitize based on context
            safe_key = cls.sanitize_string(key, "general")
            safe_value = cls.sanitize_string(value, context)
            
            sanitized[safe_key] = safe_value
        
        return sanitized
    
    @classmethod
    def is_safe_json(cls, json_str: str) -> bool:
        """Check if JSON string is safe from injection."""
        try:
            # Check for NoSQL injection patterns
            for pattern in cls.NOSQL_PATTERNS:
                if re.search(pattern, json_str, re.IGNORECASE):
                    return False
            
            # Check for executable code patterns
            if re.search(r"(function\s*\(|eval\s*\(|new\s+Function)", json_str, re.IGNORECASE):
                return False
            
            return True
        except:
            return False
    
    @classmethod
    def create_safe_sql_query(cls, query_template: str, params: Dict[str, Any]) -> tuple:
        """Create a safe parameterized SQL query."""
        # Use parameterized queries - never concatenate user input
        # This returns the template and sanitized parameters separately
        sanitized_params = {}
        
        for key, value in params.items():
            if isinstance(value, str):
                # Sanitize string parameters
                sanitized_params[key] = cls.sanitize_string(value, "sql")
            else:
                sanitized_params[key] = value
        
        return query_template, sanitized_params
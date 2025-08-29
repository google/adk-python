"""
Enhanced Tool Response Handler
===============================

Provides improved error handling and response formatting for agent tools
based on ADK v1.13.0 patterns.
"""

import json
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ToolResponseHandler:
    """Handles tool responses with improved error handling and formatting."""
    
    @staticmethod
    def format_success(data: Any, metadata: Optional[Dict] = None) -> str:
        """
        Format a successful tool response.
        
        Args:
            data: The actual response data
            metadata: Optional metadata about the response
            
        Returns:
            Formatted JSON string response
        """
        response = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        if metadata:
            response["metadata"] = metadata
            
        return json.dumps(response, indent=2)
    
    @staticmethod
    def format_error(
        error: Exception, 
        context: Optional[str] = None,
        suggestions: Optional[list] = None
    ) -> str:
        """
        Format an error response with helpful context.
        
        Args:
            error: The exception that occurred
            context: Additional context about what was being attempted
            suggestions: List of suggestions for fixing the issue
            
        Returns:
            Formatted error response
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        response = {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "type": error_type,
                "message": error_message
            }
        }
        
        if context:
            response["error"]["context"] = context
            
        if suggestions:
            response["error"]["suggestions"] = suggestions
            
        # Log the error for debugging
        logger.error(f"Tool error: {error_type}: {error_message}", exc_info=True)
        
        return json.dumps(response, indent=2)
    
    @staticmethod
    def format_partial(
        data: Any, 
        progress: Optional[float] = None,
        message: Optional[str] = None
    ) -> str:
        """
        Format a partial/streaming response.
        
        Args:
            data: Partial data available so far
            progress: Progress percentage (0-100)
            message: Status message
            
        Returns:
            Formatted partial response
        """
        response = {
            "status": "partial",
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        if progress is not None:
            response["progress"] = min(100, max(0, progress))
            
        if message:
            response["message"] = message
            
        return json.dumps(response, indent=2)
    
    @staticmethod
    def validate_response(response: Union[str, dict]) -> bool:
        """
        Validate that a response is properly formatted.
        
        Args:
            response: Response to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
                
            # Check required fields
            if "status" not in data:
                return False
                
            if data["status"] not in ["success", "error", "partial"]:
                return False
                
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
    
    @staticmethod
    def extract_content(response: Union[str, dict]) -> Optional[Any]:
        """
        Extract the actual content from a tool response.
        
        Args:
            response: Tool response
            
        Returns:
            The extracted content or None if invalid
        """
        try:
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
                
            if data.get("status") == "success":
                return data.get("data")
            elif data.get("status") == "error":
                return None
            elif data.get("status") == "partial":
                return data.get("data")
                
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
            
        return None
    
    @staticmethod
    def merge_responses(responses: list) -> str:
        """
        Merge multiple tool responses into a single response.
        
        Args:
            responses: List of tool responses
            
        Returns:
            Merged response
        """
        merged_data = []
        errors = []
        
        for response in responses:
            try:
                if isinstance(response, str):
                    data = json.loads(response)
                else:
                    data = response
                    
                if data.get("status") == "success":
                    merged_data.append(data.get("data"))
                elif data.get("status") == "error":
                    errors.append(data.get("error"))
                    
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                errors.append({"type": "ParseError", "message": str(e)})
        
        if errors:
            return ToolResponseHandler.format_error(
                Exception("Multiple errors occurred"),
                context="Merging tool responses",
                suggestions=["Check individual tool responses for details"]
            )
        
        return ToolResponseHandler.format_success(merged_data)
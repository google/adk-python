"""
Response formatting utilities for Cloud Functions
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional


def create_response(
    data: Dict[str, Any],
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None
) -> tuple:
    """
    Create a standardized HTTP response

    Args:
        data: Response data dictionary
        status_code: HTTP status code
        headers: Optional response headers

    Returns:
        Tuple of (response_body, status_code, headers)
    """
    # Add timestamp if not present
    if 'timestamp' not in data:
        data['timestamp'] = datetime.utcnow().isoformat()

    # Default headers
    default_headers = {
        'Content-Type': 'application/json',
        'X-Function-Version': '2.0.0'
    }

    if headers:
        default_headers.update(headers)

    return json.dumps(data), status_code, default_headers


def create_error_response(
    error: Exception,
    status_code: int = 500,
    context: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Create a standardized error response

    Args:
        error: Exception object
        status_code: HTTP status code
        context: Additional context information

    Returns:
        Tuple of (error_response, status_code, headers)
    """
    error_data = {
        'error': str(error),
        'error_type': type(error).__name__,
        'status': 'error',
        'timestamp': datetime.utcnow().isoformat()
    }

    if context:
        error_data['context'] = context

    return create_response(error_data, status_code)


def create_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Create a standardized success response

    Args:
        message: Success message
        data: Optional data payload
        metadata: Optional metadata

    Returns:
        Tuple of (success_response, status_code, headers)
    """
    response_data = {
        'status': 'success',
        'message': message
    }

    if data:
        response_data['data'] = data

    if metadata:
        response_data['metadata'] = metadata

    return create_response(response_data, 200)
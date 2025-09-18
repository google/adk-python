"""
Service for interacting with the ADK Security Agent backend.
"""

import os
import requests
import logging
import time
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get backend API URL from environment variable, with a default for local dev
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{BACKEND_API_URL}/api/v1/chat/message"
HEALTH_ENDPOINT = f"{BACKEND_API_URL}/health"
DATABASE_HEALTH_ENDPOINT = f"{BACKEND_API_URL}/health/database"

def check_backend_health() -> Dict[str, Any]:
    """
    Check if the backend service is running and healthy.

    Returns:
        Dict with health status information
    """
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return {
                "success": True,
                "status": "healthy",
                "details": health_data
            }
        else:
            return {
                "success": False,
                "status": "unhealthy",
                "error": f"Backend returned status {response.status_code}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status": "unreachable",
            "error": "Cannot connect to backend service. Please ensure it's running."
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": f"Health check failed: {e}"
        }


def check_database_health() -> Dict[str, Any]:
    """
    Check database connectivity and status.

    Returns:
        Dict with database health information
    """
    try:
        response = requests.get(DATABASE_HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            db_data = response.json()
            return {
                "success": True,
                "status": "healthy",
                "details": db_data
            }
        else:
            db_data = response.json() if response.content else {}
            return {
                "success": False,
                "status": "unhealthy",
                "error": db_data.get("error", f"Database check returned status {response.status_code}"),
                "details": db_data
            }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": f"Database health check failed: {e}"
        }


def send_message_with_retry(message: str, session_id: str = "default", user_id: str = "user", max_retries: int = 3) -> Dict[str, Any]:
    """
    Sends a message to the ADK agent backend with retry logic.

    Args:
        message: The user's message
        session_id: The session ID for the conversation
        user_id: The user ID
        max_retries: Maximum number of retry attempts

    Returns:
        Dict with the agent's response or error information
    """
    for attempt in range(max_retries):
        result = send_message(message, session_id, user_id)

        if result["success"]:
            return result

        # If it's the last attempt, return the error
        if attempt == max_retries - 1:
            result["retry_attempts"] = attempt + 1
            return result

        # Wait before retrying (exponential backoff)
        wait_time = 2 ** attempt  # 1s, 2s, 4s
        logger.info(f"Request failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
        time.sleep(wait_time)

    return result


def send_message(message: str, session_id: str = "default", user_id: str = "user") -> Dict[str, Any]:
    """
    Sends a message to the ADK agent backend and returns the response.

    Args:
        message: The user's message.
        session_id: The session ID for the conversation.
        user_id: The user ID.

    Returns:
        A dictionary with the agent's response or an error.
    """
    start_time = time.time()

    # First check if backend is reachable
    health_check = check_backend_health()
    if not health_check["success"]:
        logger.error(f"Backend health check failed: {health_check}")
        return {
            "success": False,
            "error": f"Backend service unavailable: {health_check['error']}",
            "suggestion": "Please ensure the backend service is running and accessible.",
            "health_status": health_check
        }

    payload = {
        "message": message,
        "session_id": session_id,
        "user_id": user_id
    }

    try:
        logger.info(f"📤 Sending message to backend: {message[:50]}{'...' if len(message) > 50 else ''}")
        logger.debug(f"Full payload: {payload}")

        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            timeout=120,
            headers={"Content-Type": "application/json"}
        )

        request_duration = time.time() - start_time
        logger.info(f"⏱️ Backend response received in {request_duration:.2f}s")

        # Handle different HTTP status codes
        if response.status_code == 200:
            response_data = response.json()
            logger.info(f"✅ Successful response from backend")

            return {
                "success": True,
                "response": response_data.get("response", "No response text found."),
                "metadata": response_data.get("metadata", {}),
                "request_duration": request_duration
            }

        elif response.status_code == 500:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", "Internal server error")
            logger.error(f"❌ Backend server error: {error_msg}")

            return {
                "success": False,
                "error": f"Backend processing error: {error_msg}",
                "suggestion": "The backend encountered an internal error. Please try again or check the logs.",
                "status_code": 500,
                "request_duration": request_duration
            }

        elif response.status_code == 503:
            logger.error(f"❌ Backend service unavailable")
            return {
                "success": False,
                "error": "Backend service is temporarily unavailable",
                "suggestion": "The service may be starting up or experiencing issues. Please wait and try again.",
                "status_code": 503,
                "request_duration": request_duration
            }

        else:
            logger.error(f"❌ Unexpected status code: {response.status_code}")
            response.raise_for_status()  # This will raise an appropriate exception

    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Connection error: {e}")
        return {
            "success": False,
            "error": "Cannot connect to the backend service",
            "suggestion": "Please ensure the backend service is running at " + BACKEND_API_URL,
            "technical_details": str(e),
            "request_duration": time.time() - start_time
        }

    except requests.exceptions.Timeout as e:
        logger.error(f"❌ Request timeout: {e}")
        return {
            "success": False,
            "error": "Request to backend timed out",
            "suggestion": "The backend is taking too long to respond. Please try a simpler query or try again later.",
            "technical_details": str(e),
            "request_duration": time.time() - start_time
        }

    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP error: {e}")
        return {
            "success": False,
            "error": f"HTTP error occurred: {e}",
            "suggestion": "There was an issue with the request. Please check your input and try again.",
            "technical_details": str(e),
            "request_duration": time.time() - start_time
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request exception: {e}")
        return {
            "success": False,
            "error": f"Failed to communicate with backend: {e}",
            "suggestion": "There was a network or communication error. Please try again.",
            "technical_details": str(e),
            "request_duration": time.time() - start_time
        }

    except ValueError as e:
        logger.error(f"❌ JSON decode error: {e}")
        return {
            "success": False,
            "error": "Backend returned invalid response format",
            "suggestion": "The backend response could not be parsed. This may indicate a backend issue.",
            "technical_details": str(e),
            "request_duration": time.time() - start_time
        }

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return {
            "success": False,
            "error": f"An unexpected error occurred: {e}",
            "suggestion": "Please try again. If the problem persists, contact support.",
            "technical_details": str(e),
            "request_duration": time.time() - start_time
        }
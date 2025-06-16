# tests/test_a2a_stock_client.py
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
import argparse
import uuid
import logging
import json
from typing import Optional

from common_impl.client import A2AClient
from common_impl.types import (
    TaskSendParams, Message, TextPart, SendTaskResponse, TaskState, Artifact,
    DataPart, A2AClientHTTPError, A2AClientJSONError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - A2A_TEST_CLIENT - %(message)s')
logger = logging.getLogger(__name__)

async def run_a2a_test(symbol: str, server_url: str, session_id: Optional[str] = None):
    """
    Sends a task to the StockInfoAgent A2A server and prints the result.
    """
    if not session_id:
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        logger.info(f"No session ID provided, generated one: {session_id}")

    task_id = uuid.uuid4().hex
    logger.info(f"Creating A2A task {task_id} for session {session_id} with symbol '{symbol}'")

    # Create the A2A client targeting the StockInfoAgent server
    a2a_client = A2AClient(url=server_url) # Pass the URL directly

    # Construct the A2A TaskSendParams
    task_params = TaskSendParams(
        id=task_id,
        sessionId=session_id,
        message=Message(
            role="user",
            parts=[TextPart(text=symbol)] # Send the symbol as text
        ),
        # acceptedOutputModes=["application/json"] # Optional: Specify expected output
    )

    logger.info(f"Sending task to A2A server at {server_url}...")

    try:
        # Send the task using the A2A client
        # Use .model_dump() for Pydantic v2+ or .dict() for v1
        response: SendTaskResponse = await a2a_client.send_task(task_params.model_dump())

        logger.info("Received response from A2A server.")

        # Process the response
        if response.error:
            logger.error(f"A2A Server returned an error: {response.error.model_dump()}")
        elif response.result:
            task_result = response.result
            logger.info(f"Task {task_result.id} final state: {task_result.status.state}")

            if task_result.status.state == TaskState.COMPLETED:
                logger.info("Task completed successfully.")
                if task_result.artifacts:
                    logger.info("Artifacts received:")
                    for artifact in task_result.artifacts:
                        logger.info(f"- Artifact Name: {artifact.name or 'N/A'}")
                        for part in artifact.parts:
                            if isinstance(part, DataPart):
                                logger.info(f"  DataPart Content: {json.dumps(part.data, indent=2)}")
                            elif isinstance(part, TextPart):
                                 logger.info(f"  TextPart Content: {part.text}")
                            else:
                                 logger.info(f"  Other Part Type: {part.type}")
                else:
                    logger.warning("Task completed but no artifacts found.")

            elif task_result.status.state == TaskState.FAILED:
                logger.error("Task failed.")
                if task_result.artifacts:
                     logger.error("Error details from artifact:")
                     for artifact in task_result.artifacts:
                         if artifact.name == "error_details":
                             for part in artifact.parts:
                                 if isinstance(part, DataPart):
                                     logger.error(f"  {json.dumps(part.data, indent=2)}")
                                 else:
                                      logger.error(f"  Unexpected error artifact part: {part}")
                elif task_result.status.message:
                     logger.error(f"Error message from status: {task_result.status.message}")
                else:
                     logger.error("Task failed with no specific error details provided.")
            else:
                logger.warning(f"Task ended in unexpected state: {task_result.status.state}")

        else:
            logger.error("Received an empty response from the server.")

    except A2AClientHTTPError as http_err:
        logger.error(f"HTTP Error connecting to A2A server: {http_err.status_code} - {http_err.message}")
    except A2AClientJSONError as json_err:
         logger.error(f"JSON Error processing A2A server response: {json_err.message}")
    except ConnectionRefusedError:
         logger.error(f"Connection refused. Is the A2A server running at {server_url}?")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test client for the StockInfoAgent A2A server.")
    parser.add_argument("symbol", help="The stock ticker symbol to query (e.g., MSFT).")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="The full URL of the A2A server endpoint.")
    parser.add_argument("--session", default=None, help="Optional session ID to use.")

    args = parser.parse_args()

    try:
        asyncio.run(run_a2a_test(args.symbol, args.url, args.session))
    except KeyboardInterrupt:
        logger.info("Test client stopped by user.")
import os
import logging
import json
from typing import Union, AsyncIterable # <--- Import these

from google.adk.agents import Agent # Import ADK Agent
from google.adk.runners import Runner # Import ADK Runner
from google.adk.sessions import InMemorySessionService, Session # Need Session components
from google.genai import types as genai_types # For Content/Part

from common_impl.server.task_manager import InMemoryTaskManager # Use the base class
from common_impl.types import (
    SendTaskRequest, SendTaskResponse, Task, TaskState, TaskStatus,
    Artifact, TextPart, DataPart, JSONRPCError, InternalError,
    SendTaskStreamingRequest, SendTaskStreamingResponse, # <--- Import these
    UnsupportedOperationError, JSONRPCResponse # <--- Import these
)


from .agent import create_agent_with_mcp_tools
from mcp.types import TextContent, CallToolResult



logger = logging.getLogger(__name__)

class StockInfoTaskManager(InMemoryTaskManager):
    """Handles A2A tasks by running the ADK StockInfoAgent."""

    def __init__(self, mcp_server_script_path: str):
        super().__init__()
        # Store the path needed by create_agent_with_mcp_tools
        self.mcp_server_script_path = mcp_server_script_path
        logger.info(f"StockInfoTaskManager initialized (ADK Runner Mode). Will use MCP server at: {self.mcp_server_script_path}")
        # NOTE: Agent and Runner are now created PER request in this model

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles A2A task by instantiating and running the ADK agent."""
        task_params = request.params
        task_id = task_params.id
        session_id = task_params.sessionId
        input_message = task_params.message

        logger.info(f"A2A Task Mgr (ADK Mode): Received task '{task_id}' in session '{session_id}'")

        # 1. Create/Update Task State (same as before)
        task = await self.upsert_task(task_params)
        task.status.state = TaskState.WORKING
        await self.update_store(task_id, task.status, None)
        logger.debug(f"A2A Task Mgr (ADK Mode): Task '{task_id}' state set to WORKING.")

        # 2. Extract Symbol (same as before)
        symbol = None
        if input_message.parts:
            for part in input_message.parts:
                if isinstance(part, TextPart) and part.text:
                    symbol = part.text.strip()
                    logger.info(f"A2A Task Mgr (ADK Mode): Extracted symbol '{symbol}' for task '{task_id}'.")
                    break
        if not symbol:
            # Handle error (same as before)
            logger.error(f"A2A Task Mgr (ADK Mode): No symbol found for task '{task_id}'.")
            task.status.state = TaskState.FAILED
            error_artifact = Artifact(name="error_details", parts=[DataPart(data={"error": "Ticker symbol not provided."})])
            task.artifacts = [error_artifact]
            await self.update_store(task_id, task.status, task.artifacts)
            return SendTaskResponse(id=request.id, result=task)

        # --- 3. Run the ADK Agent ---
        adk_agent: Agent | None = None
        exit_stack = None
        adk_result_dict: Dict[str, Any] = {"error": "ADK agent execution failed to produce a result."} # Default error

        try:
            # --- Instantiate ADK components for this request ---
            logger.info(f"A2A Task Mgr (ADK Mode): Creating ADK Agent for task '{task_id}'...")
            # This starts the MCP server subprocess and gets the exit_stack
            adk_agent, exit_stack = await create_agent_with_mcp_tools(self.mcp_server_script_path)

            # Use a transient session service and session for this specific ADK run
            temp_session_service = InMemorySessionService()
            temp_adk_session = temp_session_service.create_session(
                app_name=f"adk_task_{task_id}", # Unique app name per task
                user_id=f"a2a_user_{session_id}",
                session_id=f"adk_run_{task_id}",
                state={} # Start with empty state for the ADK agent run
            )

            adk_runner = Runner(
                agent=adk_agent,
                app_name=temp_adk_session.app_name,
                session_service=temp_session_service,
            )
            logger.info(f"A2A Task Mgr (ADK Mode): ADK Runner initialized for agent '{adk_agent.name}'.")

            # Prepare the input specifically for the ADK agent's tool
            # The ADK agent expects a natural language query that will trigger the tool
            adk_query = f"What is the stock price for {symbol}?"
            adk_content = genai_types.Content(role='user', parts=[genai_types.Part(text=adk_query)])

            logger.info(f"A2A Task Mgr (ADK Mode): Running ADK agent for task '{task_id}'...")
            async for event in adk_runner.run_async(
                session_id=temp_adk_session.id,
                user_id=temp_adk_session.user_id,
                new_message=adk_content
            ):
                 logger.debug(f"A2A Task Mgr (ADK Mode): ADK Event: {event.author} - Final: {event.is_final_response()}")
                 # We need to extract the *tool result* that the agent processed
                 # The final response *from the ADK agent* isn't directly what the A2A server should return.
                 if event.get_function_responses():
                     for func_resp in event.get_function_responses():
                          # Check the tool name is the one we expect
                          if func_resp.name == "get_current_stock_price":
                               try:
                                   # 1. Get the outer dictionary
                                   response_wrapper = func_resp.response
                                   if not isinstance(response_wrapper, dict):
                                        logger.error(f"A2A Task Mgr (ADK Mode): Outer response wasn't a dict: {type(response_wrapper)}")
                                        adk_result_dict = {"error": "Invalid outer tool response format from ADK Agent run."}
                                        break

                                   # 2. Access the nested 'result' which should hold the CallToolResult object (or similar structure)
                                   # Use .get() for safety
                                   mcp_tool_result_obj = response_wrapper.get("result")
                                   if mcp_tool_result_obj is None:
                                        logger.error(f"A2A Task Mgr (ADK Mode): Outer response dict missing 'result' key: {response_wrapper}")
                                        adk_result_dict = {"error": "Missing 'result' in outer tool response format."}
                                        break

                                   # 3. Check if the MCP tool reported an error via the isError attribute
                                   # Access attributes safely using getattr
                                   is_error = getattr(mcp_tool_result_obj, 'isError', False)
                                   mcp_content = getattr(mcp_tool_result_obj, 'content', None)

                                   if is_error:
                                       error_text = "Unknown MCP tool error"
                                       if mcp_content and len(mcp_content) > 0:
                                           # Check if the first content part has 'text'
                                           first_content_part = mcp_content[0]
                                           error_text = getattr(first_content_part, 'text', str(first_content_part))
                                           try: # Try parsing JSON error from tool's text content
                                               parsed_error = json.loads(error_text)
                                               if isinstance(parsed_error, dict) and "error" in parsed_error:
                                                   adk_result_dict = parsed_error # Use the tool's error dict
                                               else:
                                                   adk_result_dict = {"error": f"MCP Tool Error: {error_text}"}
                                           except (json.JSONDecodeError, TypeError): # Catch TypeError just in case
                                                adk_result_dict = {"error": f"MCP Tool Error: {error_text}"} # Treat raw text as error
                                       else:
                                            adk_result_dict = {"error": "MCP Tool reported error with no content."}
                                       logger.error(f"A2A Task Mgr (ADK Mode): MCP Tool reported error: {adk_result_dict}")

                                   # 4. If not an error, extract and parse the JSON from TextContent
                                   elif mcp_content and len(mcp_content) > 0:
                                       first_content_part = mcp_content[0]
                                       # Check it has 'type' == 'text' and a 'text' attribute
                                       if getattr(first_content_part, 'type', None) == 'text' and hasattr(first_content_part, 'text'):
                                            result_text = getattr(first_content_part, 'text', '')
                                            try:
                                                parsed_dict = json.loads(result_text)
                                                if isinstance(parsed_dict, dict) and "price" in parsed_dict and "symbol" in parsed_dict:
                                                    adk_result_dict = parsed_dict # Successfully parsed the result
                                                    logger.info(f"A2A Task Mgr (ADK Mode): Successfully captured tool result: {adk_result_dict}")
                                                else:
                                                    logger.error(f"A2A Task Mgr (ADK Mode): Parsed JSON from tool lacks expected keys: {parsed_dict}")
                                                    adk_result_dict = {"error": "Invalid data structure from MCP tool (missing keys)."}
                                            except json.JSONDecodeError:
                                                logger.error(f"A2A Task Mgr (ADK Mode): Failed to parse JSON from successful MCP tool response text: {result_text}")
                                                adk_result_dict = {"error": "Non-JSON success response text from MCP tool."}
                                       else:
                                           logger.error(f"A2A Task Mgr (ADK Mode): Expected TextContent, but got different type or missing text: {first_content_part}")
                                           adk_result_dict = {"error": "Unexpected content type in successful MCP response."}
                                   else:
                                       # Handle unexpected successful response format (no content)
                                       logger.error(f"A2A Task Mgr (ADK Mode): Successful MCP response has no content: {mcp_tool_result_obj}")
                                       adk_result_dict = {"error": "Missing content in successful MCP response."}

                                   break # Processed the relevant function response

                               except Exception as process_err:
                                    logger.error(f"A2A Task Mgr (ADK Mode): Error processing ADK tool response event: {process_err}", exc_info=True)
                                    adk_result_dict = {"error": "Internal error processing specialist response."}
                                    break # Stop processing on error
                               
            logger.info(f"A2A Task Mgr (ADK Mode): ADK agent run finished for task '{task_id}'.")

        except Exception as adk_err:
            logger.exception(f"A2A Task Mgr (ADK Mode): Error during ADK agent execution for task '{task_id}': {adk_err}")
            adk_result_dict = {"error": f"Failed to execute ADK agent: {str(adk_err)}"}
        finally:
            # --- CRUCIAL: Cleanup the exit_stack for the MCP process ---
            if exit_stack:
                logger.info(f"A2A Task Mgr (ADK Mode): Cleaning up MCP connection for task '{task_id}'...")
                await exit_stack.aclose()
                logger.info(f"A2A Task Mgr (ADK Mode): MCP connection closed for task '{task_id}'.")

        # --- 4. Process Result and Finalize A2A Task ---
        if "error" not in adk_result_dict and "price" in adk_result_dict:
            task.status.state = TaskState.COMPLETED
            task.status.message = None
            result_artifact = Artifact(name="stock_price_data", parts=[DataPart(data=adk_result_dict)])
            task.artifacts = [result_artifact]
            logger.info(f"A2A Task Mgr (ADK Mode): Task '{task_id}' COMPLETED successfully.")
        else:
            task.status.state = TaskState.FAILED
            task.status.message = None
            error_msg = adk_result_dict.get("error", "Unknown error from ADK Agent execution.")
            logger.error(f"A2A Task Mgr (ADK Mode): Task '{task_id}' FAILED. Reason: {error_msg}")
            error_artifact = Artifact(name="error_details", parts=[DataPart(data={"error": error_msg})])
            task.artifacts = [error_artifact]

        await self.update_store(task_id, task.status, task.artifacts)

        # 5. Return A2A Response
        return SendTaskResponse(id=request.id, result=task)
    
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """Handles streaming requests - intentionally not supported by this agent."""
        task_id = request.params.id if request.params else "unknown"
        logger.warning(f"A2A Task Manager: Received 'tasks/sendSubscribe' request for task {task_id}, but streaming is not supported by this agent.")
        # Log the incoming request
        logger.info(f"A2A Task Manager: Received Full Request (sendSubscribe): {request.model_dump_json(indent=2)}")
        # Return an error compliant with JSON-RPC and A2A types
        response = JSONRPCResponse(
            id=request.id,
            error=UnsupportedOperationError(
                message="Streaming (tasks/sendSubscribe) is not supported by this agent."
            )
        )
        logger.warning(f"A2A Task Manager: Sending UnsupportedOperation Error Response: {response.model_dump_json(indent=2)}")
        return response

    # Override other handlers as needed, e.g., return "Unsupported" error
    # async def on_get_task(...) -> GetTaskResponse: ...
    # async def on_cancel_task(...) -> CancelTaskResponse: ...
    # async def on_send_task_subscribe(...) -> ... : # Return UnsupportedOperationError
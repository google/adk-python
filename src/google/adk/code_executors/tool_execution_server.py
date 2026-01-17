# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool execution server for CodingAgent.

This module provides a FastAPI server that handles tool execution requests
from code running in containers. It routes requests to the appropriate
ADK tools with full ToolContext support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..agents.invocation_context import InvocationContext
    from ..tools.base_tool import BaseTool
    from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


class ToolCallRequest(BaseModel):
    """Request model for tool calls."""

    tool_name: str
    args: Dict[str, Any]


class ToolCallResponse(BaseModel):
    """Response model for tool calls."""

    result: Any
    success: bool
    error: Optional[str] = None


@dataclass
class ToolTrace:
    """Record of a tool call for debugging and telemetry."""

    tool_name: str
    args: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    success: bool = False
    duration_ms: float = 0.0


def detect_docker_host_address() -> str:
    """Detect the appropriate host address for Docker containers.

    On macOS and Windows (Docker Desktop), use host.docker.internal.
    On Linux, use 172.17.0.1 (default Docker bridge network gateway).

    Note: host.docker.internal only resolves from within containers,
    not from the host machine, so we check the platform instead.

    Returns:
      The detected host address.
    """
    import platform

    system = platform.system().lower()

    # macOS and Windows use Docker Desktop which supports host.docker.internal
    if system in ("darwin", "windows"):
        return "host.docker.internal"

    # Linux: use Docker bridge network gateway
    return "172.17.0.1"


class ToolExecutionServer:
    """FastAPI server for executing ADK tools via HTTP.

    This server is designed to run on the host machine and receive tool
    execution requests from code running in Docker containers.

    Attributes:
      host: Host address to bind the server to.
      port: Port to bind the server to.
      tools: Dictionary mapping tool names to tool instances.
      invocation_context: The current invocation context.
      tool_context: The current tool context.
      traces: List of tool call traces.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        tools: Optional[List[BaseTool]] = None,
        invocation_context: Optional[InvocationContext] = None,
    ):
        """Initialize the tool execution server.

        Args:
          host: Host address to bind to.
          port: Port to bind to.
          tools: List of tools to make available.
          invocation_context: The invocation context for tool calls.
        """
        self.host = host
        self.port = port
        self.tools: Dict[str, BaseTool] = {}
        self.invocation_context = invocation_context
        self.tool_context: Optional[ToolContext] = None
        self.traces: List[ToolTrace] = []
        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None
        self._app = self._create_app()

        if tools:
            for tool in tools:
                self.register_tool(tool)

    def _create_app(self) -> FastAPI:
        """Create the FastAPI application with routes."""
        app = FastAPI(
            title="ADK Tool Execution Server",
            description="HTTP server for executing ADK tools from containers",
            version="1.0.0",
        )

        @app.post("/tool_call", response_model=ToolCallResponse)
        async def handle_tool_call(request: ToolCallRequest) -> ToolCallResponse:
            """Handle a tool call request."""
            return await self._execute_tool(request.tool_name, request.args)

        @app.get("/tool_trace")
        async def get_tool_traces() -> List[Dict[str, Any]]:
            """Get all tool call traces."""
            return [
                {
                    "tool_name": t.tool_name,
                    "args": t.args,
                    "result": t.result,
                    "error": t.error,
                    "success": t.success,
                    "duration_ms": t.duration_ms,
                }
                for t in self.traces
            ]

        @app.delete("/tool_trace")
        async def clear_tool_traces() -> Dict[str, str]:
            """Clear all tool call traces."""
            self.traces.clear()
            return {"status": "cleared"}

        @app.get("/health")
        async def health_check() -> Dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy"}

        @app.get("/tools")
        async def list_tools() -> List[str]:
            """List available tools."""
            return list(self.tools.keys())

        return app

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool with the server.

        Args:
          tool: The tool to register.
        """
        self.tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def set_context(
        self,
        invocation_context: InvocationContext,
        tool_context: Optional[ToolContext] = None,
    ) -> None:
        """Set the context for tool execution.

        Args:
          invocation_context: The invocation context.
          tool_context: The tool context.
        """
        self.invocation_context = invocation_context
        self.tool_context = tool_context

    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> ToolCallResponse:
        """Execute a tool and return the result.

        Args:
          tool_name: Name of the tool to execute.
          args: Arguments to pass to the tool.

        Returns:
          The tool execution response.
        """
        import time

        start_time = time.time()
        trace = ToolTrace(tool_name=tool_name, args=args)

        if tool_name not in self.tools:
            trace.error = f"Tool not found: {tool_name}"
            trace.success = False
            trace.duration_ms = (time.time() - start_time) * 1000
            self.traces.append(trace)
            raise HTTPException(status_code=404, detail=trace.error)

        tool = self.tools[tool_name]

        try:
            # Create a tool context if we have an invocation context
            if self.invocation_context and not self.tool_context:
                from ..tools.tool_context import ToolContext

                self.tool_context = ToolContext(
                    invocation_context=self.invocation_context,
                )

            if self.tool_context:
                result = await tool.run_async(args=args, tool_context=self.tool_context)
            else:
                # If no context available, create a minimal mock context
                # This is a fallback and shouldn't happen in normal operation
                logger.warning("Executing tool %s without proper context", tool_name)
                result = await tool.run_async(args=args, tool_context=None)

            trace.result = result
            trace.success = True
            trace.duration_ms = (time.time() - start_time) * 1000
            self.traces.append(trace)

            return ToolCallResponse(result=result, success=True)

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            trace.duration_ms = (time.time() - start_time) * 1000
            self.traces.append(trace)
            logger.error("Tool execution failed: %s - %s", tool_name, e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    def start(self) -> None:
        """Start the server in a background thread."""
        if self._server_thread and self._server_thread.is_alive():
            logger.warning("Server already running")
            return

        config = uvicorn.Config(
            app=self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        def run_server():
            asyncio.run(self._server.serve())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Wait for server to be ready
        self._wait_for_server()
        logger.info("Tool execution server started on %s:%d", self.host, self.port)

    def _wait_for_server(self, timeout: float = 10.0) -> None:
        """Wait for the server to be ready.

        Args:
          timeout: Maximum time to wait in seconds.
        """
        import time

        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", self.port))
                sock.close()
                if result == 0:
                    return
            except Exception:
                pass
            time.sleep(0.1)

        logger.warning("Server may not be fully ready after %.1f seconds", timeout)

    def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.should_exit = True
            if self._server_thread:
                self._server_thread.join(timeout=5.0)
            self._server = None
            self._server_thread = None
            logger.info("Tool execution server stopped")

    def get_url(self, for_container: bool = True) -> str:
        """Get the URL for the server.

        Args:
          for_container: If True, return URL accessible from Docker containers.

        Returns:
          The server URL.
        """
        if for_container:
            host = detect_docker_host_address()
        else:
            host = "localhost" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}"

    def clear_traces(self) -> None:
        """Clear all tool call traces."""
        self.traces.clear()

    def get_traces(self) -> List[ToolTrace]:
        """Get all tool call traces.

        Returns:
          List of tool traces.
        """
        return self.traces.copy()

    def __enter__(self) -> ToolExecutionServer:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

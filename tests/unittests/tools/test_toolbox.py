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

import inspect
from typing import Any, Callable, Mapping, Optional

from aioresponses import aioresponses
from google.adk.tools import toolbox
import pytest

TEST_BASE_URL = "http://toolbox.example.com"


@pytest.fixture
def sync_client_environment():
    """
    Ensures a clean environment for ToolboxSyncClient class-level resources.
    It resets the class-level event loop and thread before the test
    and stops them after the test. This is crucial for test isolation
    due to ToolboxSyncClient's use of class-level loop/thread.
    """
    # Save current state if any (more of a defensive measure)
    original_loop = getattr(toolbox.ToolboxSyncClient, "_ToolboxSyncClient__loop", None)
    original_thread = getattr(
        toolbox.ToolboxSyncClient, "_ToolboxSyncClient__thread", None
    )

    # Force reset class state before the test.
    # This ensures any client created will start a new loop/thread.

    # Ensure no loop/thread is running from a previous misbehaving test or setup
    assert original_loop is None or not original_loop.is_running()
    assert original_thread is None or not original_thread.is_alive()

    toolbox.ToolboxSyncClient._ToolboxSyncClient__loop = None
    toolbox.ToolboxSyncClient._ToolboxSyncClient__thread = None

    yield

    # Teardown: stop the loop and join the thread created *during* the test.
    test_loop = getattr(toolbox.ToolboxSyncClient, "_ToolboxSyncClient__loop", None)
    test_thread = getattr(toolbox.ToolboxSyncClient, "_ToolboxSyncClient__thread", None)

    if test_loop and test_loop.is_running():
        test_loop.call_soon_threadsafe(test_loop.stop)
    if test_thread and test_thread.is_alive():
        test_thread.join(timeout=5)

    # Explicitly set to None to ensure a clean state for the next fixture use/test.
    toolbox.ToolboxSyncClient._ToolboxSyncClient__loop = None
    toolbox.ToolboxSyncClient._ToolboxSyncClient__thread = None


@pytest.fixture
def sync_client(sync_client_environment):
    """
    Provides a ToolboxSyncClient instance within an isolated environment.
    The client's underlying async session is automatically closed after the test.
    The class-level loop/thread are managed by sync_client_environment.
    """
    client = toolbox.ToolboxSyncClient(TEST_BASE_URL)

    yield client

    client.close()  # Closes the async_client's session.
    # Loop/thread shutdown is handled by sync_client_environment's teardown.


@pytest.fixture()
def test_tool_str_schema():
    return toolbox.protocol.ToolSchema(
        description="Test Tool with String input",
        parameters=[
            toolbox.protocol.ParameterSchema(
                name="param1", type="string", description="Description of Param1"
            )
        ],
    )


@pytest.fixture()
def test_tool_int_bool_schema():
    return toolbox.protocol.ToolSchema(
        description="Test Tool with Int, Bool",
        parameters=[
            toolbox.protocol.ParameterSchema(
                name="argA", type="integer", description="Argument A"
            ),
            toolbox.protocol.ParameterSchema(
                name="argB", type="boolean", description="Argument B"
            ),
        ],
    )


@pytest.fixture()
def test_tool_auth_schema():
    return toolbox.protocol.ToolSchema(
        description="Test Tool with Int,Bool+Auth",
        parameters=[
            toolbox.protocol.ParameterSchema(
                name="argA", type="integer", description="Argument A"
            ),
            toolbox.protocol.ParameterSchema(
                name="argB",
                type="boolean",
                description="Argument B",
                authSources=["my-auth-service"],
            ),
        ],
    )


@pytest.fixture
def tool_schema_minimal():
    return toolbox.protocol.ToolSchema(description="Minimal Test Tool", parameters=[])


# --- Helper Functions for Mocking ---
def mock_tool_load(
    aio_resp: aioresponses,
    tool_name: str,
    tool_schema: toolbox.protocol.ToolSchema,
    base_url: str = TEST_BASE_URL,
    server_version: str = "0.0.0",
    status: int = 200,
    callback: Optional[Callable] = None,
    payload_override: Optional[Any] = None,
):
    url = f"{base_url}/api/tool/{tool_name}"
    payload_data = {}
    if payload_override is not None:
        payload_data = payload_override
    else:
        manifest = toolbox.protocol.ManifestSchema(
            serverVersion=server_version, tools={tool_name: tool_schema}
        )
        payload_data = manifest.model_dump()
    aio_resp.get(url, payload=payload_data, status=status, callback=callback)


def mock_toolset_load(
    aio_resp: aioresponses,
    toolset_name: str,
    tools_dict: Mapping[str, toolbox.protocol.ToolSchema],
    base_url: str = TEST_BASE_URL,
    server_version: str = "0.0.0",
    status: int = 200,
    callback: Optional[Callable] = None,
):
    url_path = f"toolset/{toolset_name}" if toolset_name else "toolset/"
    url = f"{base_url}/api/{url_path}"
    manifest = toolbox.protocol.ManifestSchema(
        serverVersion=server_version, tools=tools_dict
    )
    aio_resp.get(url, payload=manifest.model_dump(), status=status, callback=callback)


def mock_tool_invoke(
    aio_resp: aioresponses,
    tool_name: str,
    base_url: str = TEST_BASE_URL,
    response_payload: Any = {"result": "ok"},
    status: int = 200,
    callback: Optional[Callable] = None,
):
    url = f"{base_url}/api/tool/{tool_name}/invoke"
    aio_resp.post(url, payload=response_payload, status=status, callback=callback)


# --- Tests for General ToolboxSyncClient Functionality ---


def test_sync_load_tool_success(aioresponses, test_tool_str_schema, sync_client):
    TOOL_NAME = "test_tool_sync_1"
    mock_tool_load(aioresponses, TOOL_NAME, test_tool_str_schema)
    mock_tool_invoke(
        aioresponses, TOOL_NAME, response_payload={"result": "sync_tool_ok"}
    )

    loaded_tool = sync_client.load_tool(TOOL_NAME)

    assert callable(loaded_tool)
    assert isinstance(loaded_tool, toolbox.sync_tool.ToolboxSyncTool)
    assert loaded_tool.__name__ == TOOL_NAME
    assert test_tool_str_schema.description in loaded_tool.__doc__
    sig = inspect.signature(loaded_tool)
    assert list(sig.parameters.keys()) == [
        p.name for p in test_tool_str_schema.parameters
    ]
    result = loaded_tool(param1="some value")
    assert result == "sync_tool_ok"


def test_sync_load_toolset_success(
    aioresponses, test_tool_str_schema, test_tool_int_bool_schema, sync_client
):
    TOOLSET_NAME = "my_sync_toolset"
    TOOL1_NAME = "sync_tool1"
    TOOL2_NAME = "sync_tool2"
    tools_definition = {
        TOOL1_NAME: test_tool_str_schema,
        TOOL2_NAME: test_tool_int_bool_schema,
    }
    mock_toolset_load(aioresponses, TOOLSET_NAME, tools_definition)
    mock_tool_invoke(
        aioresponses, TOOL1_NAME, response_payload={"result": f"{TOOL1_NAME}_ok"}
    )
    mock_tool_invoke(
        aioresponses, TOOL2_NAME, response_payload={"result": f"{TOOL2_NAME}_ok"}
    )

    tools = sync_client.load_toolset(TOOLSET_NAME)

    assert isinstance(tools, list)
    assert len(tools) == len(tools_definition)
    assert all(isinstance(t, toolbox.sync_tool.ToolboxSyncTool) for t in tools)
    assert {t.__name__ for t in tools} == tools_definition.keys()
    tool1 = next(t for t in tools if t.__name__ == TOOL1_NAME)
    result1 = tool1(param1="hello")
    assert result1 == f"{TOOL1_NAME}_ok"


def test_sync_invoke_tool_server_error(aioresponses, test_tool_str_schema, sync_client):
    TOOL_NAME = "sync_server_error_tool"
    ERROR_MESSAGE = "Simulated Server Error for Sync Client"
    mock_tool_load(aioresponses, TOOL_NAME, test_tool_str_schema)
    mock_tool_invoke(
        aioresponses, TOOL_NAME, response_payload={"error": ERROR_MESSAGE}, status=500
    )

    loaded_tool = sync_client.load_tool(TOOL_NAME)
    with pytest.raises(Exception, match=ERROR_MESSAGE):
        loaded_tool(param1="some input")


def test_sync_load_tool_not_found_in_manifest(
    aioresponses, test_tool_str_schema, sync_client
):
    ACTUAL_TOOL_IN_MANIFEST = "actual_tool_sync_abc"
    REQUESTED_TOOL_NAME = "non_existent_tool_sync_xyz"
    mismatched_manifest_payload = toolbox.protocol.ManifestSchema(
        serverVersion="0.0.0", tools={ACTUAL_TOOL_IN_MANIFEST: test_tool_str_schema}
    ).model_dump()
    mock_tool_load(
        aio_resp=aioresponses,
        tool_name=REQUESTED_TOOL_NAME,
        tool_schema=test_tool_str_schema,
        payload_override=mismatched_manifest_payload,
    )

    with pytest.raises(
        Exception,
        match=f"Tool '{REQUESTED_TOOL_NAME}' not found!",
    ):
        sync_client.load_tool(REQUESTED_TOOL_NAME)
    aioresponses.assert_called_once_with(
        f"{TEST_BASE_URL}/api/tool/{REQUESTED_TOOL_NAME}",
        method="GET",
    )

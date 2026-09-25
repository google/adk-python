# Copyright 2026 Google LLC
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

"""Tests for VertexAiCodeExecutor."""

# pylint: disable=protected-access

import concurrent.futures
import copy
import threading
import time
import unittest
from unittest import mock

from google.adk.code_executors import vertex_ai_code_executor
from google.adk.code_executors.vertex_ai_code_executor import CodeExecutionInput
from google.adk.code_executors.vertex_ai_code_executor import File
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor

InvocationContext = mock.MagicMock


class TestVertexAiCodeExecutor(unittest.TestCase):
  """Unit tests for VertexAiCodeExecutor."""

  def setUp(self):
    """Set up common fixtures for the tests."""
    super().setUp()
    vertex_ai_code_executor._EXTENSION_CLIENTS.clear()
    self.mock_resource_name = (
        'projects/123/locations/us-central1/extensions/456'
    )
    self.executor = VertexAiCodeExecutor(resource_name=self.mock_resource_name)

  def tearDown(self):
    super().tearDown()
    vertex_ai_code_executor._EXTENSION_CLIENTS.clear()

  def _create_mock_files(
      self, file_data: list[tuple[str, str, str]]
  ) -> list[File]:
    """Helper to create File objects from (name, content, mime_type)."""
    return [
        File(name=name, content=content, mime_type=mime_type)
        for name, content, mime_type in file_data
    ]

  # --- Test Initialization & Deepcopy Safety ---

  @mock.patch('vertexai.preview.extensions.Extension')
  def test_init_is_lazy(self, mock_extension_class):
    """Verifies __init__ does NOT create the external client."""
    _ = VertexAiCodeExecutor(resource_name=self.mock_resource_name)
    _ = VertexAiCodeExecutor()
    mock_extension_class.assert_not_called()
    self.assertNotIn(
        self.mock_resource_name, vertex_ai_code_executor._EXTENSION_CLIENTS
    )

  @mock.patch('vertexai.preview.extensions.Extension')
  def test_deepcopy_safety(self, mock_extension_class):
    """Verifies that deepcopy works without RecursionError before and after client access."""
    # Pre-access deepcopy
    executor_copy_pre = copy.deepcopy(self.executor)
    self.assertNotEqual(id(self.executor), id(executor_copy_pre))
    self.assertEqual(
        executor_copy_pre.resource_name, self.executor.resource_name
    )
    self.assertFalse(self.executor.__pydantic_private__)
    self.assertFalse(executor_copy_pre.__pydantic_private__)

    # Access client
    _ = self.executor._extension_client
    self.assertFalse(self.executor.__pydantic_private__)

    # Post-access deepcopy
    executor_copy_post = copy.deepcopy(self.executor)
    self.assertNotEqual(id(self.executor), id(executor_copy_post))
    self.assertEqual(
        executor_copy_post.resource_name, self.executor.resource_name
    )
    self.assertFalse(executor_copy_post.__pydantic_private__)

    # Default resource_name=None deepcopy
    executor_none = VertexAiCodeExecutor()
    executor_none_copy = copy.deepcopy(executor_none)
    self.assertIsNone(executor_none_copy.resource_name)

  # --- Test Lazy Loading & Thread Safety ---

  @mock.patch('vertexai.preview.extensions.Extension')
  def test_lazy_loading_and_caching(self, mock_extension_class):
    """Verifies client is created only on access and is cached."""

    mock_client_instance = mock_extension_class.return_value = mock.MagicMock()

    # 1. Access the property to trigger instantiation (Lazy Loading)
    with self.subTest(msg='Test Lazy Loading'):
      client = self.executor._extension_client
      mock_extension_class.assert_called_once_with(self.mock_resource_name)
      self.assertEqual(client, mock_client_instance)

    # 2. Access again to ensure no re-instantiation (Caching)
    with self.subTest(msg='Test Caching'):
      _ = self.executor._extension_client
      mock_extension_class.assert_called_once()

  @mock.patch.dict(vertex_ai_code_executor.os.environ, {}, clear=True)
  @mock.patch('vertexai.preview.extensions.Extension')
  def test_lazy_loading_env_var_re_resolution(self, mock_extension_class):
    """Verifies changing env var between instances creates respective clients."""
    client_1 = mock.MagicMock()
    client_2 = mock.MagicMock()
    mock_extension_class.side_effect = [client_1, client_2]

    vertex_ai_code_executor.os.environ['CODE_INTERPRETER_EXTENSION_NAME'] = (
        'projects/1/locations/us-central1/extensions/env1'
    )
    executor_1 = VertexAiCodeExecutor()
    self.assertEqual(executor_1._extension_client, client_1)
    mock_extension_class.assert_called_with(
        'projects/1/locations/us-central1/extensions/env1'
    )

    # Reusing with same env var uses cache
    executor_1_again = VertexAiCodeExecutor()
    self.assertEqual(executor_1_again._extension_client, client_1)
    self.assertEqual(mock_extension_class.call_count, 1)

    # Changing env var resolves new client
    vertex_ai_code_executor.os.environ['CODE_INTERPRETER_EXTENSION_NAME'] = (
        'projects/2/locations/us-central1/extensions/env2'
    )
    executor_2 = VertexAiCodeExecutor()
    self.assertEqual(executor_2._extension_client, client_2)
    mock_extension_class.assert_called_with(
        'projects/2/locations/us-central1/extensions/env2'
    )
    self.assertEqual(mock_extension_class.call_count, 2)

  @mock.patch.dict(vertex_ai_code_executor.os.environ, {}, clear=True)
  @mock.patch('vertexai.preview.extensions.Extension')
  def test_lazy_loading_from_hub_when_no_env_var(self, mock_extension_class):
    """Verifies client is loaded from hub when no resource_name or env var is set."""
    mock_hub_client = mock.MagicMock()
    mock_hub_client.gca_resource.name = (
        'projects/hub/locations/us-central1/extensions/hub1'
    )
    mock_extension_class.from_hub.return_value = mock_hub_client

    executor = VertexAiCodeExecutor()
    client = executor._extension_client
    mock_extension_class.from_hub.assert_called_once_with('code_interpreter')
    self.assertEqual(client, mock_hub_client)
    self.assertIn(
        'projects/hub/locations/us-central1/extensions/hub1',
        vertex_ai_code_executor._EXTENSION_CLIENTS,
    )

    # Second access reuses the cached client from hub
    executor_2 = VertexAiCodeExecutor()
    self.assertEqual(executor_2._extension_client, mock_hub_client)
    self.assertEqual(mock_extension_class.from_hub.call_count, 1)

  @mock.patch('vertexai.preview.extensions.Extension')
  def test_concurrent_lazy_loading(self, mock_extension_class):
    """Verifies concurrent accesses safely initialize the client only once."""
    mock_client_instance = mock.MagicMock()

    def create_client(*args, **kwargs):
      del args, kwargs  # Unused
      time.sleep(0.05)
      return mock_client_instance

    mock_extension_class.side_effect = create_client

    barrier = threading.Barrier(5)

    def get_client():
      barrier.wait()
      return self.executor._extension_client

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
      results = list(pool.map(lambda _: get_client(), range(5)))

    mock_extension_class.assert_called_once_with(self.mock_resource_name)
    for client in results:
      self.assertEqual(client, mock_client_instance)

  # --- Test Execution Flow ---

  @mock.patch('vertexai.preview.extensions.Extension')
  def test_execute_code_flow(self, mock_extension_class):
    """Verifies execute_code correctly maps inputs, calls the client, and parses results."""

    # 1. Setup Mocks and Response
    mock_client = mock.MagicMock()
    mock_extension_class.return_value = mock_client
    mock_response = {
        'execution_result': 'Final print output',
        'execution_error': '',
        'output_files': [
            {'name': 'plot.png', 'contents': 'base64_plot_string'},
            {'name': 'data.csv', 'contents': '1,2,3'},
        ],
    }
    mock_client.execute.return_value = mock_response

    # 2. Input Data Preparation
    input_files = self._create_mock_files(
        [('input.txt', 'test content', 'text/plain')]
    )
    input_data = CodeExecutionInput(
        code='df.plot()',
        execution_id='test-session-42',
        input_files=input_files,
    )
    context = mock.MagicMock()

    # 3. Run execution
    result = self.executor.execute_code(context, input_data)

    # 4. Verify client call arguments
    _, kwargs = mock_client.execute.call_args
    actual_code = kwargs['operation_params']['code']
    actual_files = kwargs['operation_params']['files']

    # Assertions for dynamic parts
    self.assertIn(
        'def explore_df(df: pd.DataFrame) -> None:',
        actual_code,
        'Code payload must include the explore_df helper function.',
    )
    self.assertTrue(
        actual_code.strip().endswith('df.plot()'),
        'User code must be appended at the end of the payload.',
    )
    self.assertNotIn(
        'mime_type',
        actual_files[0],
        "Files dict sent to client should NOT contain 'mime_type'.",
    )

    # Assertion for static parts
    self.assertEqual(kwargs['operation_id'], 'execute')
    self.assertEqual(
        kwargs['operation_params']['session_id'], 'test-session-42'
    )
    self.assertEqual(
        kwargs['operation_params']['files'],
        [
            # Ensure 'mime_type' is explicitly removed
            {'name': 'input.txt', 'contents': 'test content'}
        ],
    )

    # 5. Verify Output Parsing
    self.assertEqual(result.stdout, mock_response['execution_result'])
    self.assertEqual(len(result.output_files), 2)

    with self.subTest(msg='Check Image File Parsing'):
      image_file = result.output_files[0]
      self.assertEqual(image_file.name, 'plot.png')
      self.assertEqual(image_file.mime_type, 'image/png')

    with self.subTest(msg='Check CSV File Parsing'):
      csv_file = result.output_files[1]
      self.assertEqual(csv_file.name, 'data.csv')
      self.assertEqual(csv_file.mime_type, 'text/csv')

  # --- Test Error Handling ---

  @mock.patch('vertexai.preview.extensions.Extension')
  def test_execute_code_api_exception(self, mock_extension_class):
    """Verifies that exceptions from the Vertex AI client bubble up correctly."""
    mock_client = mock_extension_class.return_value = mock.MagicMock()

    # Simulate a generic API failure (e.g. 500 error or Timeout)
    mock_client.execute.side_effect = RuntimeError(
        'Vertex AI Service Unavailable'
    )

    input_data = CodeExecutionInput(code="print('fail')", input_files=[])
    context = mock.MagicMock()

    # Verify the executor does not silently swallow critical errors
    with self.assertRaises(RuntimeError) as cm:
      self.executor.execute_code(context, input_data)

    self.assertEqual(str(cm.exception), 'Vertex AI Service Unavailable')

  @mock.patch('vertexai.preview.extensions.Extension')
  def test_execute_code_malformed_response(self, mock_extension_class):
    """Verifies behavior when API returns a response missing required keys."""
    mock_client = mock_extension_class.return_value = mock.MagicMock()

    # Simulate a response that lacks 'output_files' (contract violation)
    mock_client.execute.return_value = {
        'execution_result': 'Success',
        # 'output_files': []  <-- MISSING KEY
    }

    input_data = CodeExecutionInput(code="print('ok')", input_files=[])
    context = mock.MagicMock()

    # Expect a KeyError because the source code accesses ['output_files']
    # directly
    with self.assertRaises(KeyError):
      self.executor.execute_code(context, input_data)

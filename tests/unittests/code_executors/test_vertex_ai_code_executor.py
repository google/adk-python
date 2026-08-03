import copy
from typing import List
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.code_executors.vertex_ai_code_executor import CodeExecutionInput
from google.adk.code_executors.vertex_ai_code_executor import File
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor

InvocationContext = MagicMock


class TestVertexAiCodeExecutor(unittest.TestCase):

  def setUp(self):
    """Set up common fixtures for the tests."""
    self.mock_resource_name = (
        'projects/123/locations/us-central1/extensions/456'
    )
    self.executor = VertexAiCodeExecutor(resource_name=self.mock_resource_name)

  def _create_mock_files(
      self, file_data: List[tuple[str, str, str]]
  ) -> List[File]:
    """Helper to create File objects from (name, content, mime_type)."""
    return [
        File(name=name, content=content, mime_type=mime_type)
        for name, content, mime_type in file_data
    ]

  # --- Test Initialization & Deepcopy Safety ---

  def test_init_is_lazy(self):
    """Verifies __init__ does NOT create the external client."""
    self.assertIsNone(self.executor._code_interpreter_extension)

  def test_deepcopy_safety(self):
    """Verifies that deepcopy works without RecursionError."""
    try:
      executor_copy = copy.deepcopy(self.executor)
    except RecursionError:
      self.fail('deepcopy raised RecursionError! Lazy loading fix failed.')

    self.assertNotEqual(id(self.executor), id(executor_copy))

  # --- Test Lazy Loading ---

  @patch('vertexai.preview.extensions.Extension')
  def test_lazy_loading_and_caching(self, MockExtensionClass):
    """Verifies client is created only on access and is cached."""

    mock_client_instance = MockExtensionClass.return_value = MagicMock()

    # 1. Access the property to trigger instantiation (Lazy Loading)
    with self.subTest(msg='Test Lazy Loading'):
      client = self.executor.extension_client
      MockExtensionClass.assert_called_once_with(self.mock_resource_name)
      self.assertEqual(client, mock_client_instance)

    # 2. Access again to ensure no re-instantiation (Caching)
    with self.subTest(msg='Test Caching'):
      _ = self.executor.extension_client
      MockExtensionClass.assert_called_once()

  # --- Test Execution Flow ---

  @patch('vertexai.preview.extensions.Extension')
  def test_execute_code_flow(self, MockExtensionClass):
    """Verifies execute_code correctly maps inputs, calls the client, and parses results."""

    # 1. Setup Mocks and Response
    mock_client = MagicMock()
    MockExtensionClass.return_value = mock_client
    MOCK_RESPONSE = {
        'execution_result': 'Final print output',
        'execution_error': '',
        'output_files': [
            {'name': 'plot.png', 'contents': 'base64_plot_string'},
            {'name': 'data.csv', 'contents': '1,2,3'},
        ],
    }
    mock_client.execute.return_value = MOCK_RESPONSE

    # 2. Input Data Preparation
    input_files = self._create_mock_files(
        [('input.txt', 'test content', 'text/plain')]
    )
    input_data = CodeExecutionInput(
        code='df.plot()',
        execution_id='test-session-42',
        input_files=input_files,
    )
    context = MagicMock()

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
    self.assertEqual(result.stdout, MOCK_RESPONSE['execution_result'])
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

  @patch('vertexai.preview.extensions.Extension')
  def test_execute_code_api_exception(self, MockExtensionClass):
    """Verifies that exceptions from the Vertex AI client bubble up correctly."""
    mock_client = MockExtensionClass.return_value = MagicMock()

    # Simulate a generic API failure (e.g. 500 error or Timeout)
    mock_client.execute.side_effect = RuntimeError(
        'Vertex AI Service Unavailable'
    )

    input_data = CodeExecutionInput(code="print('fail')", input_files=[])
    context = MagicMock()

    # Verify the executor does not silently swallow critical errors
    with self.assertRaises(RuntimeError) as cm:
      self.executor.execute_code(context, input_data)

    self.assertEqual(str(cm.exception), 'Vertex AI Service Unavailable')

  @patch('vertexai.preview.extensions.Extension')
  def test_execute_code_malformed_response(self, MockExtensionClass):
    """Verifies behavior when API returns a response missing required keys."""
    mock_client = MockExtensionClass.return_value = MagicMock()

    # Simulate a response that lacks 'output_files' (contract violation)
    mock_client.execute.return_value = {
        'execution_result': 'Success',
        # 'output_files': []  <-- MISSING KEY
    }

    input_data = CodeExecutionInput(code="print('ok')", input_files=[])
    context = MagicMock()

    # Expect a KeyError because the source code accesses ['output_files'] directly
    with self.assertRaises(KeyError):
      self.executor.execute_code(context, input_data)

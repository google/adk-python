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

"""Handles Code Execution related logic."""

from __future__ import annotations

import base64
import copy
import dataclasses
import datetime
import logging
import os
import re
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...code_executors.base_code_executor import BaseCodeExecutor
from ...code_executors.built_in_code_executor import BuiltInCodeExecutor
from ...code_executors.code_execution_utils import CodeExecutionInput
from ...code_executors.code_execution_utils import CodeExecutionResult
from ...code_executors.code_execution_utils import CodeExecutionUtils
from ...code_executors.code_execution_utils import File
from ...code_executors.code_executor_context import CodeExecutorContext
from ...events.event import Event
from ...events.event_actions import EventActions
from ...models.llm_response import LlmResponse
from ...utils.context_utils import Aclosing
from ._base_llm_processor import BaseLlmRequestProcessor
from ._base_llm_processor import BaseLlmResponseProcessor

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)


@dataclasses.dataclass
class DataFileUtil:
  """A structure that contains a data file name and its content."""

  extension: str
  """
  The file extension (e.g., ".csv").
  """

  loader_code_template: str
  """
  The code template to load the data file.
  """


_DATA_FILE_UTIL_MAP = {
    'text/csv': DataFileUtil(
        extension='.csv',
        loader_code_template="pd.read_csv('{filename}')",
    ),
}

_DATA_FILE_HELPER_LIB = '''
import pandas as pd

def explore_df(df: pd.DataFrame) -> None:
  """Prints some information about a pandas DataFrame."""

  with pd.option_context(
      'display.max_columns', None, 'display.expand_frame_repr', False
  ):
    # Print the column names to never encounter KeyError when selecting one.
    df_dtypes = df.dtypes

    # Obtain information about data types and missing values.
    df_nulls = (len(df) - df.isnull().sum()).apply(
        lambda x: f'{x} / {df.shape[0]} non-null'
    )

    # Explore unique total values in columns using `.unique()`.
    df_unique_count = df.apply(lambda x: len(x.unique()))

    # Explore unique values in columns using `.unique()`.
    df_unique = df.apply(lambda x: crop(str(list(x.unique()))))

    df_info = pd.concat(
        (
            df_dtypes.rename('Dtype'),
            df_nulls.rename('Non-Null Count'),
            df_unique_count.rename('Unique Values Count'),
            df_unique.rename('Unique Values'),
        ),
        axis=1,
    )
    df_info.index.name = 'Columns'
    print(f"""Total rows: {df.shape[0]}
Total columns: {df.shape[1]}

{df_info}""")
'''


class _CodeExecutionRequestProcessor(BaseLlmRequestProcessor):
  """Processes code execution requests."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    if not isinstance(invocation_context.agent, LlmAgent):
      return
    if not invocation_context.agent.code_executor:
      return

    async with Aclosing(
        _run_pre_processor(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        yield event

    # Convert the code execution parts to text parts.
    if not isinstance(invocation_context.agent.code_executor, BaseCodeExecutor):
      return
    for content in llm_request.contents:
      CodeExecutionUtils.convert_code_execution_parts(
          content,
          invocation_context.agent.code_executor.code_block_delimiters[0]
          if invocation_context.agent.code_executor.code_block_delimiters
          else ('', ''),
          invocation_context.agent.code_executor.execution_result_delimiters,
      )


request_processor = _CodeExecutionRequestProcessor()


class _CodeExecutionResponseProcessor(BaseLlmResponseProcessor):
  """Processes code execution responses."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    # Skip if the response is partial (streaming).
    if llm_response.partial:
      return

    async with Aclosing(
        _run_post_processor(invocation_context, llm_response)
    ) as agen:
      async for event in agen:
        yield event


response_processor = _CodeExecutionResponseProcessor()


async def _run_pre_processor(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> AsyncGenerator[Event, None]:
  """Pre-process the user message by adding the user message to the Colab notebook."""
  from ...agents.llm_agent import LlmAgent

  if not isinstance(invocation_context.agent, LlmAgent):
    return

  agent = invocation_context.agent
  code_executor = agent.code_executor

  if not code_executor or not isinstance(code_executor, BaseCodeExecutor):
    return

  if isinstance(code_executor, BuiltInCodeExecutor):
    code_executor.process_llm_request(llm_request)
    return

  if not code_executor.optimize_data_file:
    return

  code_executor_context = CodeExecutorContext(invocation_context.session.state)

  # Skip if the error count exceeds the max retry attempts.
  if (
      code_executor_context.get_error_count(invocation_context.invocation_id)
      >= code_executor.error_retry_attempts
  ):
    return

  # [Step 1] Extract data files from the session_history and store them in
  # memory. Meanwhile, mutate the inline data file to text part in session
  # history from all turns.
  all_input_files = _extrac_and_replace_inline_files(
      code_executor_context, llm_request
  )

  # [Step 2] Run Explore_Df code on the data files from the current turn. We
  # only need to explore the new data files because the previous data files
  # should already be explored and cached in the code execution runtime.
  processed_file_names = set(code_executor_context.get_processed_file_names())
  files_to_process = [
      f for f in all_input_files if f.name not in processed_file_names
  ]
  for file in files_to_process:
    code_str = _get_data_file_preprocessing_code(file)
    # Skip for unsupported file or executor types.
    if not code_str:
      return

    # Emit the code to execute, and add it to the LLM request.
    code_content = types.Content(
        role='model',
        parts=[
            types.Part(text=f'Processing input file: `{file.name}`'),
            CodeExecutionUtils.build_executable_code_part(code_str),
        ],
    )
    llm_request.contents.append(copy.deepcopy(code_content))
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=agent.name,
        branch=invocation_context.branch,
        content=code_content,
    )

    code_execution_result = code_executor.execute_code(
        invocation_context,
        CodeExecutionInput(
            code=code_str,
            input_files=[file],
            execution_id=_get_or_set_execution_id(
                invocation_context, code_executor_context
            ),
        ),
    )
    logger.debug('Executed code:\n```\n%s\n```', code_str)
    # Update the processing results to code executor context.
    code_executor_context.update_code_execution_result(
        invocation_context.invocation_id,
        code_str,
        code_execution_result.stdout,
        code_execution_result.stderr,
    )
    code_executor_context.add_processed_file_names([file.name])

    # Emit the execution result, and add it to the LLM request.
    execution_result_event = await _post_process_code_execution_result(
        invocation_context, code_executor_context, code_execution_result
    )
    yield execution_result_event
    llm_request.contents.append(copy.deepcopy(execution_result_event.content))


async def _run_post_processor(
    invocation_context: InvocationContext,
    llm_response,
) -> AsyncGenerator[Event, None]:
  """Post-process the model response by extracting and executing the first code block."""
  agent = invocation_context.agent
  code_executor = agent.code_executor

  if not code_executor or not isinstance(code_executor, BaseCodeExecutor):
    return
  if not llm_response or not llm_response.content:
    return

  if isinstance(code_executor, BuiltInCodeExecutor):
    event_actions = EventActions()

    # If an image is generated, save it to the artifact service and add it to
    # the event actions.
    for part in llm_response.content.parts:
      if part.inline_data and part.inline_data.mime_type.startswith('image/'):
        if invocation_context.artifact_service is None:
          raise ValueError('Artifact service is not initialized.')

        if part.inline_data.display_name:
          file_name = part.inline_data.display_name
        else:
          now = datetime.datetime.now().astimezone()
          timestamp = now.strftime('%Y%m%d_%H%M%S')
          file_extension = part.inline_data.mime_type.split('/')[-1]
          file_name = f'{timestamp}.{file_extension}'

        version = await invocation_context.artifact_service.save_artifact(
            app_name=invocation_context.app_name,
            user_id=invocation_context.user_id,
            session_id=invocation_context.session.id,
            filename=file_name,
            artifact=types.Part.from_bytes(
                data=part.inline_data.data,
                mime_type=part.inline_data.mime_type,
            ),
        )
        event_actions.artifact_delta[file_name] = version
        part.inline_data = None
        part.text = f'Saved as artifact: {file_name}. '

    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=agent.name,
        branch=invocation_context.branch,
        actions=event_actions,
    )
    return

  code_executor_context = CodeExecutorContext(invocation_context.session.state)
  # Skip if the error count exceeds the max retry attempts.
  if (
      code_executor_context.get_error_count(invocation_context.invocation_id)
      >= code_executor.error_retry_attempts
  ):
    return

  # [Step 1] Extract code from the model predict response and truncate the
  # content to the part with the first code block.
  response_content = llm_response.content
  code_str = CodeExecutionUtils.extract_code_and_truncate_content(
      response_content, code_executor.code_block_delimiters
  )
  # Terminal state: no code to execute.
  if not code_str:
    return

  # [Step 2] Executes the code and emit 2 Events for code and execution result.
  yield Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      branch=invocation_context.branch,
      content=response_content,
      actions=EventActions(),
  )

  code_execution_result = code_executor.execute_code(
      invocation_context,
      CodeExecutionInput(
          code=code_str,
          input_files=code_executor_context.get_input_files(),
          execution_id=_get_or_set_execution_id(
              invocation_context, code_executor_context
          ),
      ),
  )
  logger.debug('Executed code:\n```\n%s\n```', code_str)
  code_executor_context.update_code_execution_result(
      invocation_context.invocation_id,
      code_str,
      code_execution_result.stdout,
      code_execution_result.stderr,
  )
  yield await _post_process_code_execution_result(
      invocation_context, code_executor_context, code_execution_result
  )

  # [Step 3] Skip processing the original model response
  # to continue code generation loop.
  llm_response.content = None


def _extrac_and_replace_inline_files(
    code_executor_context: CodeExecutorContext,
    llm_request: LlmRequest,
) -> list[File]:
  """Extracts and replaces inline files with file names in the LLM request."""
  all_input_files = code_executor_context.get_input_files()
  saved_file_names = set(f.name for f in all_input_files)

  # [Step 1] Process input files from LlmRequest and cache them in CodeExecutor.
  for i in range(len(llm_request.contents)):
    content = llm_request.contents[i]
    # Only process the user message.
    if content.role != 'user' and not content.parts:
      continue

    for j in range(len(content.parts)):
      part = content.parts[j]
      # Skip if the inline data is not supported.
      if (
          not part.inline_data
          or part.inline_data.mime_type not in _DATA_FILE_UTIL_MAP
      ):
        continue

      # Replace the inline data file with a file name placeholder.
      mime_type = part.inline_data.mime_type
      file_name = f'data_{i+1}_{j+1}' + _DATA_FILE_UTIL_MAP[mime_type].extension
      llm_request.contents[i].parts[j] = types.Part(
          text='\nAvailable file: `%s`\n' % file_name
      )

      # Add the inlne data as input file to the code executor context.
      file = File(
          name=file_name,
          content=CodeExecutionUtils.get_encoded_file_content(
              part.inline_data.data
          ).decode(),
          mime_type=mime_type,
      )
      if file_name not in saved_file_names:
        code_executor_context.add_input_files([file])
        all_input_files.append(file)

  return all_input_files


def _get_or_set_execution_id(
    invocation_context: InvocationContext,
    code_executor_context: CodeExecutorContext,
) -> Optional[str]:
  """Returns the ID for stateful code execution or None if not stateful."""
  if not invocation_context.agent.code_executor.stateful:
    return None

  execution_id = code_executor_context.get_execution_id()
  if not execution_id:
    execution_id = invocation_context.session.id
    code_executor_context.set_execution_id(execution_id)
  return execution_id


async def _post_process_code_execution_result(
    invocation_context: InvocationContext,
    code_executor_context: CodeExecutorContext,
    code_execution_result: CodeExecutionResult,
) -> Event:
  """Post-process the code execution result and emit an Event."""
  if invocation_context.artifact_service is None:
    raise ValueError('Artifact service is not initialized.')

  result_content = types.Content(
      role='model',
      parts=[
          CodeExecutionUtils.build_code_execution_result_part(
              code_execution_result
          ),
      ],
  )
  event_actions = EventActions(
      state_delta=code_executor_context.get_state_delta()
  )

  # Handle code execution error retry.
  if code_execution_result.stderr:
    code_executor_context.increment_error_count(
        invocation_context.invocation_id
    )
  else:
    code_executor_context.reset_error_count(invocation_context.invocation_id)

  # Handle output files.
  for output_file in code_execution_result.output_files:
    version = await invocation_context.artifact_service.save_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        filename=output_file.name,
        artifact=types.Part.from_bytes(
            data=get_content_as_bytes(output_file.content),
            mime_type=output_file.mime_type,
        ),
    )
    event_actions.artifact_delta[output_file.name] = version

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      branch=invocation_context.branch,
      content=result_content,
      actions=event_actions,
  )


def get_content_as_bytes(output_content: str | bytes) -> bytes:
  """Converts output_content to bytes.

  - If output_content is already bytes, it's returned as is.
  - If output_content is a string: convert base64-decoded to bytes.

  Args:
    output_content: The content, which can be a str or bytes.

  Returns:
    The content as a bytes object.
  """
  if isinstance(output_content, bytes):
    # Already bytes, no conversion needed.
    return output_content

  return base64.b64decode(output_content)


def _get_data_file_preprocessing_code(file: File) -> Optional[str]:
  """Returns the code to explore the data file."""

  def _get_normalized_file_name(file_name: str) -> str:
    var_name, _ = os.path.splitext(file_name)
    # Replace non-alphanumeric characters with underscores
    var_name = re.sub(r'[^a-zA-Z0-9_]', '_', var_name)

    # If the filename starts with a digit, prepend an underscore
    if var_name[0].isdigit():
      var_name = '_' + var_name
    return var_name

  if file.mime_type not in _DATA_FILE_UTIL_MAP:
    return

  var_name = _get_normalized_file_name(file.name)
  loader_code = _DATA_FILE_UTIL_MAP[file.mime_type].loader_code_template.format(
      filename=file.name
  )
  return f"""
{_DATA_FILE_HELPER_LIB}

# Load the dataframe.
{var_name} = {loader_code}

# Use `explore_df` to guide my analysis.
explore_df({var_name})
"""

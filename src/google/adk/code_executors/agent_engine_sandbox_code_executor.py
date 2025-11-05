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

from __future__ import annotations

import json
import logging
import mimetypes
import re
from typing import Optional

from typing_extensions import override
import vertexai
from vertexai import types

from ..agents.invocation_context import InvocationContext
from ..utils.feature_decorator import experimental
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult
from .code_execution_utils import File

logger = logging.getLogger('google_adk.' + __name__)


@experimental
class AgentEngineSandboxCodeExecutor(BaseCodeExecutor):
  """A code executor that uses Agent Engine Code Execution Sandbox to execute code.

  Attributes:
    sandbox_resource_name: If set, load the existing resource name of the code
      interpreter extension instead of creating a new one. Format:
      projects/123/locations/us-central1/reasoningEngines/456/sandboxEnvironments/789
  """

  sandbox_resource_name: str = None

  def __init__(
      self,
      sandbox_resource_name: Optional[str] = None,
      agent_engine_resource_name: Optional[str] = None,
      **data,
  ):
    """Initializes the AgentEngineSandboxCodeExecutor.

    Args:
      sandbox_resource_name: If set, load the existing resource name of code
        execution sandbox, if not set, create a new one. Format:
        projects/123/locations/us-central1/reasoningEngines/456/
        sandboxEnvironments/789
      agent_engine_resource_name: The resource name of the agent engine to use
        to create the code execution sandbox. Format:
        projects/123/locations/us-central1/reasoningEngines/456, when both
        sandbox_resource_name and agent_engine_resource_name are set,
        agent_engine_resource_name will be ignored.
      **data: Additional keyword arguments to be passed to the base class.
    """
    super().__init__(**data)
    sandbox_resource_name_pattern = r'^projects/([a-zA-Z0-9-_]+)/locations/([a-zA-Z0-9-_]+)/reasoningEngines/(\d+)/sandboxEnvironments/(\d+)$'
    agent_engine_resource_name_pattern = r'^projects/([a-zA-Z0-9-_]+)/locations/([a-zA-Z0-9-_]+)/reasoningEngines/(\d+)$'

    if sandbox_resource_name is not None:
      self.sandbox_resource_name = sandbox_resource_name
      self._project_id, self._location = (
          self._get_project_id_and_location_from_resource_name(
              sandbox_resource_name, sandbox_resource_name_pattern
          )
      )
    elif agent_engine_resource_name is not None:
      self._project_id, self._location = (
          self._get_project_id_and_location_from_resource_name(
              agent_engine_resource_name, agent_engine_resource_name_pattern
          )
      )
      # @TODO - Add TTL for sandbox creation after it is available
      # in SDK.
      operation = self._get_api_client().agent_engines.sandboxes.create(
          spec={'code_execution_environment': {}},
          name=agent_engine_resource_name,
          config=types.CreateAgentEngineSandboxConfig(
              display_name='default_sandbox'
          ),
      )
      self.sandbox_resource_name = operation.response.name
    else:
      raise ValueError(
          'Either sandbox_resource_name or agent_engine_resource_name must be'
          ' set.'
      )

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    # Execute the code.
    input_data = {
        'code': code_execution_input.code,
    }
    if code_execution_input.input_files:
      input_data['files'] = [
          {
              'name': f.name,
              'contents': f.content,
              'mimeType': f.mime_type,
          }
          for f in code_execution_input.input_files
      ]

    code_execution_response = (
        self._get_api_client().agent_engines.sandboxes.execute_code(
            name=self.sandbox_resource_name,
            input_data=input_data,
        )
    )
    logger.debug('Executed code:\n```\n%s\n```', code_execution_input.code)
    saved_files = []
    stdout = ''
    stderr = ''
    for output in code_execution_response.outputs:
      if output.mime_type == 'application/json' and (
          output.metadata is None
          or output.metadata.attributes is None
          or 'file_name' not in output.metadata.attributes
      ):
        json_output_data = json.loads(output.data.decode('utf-8'))
        stdout = json_output_data.get('stdout', '')
        stderr = json_output_data.get('stderr', '')
      else:
        file_name = ''
        if (
            output.metadata is not None
            and output.metadata.attributes is not None
        ):
          file_name = output.metadata.attributes.get('file_name', b'').decode(
              'utf-8'
          )
        mime_type = output.mime_type
        if not mime_type:
          mime_type, _ = mimetypes.guess_type(file_name)
        saved_files.append(
            File(
                name=file_name,
                content=output.data,
                mime_type=mime_type,
            )
        )

    # Collect the final result.
    return CodeExecutionResult(
        stdout=stdout,
        stderr=stderr,
        output_files=saved_files,
    )

  def _get_api_client(self):
    """Instantiates an API client for the given project and location.

    It needs to be instantiated inside each request so that the event loop
    management can be properly propagated.

    Returns:
      An API client for the given project and location.
    """
    return vertexai.Client(project=self._project_id, location=self._location)

  def _get_project_id_and_location_from_resource_name(
      self, resource_name: str, pattern: str
  ) -> tuple[str, str]:
    """Extracts the project ID and location from the resource name."""
    match = re.fullmatch(pattern, resource_name)

    if not match:
      raise ValueError(f'resource name {resource_name} is not valid.')

    return match.groups()[0], match.groups()[1]

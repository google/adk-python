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

import logging
from typing import Optional

from google.genai import types

from ..agents.invocation_context import InvocationContext
from .base_plugin import BasePlugin

logger = logging.getLogger('google_adk.' + __name__)


class SaveFilesAsArtifactsPlugin(BasePlugin):
  """A plugin that saves files embedded in user messages as artifacts.

  This is useful to allow users to upload files in the chat experience and have
  those files available to the agent.

  We use Blob.display_name to determine
  the file name. Artifacts with the same name will be overwritten. A placeholder
  with the artifact name will be put in place of the embedded file in the user
  message so the model knows where to find the file. You may want to add
  load_artifacts tool to the agent, or load the artifacts in your own tool to
  use the files.
  """

  def __init__(self, name: str = 'save_files_as_artifacts_plugin'):
    """Initialize the save files as artifacts plugin.

    Args:
      name: The name of the plugin instance.
    """
    super().__init__(name)

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    """Process user message and save any attached files as artifacts."""
    if not invocation_context.artifact_service:
      logger.warning(
          'Artifact service is not set. SaveFilesAsArtifactsPlugin'
          ' will not be enabled.'
      )
      return user_message

    if not user_message.parts:
      return user_message

    for i, part in enumerate(user_message.parts):
      if part.inline_data is None:
        continue

      try:
        # Use display_name if available, otherwise generate a filename
        file_name = part.inline_data.display_name
        if not file_name:
          file_name = f'artifact_{invocation_context.invocation_id}_{i}'
          logger.info(
              f'No display_name found, using generated filename: {file_name}'
          )

        await invocation_context.artifact_service.save_artifact(
            app_name=invocation_context.app_name,
            user_id=invocation_context.user_id,
            session_id=invocation_context.session.id,
            filename=file_name,
            artifact=part,
        )

        # Replace the inline data with a placeholder text
        user_message.parts[i] = types.Part(
            text=f'[Uploaded Artifact: "{file_name}"]'
        )
        logger.info(f'Successfully saved artifact: {file_name}')

      except Exception as e:
        logger.error(f'Failed to save artifact for part {i}: {e}')
        # Keep the original part if saving fails
        continue

    return user_message

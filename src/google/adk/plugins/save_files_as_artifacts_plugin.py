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

import copy
import logging
from typing import Optional

from google.genai import types

from ..agents.invocation_context import InvocationContext
from .base_plugin import BasePlugin

logger = logging.getLogger('google_adk.' + __name__)


class SaveFilesAsArtifactsPlugin(BasePlugin):
  """A plugin that saves files embedded in user messages as artifacts.

  This is useful to allow users to upload files in the chat experience and have
  those files available to the agent within the current session.

  We use Blob.display_name to determine the file name. By default, artifacts are
  session-scoped. For cross-session persistence, prefix the filename with
  "user:".
  Artifacts with the same name will be overwritten. A placeholder with the
  artifact name will be put in place of the embedded file in the user message
  so the model knows where to find the file. You may want to add load_artifacts
  tool to the agent, or load the artifacts in your own tool to use the files.
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
      return None

    new_parts = []
    modified = False

    for i, part in enumerate(user_message.parts):
      if part.inline_data is None:
        new_parts.append(part)
        continue

      try:
        # Use display_name if available, otherwise generate a filename
        file_name = part.inline_data.display_name
        if not file_name:
          file_name = f'artifact_{invocation_context.invocation_id}_{i}'
          logger.info(
              f'No display_name found, using generated filename: {file_name}'
          )

        # Store original filename for display to user/ placeholder
        display_name = file_name

        # Create a copy to stop mutation of the saved artifact if the original part is modified
        await invocation_context.artifact_service.save_artifact(
            app_name=invocation_context.app_name,
            user_id=invocation_context.user_id,
            session_id=invocation_context.session.id,
            filename=file_name,
            artifact=copy.copy(part),
        )

        # Replace the inline data with a placeholder text (using the clean name)
        new_parts.append(
            types.Part(text=f'[Uploaded Artifact: "{display_name}"]')
        )
        modified = True
        logger.info(f'Successfully saved artifact: {file_name}')

      except Exception as e:
        logger.error(f'Failed to save artifact for part {i}: {e}')
        # Keep the original part if saving fails
        new_parts.append(part)
        continue

    if modified:
      return types.Content(role=user_message.role, parts=new_parts)
    else:
      return None

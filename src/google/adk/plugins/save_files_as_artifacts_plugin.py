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
import urllib.parse

from google.genai import types

from ..agents.invocation_context import InvocationContext
from .base_plugin import BasePlugin

logger = logging.getLogger('google_adk.' + __name__)

# Schemes supported by our current LLM connectors. Vertex exposes `gs://` while
# hosted endpoints use HTTPS. Expand this list when BaseLlm surfaces provider
# capabilities.
_MODEL_ACCESSIBLE_URI_SCHEMES = {'gs', 'https', 'http'}


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
        inline_data = part.inline_data
        file_name = inline_data.display_name
        if not file_name:
          file_name = f'artifact_{invocation_context.invocation_id}_{i}'
          logger.info(
              f'No display_name found, using generated filename: {file_name}'
          )

        # Store original filename for display to user/ placeholder
        display_name = file_name

        # Create a copy to stop mutation of the saved artifact if the original part is modified
        version = await invocation_context.artifact_service.save_artifact(
            app_name=invocation_context.app_name,
            user_id=invocation_context.user_id,
            session_id=invocation_context.session.id,
            filename=file_name,
            artifact=copy.copy(part),
        )

        placeholder_part = types.Part(
            text=f'[Uploaded Artifact: "{display_name}"]'
        )
        new_parts.append(placeholder_part)

        file_part = await self._build_file_reference_part(
            invocation_context=invocation_context,
            filename=file_name,
            version=version,
            mime_type=inline_data.mime_type,
            display_name=display_name,
        )
        if file_part:
          new_parts.append(file_part)
        else:
          logger.debug(
              'Artifact %s is not exposed via a model-accessible URI; keeping'
              ' inline data in user message.',
              file_name,
          )
          new_parts.append(part)

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

  async def _build_file_reference_part(
      self,
      *,
      invocation_context: InvocationContext,
      filename: str,
      version: int,
      mime_type: Optional[str],
      display_name: str,
  ) -> Optional[types.Part]:
    """Constructs a file reference part if the artifact URI is model-accessible."""

    artifact_service = invocation_context.artifact_service
    if not artifact_service:
      return None

    try:
      artifact_version = await artifact_service.get_artifact_version(
          app_name=invocation_context.app_name,
          user_id=invocation_context.user_id,
          session_id=invocation_context.session.id,
          filename=filename,
          version=version,
      )
    except Exception as exc:  # pylint: disable=broad-except
      logger.warning(
          'Failed to resolve artifact version for %s: %s', filename, exc
      )
      return None

    if (
        not artifact_version
        or not artifact_version.canonical_uri
        or not _is_model_accessible_uri(artifact_version.canonical_uri)
    ):
      return None

    file_data = types.FileData(
        file_uri=artifact_version.canonical_uri,
        mime_type=mime_type or artifact_version.mime_type,
        display_name=display_name,
    )
    return types.Part(file_data=file_data)


def _is_model_accessible_uri(uri: str) -> bool:
  try:
    parsed = urllib.parse.urlparse(uri)
  except ValueError:
    return False

  if not parsed.scheme:
    return False

  return parsed.scheme.lower() in _MODEL_ACCESSIBLE_URI_SCHEMES

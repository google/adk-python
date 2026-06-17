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
from __future__ import annotations

from typing import Union

from a2a.server.events import Event as A2AEvent
from a2a.types import Artifact
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskStatusUpdateEvent
from google.adk.a2a.executor.config import ExecuteInterceptor
from google.adk.a2a.executor.config import ExecutorContext

from ....events.event import Event
from ...converters.part_converter import convert_genai_part_to_a2a_part


async def _after_agent(
    ctx: ExecutorContext, a2a_event: A2AEvent, adk_event: Event
) -> Union[A2AEvent, list[A2AEvent]]:
  """After agent interceptor that includes artifacts in A2A events."""
  if isinstance(a2a_event, (TaskStatusUpdateEvent, TaskArtifactUpdateEvent)):
    artifact_service = ctx.runner.artifact_service
    if artifact_service and adk_event.actions.artifact_delta:
      new_events = []
      for filename, version in adk_event.actions.artifact_delta.items():
        genai_part = await artifact_service.load_artifact(
            app_name=ctx.app_name,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            filename=filename,
            version=version,
        )
        if genai_part:
          a2a_part = convert_genai_part_to_a2a_part(genai_part)
          if a2a_part:
            a2a_artifact = Artifact(
                artifact_id=f"{filename}_{version}",
                name=filename,
                parts=[a2a_part],
            )
            new_event = TaskArtifactUpdateEvent(
                task_id=a2a_event.task_id,
                context_id=a2a_event.context_id,
                artifact=a2a_artifact,
                metadata=a2a_event.metadata,
                append=False,
                last_chunk=True,
            )
            new_events.append(new_event)

      adk_event.actions.artifact_delta = {}

      if new_events:
        return [a2a_event] + new_events

  return a2a_event


include_artifacts_in_a2a_event_interceptor = ExecuteInterceptor(
    after_event=_after_agent
)

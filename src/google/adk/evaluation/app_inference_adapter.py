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

from typing import Optional
from typing import TYPE_CHECKING

from ..artifacts.base_artifact_service import BaseArtifactService
from ..memory.base_memory_service import BaseMemoryService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ._retry_options_utils import EnsureRetryOptionsPlugin
from .request_intercepter_plugin import _RequestIntercepterPlugin
from .simulation.user_simulator import UserSimulator

if TYPE_CHECKING:
  from .eval_case import SessionInput


class AppInferenceAdapter:
  """Adapter to generate inferences from App without importing cli.*"""

  @staticmethod
  async def generate_inferences_from_app(
      app,
      user_simulator: UserSimulator,
      initial_session: Optional["SessionInput"],
      session_id: str,
      session_service: BaseSessionService,
      artifact_service: BaseArtifactService,
      memory_service: BaseMemoryService,
  ):
    """Shared app inference logic extracted from EvaluationGenerator."""

    user_id = initial_session.user_id if initial_session else "test_user_id"
    app_name = initial_session.app_name if initial_session else app.name

    # Create session
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_session.state if initial_session else {},
    )

    # Add evaluation-required plugins
    request_intercepter_plugin = _RequestIntercepterPlugin(
        name="request_intercepter_plugin"
    )
    ensure_retry_options_plugin = EnsureRetryOptionsPlugin(
        name="ensure_retry_options"
    )

    # Duplicate app safely
    app_for_runner = app.model_copy(deep=True)

    plugin_names = {p.name for p in app_for_runner.plugins}
    if request_intercepter_plugin.name not in plugin_names:
      app_for_runner.plugins.append(request_intercepter_plugin)
    if ensure_retry_options_plugin.name not in plugin_names:
      app_for_runner.plugins.append(ensure_retry_options_plugin)

    # Run simulation loop via runner
    async with Runner(
        app=app_for_runner,
        session_service=session_service,
        artifact_service=artifact_service,
        memory_service=memory_service,
    ) as runner:

      # Reuse existing eval user simulation loop
      from .evaluation_generator import EvaluationGenerator

      return await EvaluationGenerator._run_user_simulation_loop(
          runner=runner,
          user_id=user_id,
          session_id=session_id,
          user_simulator=user_simulator,
          request_intercepter_plugin=request_intercepter_plugin,
      )

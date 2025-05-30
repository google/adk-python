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

import asyncio
import logging
from typing import List

from ...runners import Runner

logger = logging.getLogger("google_adk." + __name__)


async def close_runners(runners: List[Runner]) -> None:
  cleanup_tasks = [asyncio.create_task(runner.close()) for runner in runners]
  if cleanup_tasks:
    # Wait for all cleanup tasks with timeout
    done, pending = await asyncio.wait(
        cleanup_tasks,
        timeout=30.0,  # 30 second timeout for cleanup
        return_when=asyncio.ALL_COMPLETED,
    )

    # If any tasks are still pending, log it
    if pending:
      logger.warning(
          "%s runner close tasks didn't complete in time", len(pending)
      )
      for task in pending:
        task.cancel()

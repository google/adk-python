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

"""Agent optimization: run-scoped governance surface (experimental).

The run-context contract is exported here as the supported public surface;
optimizer and sampler classes remain importable from their submodules.
"""

from ._run_context import ContextAlreadyAttachedError
from ._run_context import ModelCallEvent
from ._run_context import ModelCallState
from ._run_context import OptimizationBudgetExceeded
from ._run_context import OptimizationBudgets
from ._run_context import OptimizationCancelledError
from ._run_context import OptimizationFailedError
from ._run_context import OptimizationProviderError
from ._run_context import OptimizationRunContext
from ._run_context import OptimizationRunContextError
from ._run_context import OptimizationRunFinalizedError
from ._run_context import OptimizationRunSnapshot
from ._run_context import OptimizerCapabilities
from ._run_context import RunStatus
from ._run_context import STAGE_CANDIDATE_GENERATION
from ._run_context import STAGE_REFLECTION
from ._run_context import TokenBudgetStatus
from ._run_context import UnsupportedOptimizationContextError
from ._run_context import UsageCoverage

__all__ = [
    "ContextAlreadyAttachedError",
    "ModelCallEvent",
    "ModelCallState",
    "OptimizationBudgetExceeded",
    "OptimizationBudgets",
    "OptimizationCancelledError",
    "OptimizationFailedError",
    "OptimizationProviderError",
    "OptimizationRunContext",
    "OptimizationRunContextError",
    "OptimizationRunFinalizedError",
    "OptimizationRunSnapshot",
    "OptimizerCapabilities",
    "RunStatus",
    "STAGE_CANDIDATE_GENERATION",
    "STAGE_REFLECTION",
    "TokenBudgetStatus",
    "UnsupportedOptimizationContextError",
    "UsageCoverage",
]

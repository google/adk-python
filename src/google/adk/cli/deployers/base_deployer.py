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

from abc import ABC
from abc import abstractmethod
from typing import Tuple


class Deployer(ABC):

  @abstractmethod
  def deploy(
      self,
      temp_folder: str,
      service_name: str,
      provider_args: Tuple[str],
      env_vars: Tuple[str],
      **kwargs,
  ):
    """Deploys the agent to the target platform."""
    pass

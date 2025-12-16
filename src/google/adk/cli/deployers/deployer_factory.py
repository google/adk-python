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

from ..deployers.cloud_run_deployer import CloudRunDeployer
from ..deployers.docker_deployer import DockerDeployer

# Future deployers can be added here


class DeployerFactory:

  @staticmethod
  def get_deployer(cloud_provider: str):
    """Returns the appropriate deployer based on the cloud provider."""
    deployers = {
        'docker': DockerDeployer(),
        'cloud_run': CloudRunDeployer(),
        # Future providers: 'aws': AWSDeployer(), 'k8s': KubernetesDeployer()
    }

    if cloud_provider not in deployers:
      raise ValueError(f'Unsupported cloud provider: {cloud_provider}')

    return deployers[cloud_provider]

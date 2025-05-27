from ..deployers.gcp_deployer import GCPDeployer
from ..deployers.docker_deployer import DockerDeployer
# Future deployers can be added here


class DeployerFactory:

  @staticmethod
  def get_deployer(cloud_provider: str):
    """Returns the appropriate deployer based on the cloud provider."""
    deployers = {
        'docker': DockerDeployer(),
        'gcp': GCPDeployer(),
        # Future providers: 'aws': AWSDeployer(), 'k8s': KubernetesDeployer()
    }

    if cloud_provider not in deployers:
      raise ValueError(f'Unsupported cloud provider: {cloud_provider}')

    return deployers[cloud_provider]

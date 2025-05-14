from ..deployers.gcp_deployer import GCPDeployer
from ..deployers.local_docker_deployer import LocalDockerDeployer
# Future deployers can be added here

class DeployerFactory:
    @staticmethod
    def get_deployer(cloud_provider: str):
        """Returns the appropriate deployer based on the cloud provider."""
        deployers = {
            'local': LocalDockerDeployer(),
            'gcp': GCPDeployer(),
            # Future providers: 'aws': AWSDeployer(), 'k8s': KubernetesDeployer()
        }
        
        if cloud_provider not in deployers:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")
        
        return deployers[cloud_provider]

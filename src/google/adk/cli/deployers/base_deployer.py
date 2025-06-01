from abc import ABC
from abc import abstractmethod
from typing import Dict


class Deployer(ABC):

  @abstractmethod
  def deploy(
      self,
      temp_folder: str,
      service_name: str,
      provider_args: Dict[str, str],
      env_vars: Dict[str, str],
      **kwargs,
  ):
    """Deploys the agent to the target platform."""
    pass

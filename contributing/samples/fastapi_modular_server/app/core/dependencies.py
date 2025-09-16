# dependencies.py - Improved version
from functools import lru_cache
import logging

from app.config.settings import Settings
from app.core.mapping.sse_event_mapper import SSEEventMapper
from fastapi import Depends
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.cli.utils.agent_loader import AgentLoader
from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import BaseSessionService
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService

logger = logging.getLogger(__name__)


# Dependency to get settings
def get_settings() -> Settings:
  """Get application settings."""
  from app.config.settings import settings  # Import here to avoid circular imports

  return settings


class ADKServices:
  """Container for ADK services to avoid long parameter lists."""

  def __init__(
      self,
      agent_loader: AgentLoader,
      session_service: BaseSessionService,
      memory_service: InMemoryMemoryService,
      artifact_service: InMemoryArtifactService,
      credential_service: InMemoryCredentialService,
      eval_sets_manager: LocalEvalSetsManager,
      eval_set_results_manager: LocalEvalSetResultsManager,
  ):
    self.agent_loader = agent_loader
    self.session_service = session_service
    self.memory_service = memory_service
    self.artifact_service = artifact_service
    self.credential_service = credential_service
    self.eval_sets_manager = eval_sets_manager
    self.eval_set_results_manager = eval_set_results_manager


def _create_adk_services_impl(settings: Settings) -> ADKServices:
  """Internal function to create ADK services."""
  try:
    # Ensure directories exist
    settings.agent_parent_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Looking for agents in: {settings.agent_parent_dir}')

    # Create services
    agent_loader = AgentLoader(agents_dir=str(settings.agent_parent_dir))
    session_service = (
        DatabaseSessionService(db_url=settings.session_db_url)
        if settings.session_db_url
        else InMemorySessionService()
    )
    memory_service = InMemoryMemoryService()
    artifact_service = InMemoryArtifactService()
    credential_service = InMemoryCredentialService()
    eval_sets_manager = LocalEvalSetsManager(
        agents_dir=str(settings.agent_parent_dir)
    )
    eval_set_results_manager = LocalEvalSetResultsManager(
        agents_dir=str(settings.agent_parent_dir)
    )

    logger.info('All ADK services created successfully')

    return ADKServices(
        agent_loader=agent_loader,
        session_service=session_service,
        memory_service=memory_service,
        artifact_service=artifact_service,
        credential_service=credential_service,
        eval_sets_manager=eval_sets_manager,
        eval_set_results_manager=eval_set_results_manager,
    )

  except Exception as e:
    logger.error(f'Failed to create ADK services: {e}')
    raise


# Cache based on settings identity to ensure singleton behavior
_adk_services_cache: dict[int, ADKServices] = {}


def get_adk_services(settings: Settings = Depends(get_settings)) -> ADKServices:
  """Dependency provider for ADK services as a singleton."""
  settings_id = id(settings)
  if settings_id not in _adk_services_cache:
    _adk_services_cache[settings_id] = _create_adk_services_impl(settings)
  return _adk_services_cache[settings_id]


# Use lru_cache for stateless singletons
@lru_cache()
def get_sse_event_mapper() -> SSEEventMapper:
  """Dependency provider for the SSEEventMapper as a singleton."""
  return SSEEventMapper()

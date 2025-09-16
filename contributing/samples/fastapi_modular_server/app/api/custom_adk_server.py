from contextlib import asynccontextmanager
from importlib.resources import files
import logging
from pathlib import Path
import threading
import time
from typing import Any

# Import the new modular routers
from app.api.routers.agent_router import AgentRouter
from app.config.settings import Settings
from app.core.dependencies import ADKServices
from fastapi import FastAPI
from google.adk.cli.adk_web_server import AdkWebServer
from google.adk.cli.utils.agent_change_handler import AgentChangeEventHandler
# Import agent refresh capabilities
from watchdog.observers import Observer

# Configure logging
logger = logging.getLogger(__name__)


class CustomAdkWebServer(AdkWebServer):
  """
  Enhanced ADK Web Server with modular router support and agent refresh capabilities.
  Maintains backward compatibility while adding robust agent reloading functionality.
  """

  def __init__(
      self, settings: Settings, adk_services: ADKServices, agents_dir: str
  ):
    """
    Initialize the custom ADK web server.

    Args:
        settings: Application settings
        adk_services: Container with all ADK services
        agents_dir: Directory containing agents
    """
    self.settings = settings
    self.adk_services = adk_services
    self.reload_agents = settings.reload_agents
    self.agents_root = Path(agents_dir)

    # Extract services from container for parent class
    # pass individual services as keyword arguments to the super init
    super().__init__(
        agent_loader=self.adk_services.agent_loader,
        session_service=self.adk_services.session_service,
        memory_service=self.adk_services.memory_service,
        artifact_service=self.adk_services.artifact_service,
        credential_service=self.adk_services.credential_service,
        eval_sets_manager=self.adk_services.eval_sets_manager,
        eval_set_results_manager=self.adk_services.eval_set_results_manager,
        agents_dir=str(self.agents_root),
    )

    # Modular routers
    self.agent_router: AgentRouter | None = None

    # Agent refresh components
    self.observer = None
    self.agent_change_handler = None
    self._setup_agent_refresh()

  def _setup_agent_refresh(self):
    """Initialize agent refresh capabilities if enabled."""
    if not self.reload_agents:
      logger.info("Agent refresh disabled.")
      return

    try:
      self.observer = Observer()
      self.agent_change_handler = AgentChangeEventHandler(
          agent_loader=self.adk_services.agent_loader,
          runners_to_clean=self.runners_to_clean,
          current_app_name_ref=self.current_app_name_ref,
      )
      self._start_observer()
      logger.info(f"Agent refresh enabled for root: {self.agents_root}")
    except Exception as e:
      logger.error(f"Failed to setup agent refresh: {e}", exc_info=True)
      self.reload_agents = False

  def _start_observer(self):
    """Start the file system observer for agent changes."""
    if not self.observer or not self.agent_change_handler:
      return

    try:
      if self.agents_root.exists():
        self.observer.schedule(
            self.agent_change_handler, str(self.agents_root), recursive=True
        )
        observer_thread = threading.Thread(
            target=self.observer.start, daemon=True
        )
        observer_thread.start()
        logger.info(f"Started file system observer for: {self.agents_root}")
    except Exception as e:
      logger.error(f"Failed to start observer: {e}", exc_info=True)

  def _stop_observer(self):
    """Stop the file system observer."""
    if self.observer and self.observer.is_alive():
      try:
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped file system observer")
      except Exception as e:
        logger.error(f"Error stopping observer: {e}", exc_info=True)

  def _initialize_routers(self):
    """Initialize the modular routers."""
    try:
      # Pass the web server instance, which now has proper dependency injection
      self.agent_router = AgentRouter(self)
      logger.info("AgentRouter initialized successfully.")

    except Exception as e:
      logger.error(f"Failed to initialize modular routers: {e}", exc_info=True)

  def get_enhanced_fast_api_app(self) -> FastAPI:
    """Assemble and return the enhanced FastAPI application."""
    web_assets_dir = None
    if self.settings.serve_web_interface:
      try:
        web_assets_dir = str(files("google.adk.cli.browser").joinpath(""))
      except ModuleNotFoundError:
        logger.warning(
            "Could not locate ADK web assets. UI will not be served."
        )

    # Get the FastAPI app from ADK with custom lifespan
    app = self.get_fast_api_app(
        lifespan=self._setup_custom_lifespan(),
        allow_origins=self.settings.allow_origins,
        web_assets_dir=web_assets_dir,
    )

    # Add custom routes defined directly on this server
    self.add_custom_routes(app)

    # Initialize and register our modular routers
    self._initialize_routers()
    self._register_modular_routers(app)

    return app

  def add_custom_routes(self, app: FastAPI):
    """Add server-specific, non-modular routes to the app."""

    @app.get("/diagnostic", tags=["Diagnostics"])
    async def diagnostic():
      """Provides diagnostic information about the server setup."""
      agent_loader_status = (
          "initialized" if self.adk_services.agent_loader else "not_initialized"
      )

      return {
          "status": "success",
          "message": "Server components are active.",
          "agent_loader_status": agent_loader_status,
          "settings": {
              "agent_dir": str(self.settings.agent_parent_dir),
              "reload_agents": self.settings.reload_agents,
              "debug": self.settings.debug,
              "app_name": self.settings.app_name,
          },
          "services": {
              "session_service": type(
                  self.adk_services.session_service
              ).__name__,
              "memory_service": type(self.adk_services.memory_service).__name__,
              "artifact_service": type(
                  self.adk_services.artifact_service
              ).__name__,
          },
      }

    @app.get("/health", tags=["Health"])
    async def health_check():
      """Health check endpoint."""
      try:
        # Perform basic health checks
        checks = {
            "agent_loader": self.adk_services.agent_loader is not None,
            "session_service": self.adk_services.session_service is not None,
            "memory_service": self.adk_services.memory_service is not None,
        }

        all_healthy = all(checks.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "timestamp": time.time(),
        }
      except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time(),
        }

    @app.get("/", include_in_schema=False)
    async def root():
      """Root endpoint with API information."""
      return {
          "message": "ADK Custom FastAPI server is running",
          "app_name": self.settings.app_name,
          "docs_url": "/docs",
          "diagnostic_url": "/diagnostic",
          "health_url": "/health",
      }

  def _register_modular_routers(self, app: FastAPI):
    """Register the main modular routers, overriding ADK defaults."""
    try:
      routes_to_remove = []
      for route in app.routes:
        # Identify original ADK routes by path and methods
        if route.path in [
            "/run_sse",
        ] and hasattr(route, "methods"):
          # Check if it's a POST method route
          if "POST" in route.methods:
            routes_to_remove.append(route)

      for route in routes_to_remove:
        app.routes.remove(route)

      if routes_to_remove:
        logger.info(
            f"Successfully removed {len(routes_to_remove)} original ADK routes"
            " for override."
        )

      if self.agent_router:
        app.include_router(
            self.agent_router.get_router(),
        )
        logger.info(
            "Registered AgentRouter, overriding default agent endpoints."
        )

    except Exception as e:
      logger.error(f"Failed to register modular routers: {e}", exc_info=True)

  def _setup_custom_lifespan(self):
    """Setup custom lifespan events for startup and shutdown."""

    @asynccontextmanager
    async def custom_lifespan(app: FastAPI):
      logger.info("Server startup sequence initiated...")

      # Startup logic
      try:
        # Validate services are properly initialized
        if not self.adk_services.agent_loader:
          logger.warning("Agent loader not properly initialized")

        # Log configuration
        logger.info(
            f"Server configured with settings: {self.settings.app_name}"
        )
        logger.info(f"Agent directory: {self.agents_root}")
        logger.info(f"Debug mode: {self.settings.debug}")

      except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)

      yield

      # Shutdown logic
      logger.info("Server shutdown sequence initiated...")
      try:
        self._stop_observer()

        # Additional cleanup can go here
        logger.info("Server shutdown completed successfully")

      except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

    return custom_lifespan

  def get_service(self, service_name: str) -> Any:
    """
    Get a specific service from the ADK services container.

    Args:
        service_name: Name of the service to retrieve

    Returns:
        The requested service

    Raises:
        AttributeError: If the service doesn't exist
    """
    return getattr(self.adk_services, service_name)

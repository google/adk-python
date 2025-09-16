# main.py - Improved version
import logging

from app.api.custom_adk_server import CustomAdkWebServer
from app.config.settings import settings
from app.core.dependencies import get_adk_services
from app.core.logging import setup_logging
from fastapi import FastAPI
import uvicorn

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
  """
  Application Factory.

  Creates and configures the main FastAPI application instance by orchestrating
  the setup of logging, services, and the custom web server.

  Returns:
      The fully configured FastAPI application instance.
  """
  # 1. Setup logging as the very first step.
  setup_logging(settings)

  logger.info(
      f"Starting application factory for '{settings.app_name}' "
      f"in {'DEBUG' if settings.debug else 'PRODUCTION'} mode."
  )

  try:
    # 2. Create the foundational ADK services.
    logger.debug("Initializing core ADK services...")
    adk_services = get_adk_services(settings)

    # 3. Create an instance of your custom web server.
    logger.debug("Creating CustomAdkWebServer instance...")
    custom_server = CustomAdkWebServer(
        settings=settings,
        adk_services=adk_services,  # Pass the services container
        agents_dir=str(settings.agent_parent_dir),
    )

    # 4. Get the final, fully configured FastAPI app
    logger.debug(
        "Assembling the enhanced FastAPI app from the custom server..."
    )
    app = custom_server.get_enhanced_fast_api_app()

    logger.info("FastAPI application created and configured successfully.")
    return app

  except Exception as e:
    logger.critical(
        f"FATAL: Failed to create FastAPI application: {e}", exc_info=True
    )
    raise


# Create the global 'app' instance by calling the factory.
app = create_app()


if __name__ == "__main__":
  print("--- Starting ADK FastAPI Server for Development ---")
  print(f"Host: http://{settings.host}:{settings.port}")
  print(f"API Docs: http://{settings.host}:{settings.port}/docs")
  if settings.serve_web_interface:
    print(f"Web UI: http://{settings.host}:{settings.port}/ui")
  print(f"Reload on changes: {settings.debug}")
  print("----------------------------------------------------")

  uvicorn.run(
      "main:app",
      host=settings.host,
      port=settings.port,
      reload=settings.debug,
      log_level=settings.log_level.lower(),
  )

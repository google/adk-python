import logging
import sys


def setup_logging(settings) -> None:
  """Configure application logging."""
  logging.basicConfig(
      level=getattr(logging, settings.log_level.upper()),
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      handlers=[
          logging.StreamHandler(sys.stdout),
          logging.FileHandler("app.log")
          if not settings.debug
          else logging.NullHandler(),
      ],
  )

  # Configure specific loggers
  logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
  logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

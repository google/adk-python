from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
  """Application settings using Pydantic for validation and environment variable support."""

  # Application
  app_name: str = "ADK Agent FastAPI Server"
  debug: bool = Field(default=False, description="Enable debug mode")
  port: int = Field(default=8881, description="Server port")
  host: str = Field(default="localhost", description="Server host")

  # Paths
  current_dir: Path = Field(
      default_factory=lambda: Path(__file__).parent.parent.absolute()
  )
  agent_parent_dir: Optional[Path] = None
  artifact_root_path: Optional[Path] = None

  # Database
  session_db_url: Optional[str] = None

  # ADK Configuration
  serve_web_interface: bool = Field(
      default=True, description="Serve web interface"
  )
  reload_agents: bool = Field(
      default=True, description="Enable agent hot-reload"
  )

  # CORS
  allow_origins: list[str] = Field(
      default=["*"], description="CORS allowed origins"
  )

  # Logging
  log_level: str = Field(default="INFO", description="Logging level")

  model_config = {
      "env_file": ".env",
      "env_file_encoding": "utf-8",
      "case_sensitive": False,
      "extra": "allow",  # Allows extra fields and makes them accessible
  }

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Set computed defaults after initialization
    if self.agent_parent_dir is None:
      self.agent_parent_dir = self.current_dir / "agents"


# Global settings instance
settings = Settings()

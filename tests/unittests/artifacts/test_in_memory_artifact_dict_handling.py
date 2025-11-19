import sys
import types as _types

# Prefer the real `google.genai.types` if installed; fall back to a minimal
# local stub only when the real package is not available. This avoids
# overriding the real `google` package and ensures type hints used across
# the ADK package are satisfied when the dependency is present.
try:
  from google.genai import types as types_mod  # type: ignore
except Exception:
  genai = _types.ModuleType("google.genai")
  types_mod = _types.ModuleType("google.genai.types")

  # Minimal Part and Content classes used in tests when google-genai
  # is not installed. These are intentionally small and only provide
  # the attributes the InMemoryArtifactService touches.
  class Part:
    def __init__(self, inline_data=None, text=None, file_data=None):
      self.inline_data = inline_data
      self.text = text
      self.file_data = file_data

    def __eq__(self, other):
      if not isinstance(other, Part):
        return False
      return (
        (self.inline_data == other.inline_data)
        and (self.text == other.text)
        and (self.file_data == other.file_data)
      )

  class Content:
    pass

  types_mod.Part = Part
  types_mod.Content = Content
  setattr(genai, "types", types_mod)
  sys.modules["google.genai"] = genai
  sys.modules["google.genai.types"] = types_mod

import importlib.util
import importlib.machinery
import types as _types
import os

# Create lightweight stubs for package modules to avoid importing the
# whole ADK package during this focused unit test. We populate
# `sys.modules` entries so relative imports inside the target module
# resolve against these stubs.
if "google.adk" not in sys.modules:
  sys.modules.setdefault("google", _types.ModuleType("google"))
  sys.modules.setdefault("google.adk", _types.ModuleType("google.adk"))
  sys.modules.setdefault("google.adk.artifacts", _types.ModuleType("google.adk.artifacts"))

# Stub for base_artifact_service used by the in-memory service.
base_mod = _types.ModuleType("google.adk.artifacts.base_artifact_service")
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class ArtifactVersion:
  version: int
  canonical_uri: str
  mime_type: Optional[str] = None
  custom_metadata: Optional[dict[str, Any]] = None

class BaseArtifactService:
  pass

base_mod.ArtifactVersion = ArtifactVersion
base_mod.BaseArtifactService = BaseArtifactService
sys.modules["google.adk.artifacts.base_artifact_service"] = base_mod

# Minimal artifact_util stub
artifact_util_mod = _types.ModuleType("google.adk.artifacts.artifact_util")
def parse_artifact_uri(uri: str):
  return None
def is_artifact_ref(artifact):
  if isinstance(artifact, dict):
    fd = artifact.get("file_data")
    if not fd:
      return False
    uri = fd.get("file_uri")
    return isinstance(uri, str) and uri.startswith("artifact://")
  # object-like: attempt attribute access
  fd = getattr(artifact, "file_data", None)
  if not fd:
    return False
  uri = getattr(fd, "file_uri", None)
  return isinstance(uri, str) and uri.startswith("artifact://")

artifact_util_mod.parse_artifact_uri = parse_artifact_uri
artifact_util_mod.is_artifact_ref = is_artifact_ref
sys.modules["google.adk.artifacts.artifact_util"] = artifact_util_mod

# Stub google.genai.types if not available
try:
  from google.genai import types as types_mod  # type: ignore
except Exception:
  genai = _types.ModuleType("google.genai")
  types_mod = _types.ModuleType("google.genai.types")
  class Part:
    def __init__(self, inline_data=None, text=None, file_data=None):
      self.inline_data = inline_data
      self.text = text
      self.file_data = file_data

    def __eq__(self, other):
      if not isinstance(other, Part):
        return False
      return (
        (self.inline_data == other.inline_data)
        and (self.text == other.text)
        and (self.file_data == other.file_data)
      )
  class Content:
    pass
  types_mod.Part = Part
  types_mod.Content = Content
  sys.modules["google.genai"] = genai
  sys.modules["google.genai.types"] = types_mod

# Load the in-memory artifact service module directly from source
module_path = os.path.join(os.getcwd(), "src", "google", "adk", "artifacts", "in_memory_artifact_service.py")
spec = importlib.util.spec_from_file_location("google.adk.artifacts.in_memory_artifact_service", module_path)
in_memory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = in_memory
spec.loader.exec_module(in_memory)
InMemoryArtifactService = in_memory.InMemoryArtifactService

import pytest


@pytest.mark.asyncio
async def test_save_artifact_with_dict_and_part():
  svc = InMemoryArtifactService()

  # Case 1: artifact passed as plain dict (simulating AgentSpace upload)
  dict_artifact = {
    "file_data": {"file_uri": "memory://apps/a/u/s/f/versions/0", "mime_type": "text/plain"}
  }
  ver = await svc.save_artifact(
    app_name="a",
    user_id="u",
    filename="s/f",
    session_id="session",
    artifact=dict_artifact,
  )
  assert ver == 0

  # Case 2: artifact passed as object-like Part
  part = types_mod.Part(text="hello")
  ver2 = await svc.save_artifact(
    app_name="a",
    user_id="u",
    filename="s/f",
    session_id="session",
    artifact=part,
  )
  assert ver2 == 1

  # Ensure load returns the same object-like Part for the last saved
  loaded = await svc.load_artifact(app_name="a", user_id="u", filename="s/f", session_id="session")
  assert isinstance(loaded, types_mod.Part)


@pytest.mark.asyncio
async def test_save_artifact_with_inline_dict():
  svc = InMemoryArtifactService()
  inline = {"inline_data": {"mime_type": "image/png", "data": b"\x89PNG"}}
  ver = await svc.save_artifact(
    app_name="app",
    user_id="user",
    filename="user:avatar.png",
    artifact=inline,
  )
  assert ver == 0

  # list keys should include the user-scoped filename
  keys = await svc.list_artifact_keys(app_name="app", user_id="user")
  assert "user:avatar.png" in keys

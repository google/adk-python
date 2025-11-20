import os
import sys
import types as _types
import importlib.util

# Reuse the same focused loader pattern as the other artifact tests.
try:
  from google.genai import types as types_mod  # type: ignore
except Exception:
  types_mod = _types.ModuleType("google.genai.types")
  class Part:
    def __init__(self, inline_data=None, text=None, file_data=None):
      self.inline_data = inline_data
      self.text = text
      self.file_data = file_data
    def __eq__(self, other):
      if not isinstance(other, Part):
        return False
      return (self.inline_data == other.inline_data and self.text == other.text and self.file_data == other.file_data)
  types_mod.Part = Part
  sys.modules.setdefault("google.genai", _types.ModuleType("google.genai"))
  sys.modules["google.genai.types"] = types_mod

# Ensure artifact_util and base stub exist
sys.modules.setdefault("google", _types.ModuleType("google"))
sys.modules.setdefault("google.adk", _types.ModuleType("google.adk"))
sys.modules.setdefault("google.adk.artifacts", _types.ModuleType("google.adk.artifacts"))

if "google.adk.artifacts.base_artifact_service" not in sys.modules:
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

artifact_util_path = os.path.join(os.getcwd(), "src", "google", "adk", "artifacts", "artifact_util.py")
spec_util = importlib.util.spec_from_file_location("google.adk.artifacts.artifact_util", artifact_util_path)
artifact_util = importlib.util.module_from_spec(spec_util)
sys.modules[spec_util.name] = artifact_util
spec_util.loader.exec_module(artifact_util)

# Load module directly
module_path = os.path.join(os.getcwd(), "src", "google", "adk", "artifacts", "in_memory_artifact_service.py")
spec = importlib.util.spec_from_file_location("google.adk.artifacts.in_memory_artifact_service", module_path)
in_memory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = in_memory
spec.loader.exec_module(in_memory)
InMemoryArtifactService = in_memory.InMemoryArtifactService
def make_artifact_uri(app_name: str, user_id: str, filename: str, version: int, session_id: str | None = None) -> str:
  if session_id:
    return f"artifact://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{version}"
  return f"artifact://apps/{app_name}/users/{user_id}/artifacts/{filename}/versions/{version}"

import pytest


@pytest.mark.asyncio
async def test_artifact_ref_resolution():
  svc = InMemoryArtifactService()

  # Save a real target artifact first
  target = types_mod.Part(text="the content")
  v = await svc.save_artifact(app_name="a", user_id="u", filename="target.txt", session_id="s", artifact=target)
  assert v == 0

  # Create an artifact that references the stored artifact
  ref_uri = make_artifact_uri("a", "u", "target.txt", 0, session_id="s")
  ref_artifact = {"file_data": {"file_uri": ref_uri}}

  vr = await svc.save_artifact(app_name="a", user_id="u", filename="ref.txt", session_id="s", artifact=ref_artifact)
  # This is the first save for `ref.txt`, so version should be 0.
  assert vr == 0

  loaded = await svc.load_artifact(app_name="a", user_id="u", filename="ref.txt", session_id="s")
  # loading the ref should resolve to the original target artifact
  assert isinstance(loaded, types_mod.Part)
  assert loaded.text == "the content"


@pytest.mark.asyncio
async def test_empty_artifact_returns_none():
  svc = InMemoryArtifactService()
  # Save an empty Part object (should be considered empty)
  empty = types_mod.Part()
  await svc.save_artifact(app_name="x", user_id="y", filename="file", session_id="sess", artifact=empty)
  loaded = await svc.load_artifact(app_name="x", user_id="y", filename="file", session_id="sess")
  assert loaded is None

import sys
import types as _types
import importlib.util
import os
import logging

# Load module in the focused way to avoid importing whole ADK package.
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

sys.modules.setdefault("google", _types.ModuleType("google"))
sys.modules.setdefault("google.adk", _types.ModuleType("google.adk"))
sys.modules.setdefault("google.adk.artifacts", _types.ModuleType("google.adk.artifacts"))

module_path = os.path.join(os.getcwd(), "src", "google", "adk", "artifacts", "in_memory_artifact_service.py")
spec = importlib.util.spec_from_file_location("google.adk.artifacts.in_memory_artifact_service", module_path)
in_memory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = in_memory
spec.loader.exec_module(in_memory)
InMemoryArtifactService = in_memory.InMemoryArtifactService

import pytest


@pytest.mark.asyncio
async def test_fallback_mime_logged_and_set(caplog):
  svc = InMemoryArtifactService()
  # Artifact with no inline/text/file fields -> unknown shape
  weird = {"weird": True}

  caplog.set_level(logging.DEBUG)
  ver = await svc.save_artifact(app_name="app", user_id="u", filename="f.bin", session_id="s", artifact=weird)
  assert ver == 0

  # Verify artifact version metadata reports fallback mime type
  av = await svc.get_artifact_version(app_name="app", user_id="u", filename="f.bin", session_id="s")
  assert av is not None
  assert av.mime_type == "application/octet-stream"

  # Confirm debug log emitted about fallback
  found = any("falling back to application/octet-stream" in rec.message for rec in caplog.records)
  assert found, "Expected debug log about fallback mime type"

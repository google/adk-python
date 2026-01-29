from google.adk.models.lite_llm import _part_has_payload
from google.genai import types
import pytest


def test_part_has_payload_with_function_response():
  part = types.Part.from_function_response(
      name="test_fn", response={"result": "success"}
  )
  assert _part_has_payload(part) is True


def test_part_has_payload_without_payload():
  part = types.Part()
  assert _part_has_payload(part) is False

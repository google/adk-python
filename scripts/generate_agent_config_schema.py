# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to generate AgentConfig.json from AgentConfig."""

from __future__ import annotations

import json
import os
from typing import Any
from typing import TYPE_CHECKING

from google.adk.agents.agent_config import AgentConfig
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import JsonSchemaMode
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import override

if TYPE_CHECKING:
  from pydantic._internal._core_utils import CoreSchemaOrField

_BUILTIN_AGENT_CLASS_ENUMS: dict[str, list[str]] = {
    "LlmAgentConfig": [
        "LlmAgent",
        "google.adk.agents.LlmAgent",
        "google.adk.agents.llm_agent.LlmAgent",
        "Agent",
        "google.adk.agents.Agent",
        "google.adk.agents.llm_agent.Agent",
        "",
    ],
    "LoopAgentConfig": [
        "LoopAgent",
        "google.adk.agents.LoopAgent",
        "google.adk.agents.loop_agent.LoopAgent",
    ],
    "ParallelAgentConfig": [
        "ParallelAgent",
        "google.adk.agents.ParallelAgent",
        "google.adk.agents.parallel_agent.ParallelAgent",
    ],
    "SequentialAgentConfig": [
        "SequentialAgent",
        "google.adk.agents.SequentialAgent",
        "google.adk.agents.sequential_agent.SequentialAgent",
    ],
}


class _FallbackGenerateJsonSchema(GenerateJsonSchema):
  """Base schema generator that handles invalid types by falling back."""

  @override
  def handle_invalid_for_json_schema(
      self, schema: CoreSchemaOrField, error_info: str
  ) -> JsonSchemaValue:
    try:
      return super().handle_invalid_for_json_schema(schema, error_info)
    except PydanticInvalidForJsonSchema:
      # Return a fallback schema instead of failing
      return {
          "type": "object",
          "description": f"Fallback for invalid schema: {error_info}",
      }


class CustomGenerateJsonSchema(_FallbackGenerateJsonSchema):
  """Custom schema generator that aligns AgentConfig JSON schema with runtime."""

  @override
  def generate(
      self, schema: CoreSchemaOrField, mode: JsonSchemaMode = "validation"
  ) -> JsonSchemaValue:
    # Generate snake_case (field names) and camelCase (aliases) schemas and
    # merge their properties so models with populate_by_name/validate_by_name
    # (such as google.genai.types.GenerateContentConfig) accept both spellings.
    snake_gen = _FallbackGenerateJsonSchema(
        by_alias=False, ref_template=self.ref_template
    )
    camel_gen = _FallbackGenerateJsonSchema(
        by_alias=True, ref_template=self.ref_template
    )
    json_schema = snake_gen.generate(schema, mode=mode)
    camel_schema = camel_gen.generate(schema, mode=mode)

    defs: dict[str, Any] = json_schema.get("$defs", {})
    camel_defs: dict[str, Any] = camel_schema.get("$defs", {})
    for def_name, def_schema in defs.items():
      camel_def = camel_defs.get(def_name, {})
      if "properties" in def_schema and "properties" in camel_def:
        for prop_name, prop_schema in camel_def["properties"].items():
          if prop_name not in def_schema["properties"]:
            def_schema["properties"][prop_name] = prop_schema

    # Make the top-level oneOf branches mutually exclusive by constraining
    # `agent_class` on each branch to match runtime discrimination.
    all_builtin_names = [
        name
        for names in _BUILTIN_AGENT_CLASS_ENUMS.values()
        for name in names
        if name
    ]
    for def_name, allowed_names in _BUILTIN_AGENT_CLASS_ENUMS.items():
      if def_name not in defs:
        continue
      def_schema = defs[def_name]
      agent_class_prop = dict(
          def_schema.get("properties", {}).get("agent_class", {})
      )
      agent_class_prop["type"] = "string"
      agent_class_prop["enum"] = allowed_names
      def_schema.setdefault("properties", {})["agent_class"] = agent_class_prop
      if def_name != "LlmAgentConfig":
        required = list(def_schema.get("required", []))
        if "agent_class" not in required:
          required.append("agent_class")
        def_schema["required"] = required

    if "BaseAgentConfig" in defs:
      base_def = defs["BaseAgentConfig"]
      agent_class_prop = dict(
          base_def.get("properties", {}).get("agent_class", {})
      )
      agent_class_prop.pop("default", None)
      agent_class_prop["type"] = "string"
      agent_class_prop["minLength"] = 1
      agent_class_prop["not"] = {"enum": all_builtin_names}
      base_def.setdefault("properties", {})["agent_class"] = agent_class_prop
      required = list(base_def.get("required", []))
      if "agent_class" not in required:
        required.append("agent_class")
      base_def["required"] = required

    return json_schema


def main() -> None:
  """Generates the AgentConfig.json schema."""
  # Use the custom generator to avoid failing on httpx.Client
  schema = AgentConfig.model_json_schema(
      schema_generator=CustomGenerateJsonSchema
  )

  # Find the repo root relative to this file.
  script_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.dirname(script_dir)

  output_path = os.path.join(
      repo_root, "src/google/adk/agents/config_schemas/AgentConfig.json"
  )

  # Ensure directory exists
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
    f.write("\n")

  print(f"Successfully generated {output_path}")


if __name__ == "__main__":
  main()

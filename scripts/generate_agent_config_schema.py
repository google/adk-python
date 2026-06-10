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

from google.adk.agents.agent_config import AgentConfig
from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import PydanticInvalidForJsonSchema


class CustomGenerateJsonSchema(GenerateJsonSchema):
  """Custom schema generator that handles invalid types by falling back."""

  def handle_invalid_for_json_schema(self, schema, error_info):
    try:
      return super().handle_invalid_for_json_schema(schema, error_info)
    except PydanticInvalidForJsonSchema:
      # Return a fallback schema instead of failing
      return {
          "type": "object",
          "description": f"Fallback for invalid schema: {error_info}",
      }


def main():
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

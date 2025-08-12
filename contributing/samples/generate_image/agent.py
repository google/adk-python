# Copyright 2025 Google LLC
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

from google.adk import Agent
from google.adk.tools import load_artifacts
from google.adk.tools.tool_context import ToolContext
from google.genai import Client
from google.genai import types

# Only Vertex AI supports image generation for now.
client = Client()


async def generate_image(prompt: str, tool_context: 'ToolContext'):
  """Generates an image based on the prompt."""
  response = client.models.generate_images(
      model='imagen-3.0-generate-002',
      prompt=prompt,
      config={'number_of_images': 1},
  )
  if not response.generated_images:
    return {'status': 'failed'}
  image_bytes = response.generated_images[0].image.image_bytes
  await tool_context.save_artifact(
      'image.png',
      types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
  )
  return {
      'status': 'success',
      'detail': 'Image generated successfully and stored in artifacts.',
      'filename': 'image.png',
  }


root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='root_agent',
    description="""An agent that generates images and answer questions about the images.""",
    instruction="""You are an agent whose job is to generate or edit an image based on the user's prompt.
""",
    tools=[generate_image, load_artifacts],
)

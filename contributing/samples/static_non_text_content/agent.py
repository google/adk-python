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

"""Static non-text content sample agent demonstrating static instructions with non-text parts."""

import base64

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Sample image data (a simple 1x1 yellow pixel PNG)
SAMPLE_IMAGE_DATA = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)

# Sample document content (simplified contributing guide)
SAMPLE_DOCUMENT = """# Contributing Guide

## Best Practices

1. **Code Quality**: Always write clean, well-documented code
2. **Testing**: Include comprehensive tests for new features
3. **Documentation**: Update documentation when adding new functionality
4. **Review Process**: Submit pull requests for code review
5. **Conventions**: Follow established coding conventions and style guides

## Guidelines

- Use meaningful variable and function names
- Write descriptive commit messages
- Keep functions small and focused
- Handle errors gracefully
- Consider performance implications
- Maintain backward compatibility when possible

This guide helps ensure consistent, high-quality contributions to the project.
"""


def create_static_instruction_with_file_upload():
  """Create static instruction content with both inline_data and file_data.

  This function creates a static instruction that demonstrates both inline_data
  (for images) and file_data (for documents). Always includes Files API upload,
  and adds additional GCS file reference when using Vertex AI.
  """
  import os
  import tempfile

  from google.adk.utils.variant_utils import get_google_llm_variant
  from google.adk.utils.variant_utils import GoogleLLMVariant

  from google import genai

  # Determine API variant
  api_variant = get_google_llm_variant()
  print(f"Using API variant: {api_variant}")

  # Prepare file data parts based on API variant
  file_data_parts = []

  if api_variant == GoogleLLMVariant.VERTEX_AI:
    print("Using Vertex AI - adding GCS URI and HTTPS URL references")

    # Add GCS file reference
    file_data_parts.append(
        types.Part(
            file_data=types.FileData(
                file_uri=(
                    "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
                ),
                mime_type="application/pdf",
                display_name="Gemma Research Paper",
            )
        )
    )

    # Add the same document via HTTPS URL to demonstrate both access methods
    file_data_parts.append(
        types.Part(
            file_data=types.FileData(
                file_uri="https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
                mime_type="application/pdf",
                display_name="AI Research Paper (HTTPS)",
            )
        )
    )

    additional_text = (
        " You also have access to a Gemma research paper from GCS"
        " and an AI research paper from HTTPS URL."
    )

  else:
    print("Using Gemini Developer API - uploading to Files API")
    client = genai.Client()

    # Check if file already exists
    display_name = "Contributing Guide"
    uploaded_file = None

    # List existing files to see if we already uploaded this document
    existing_files = client.files.list()
    for file in existing_files:
      if file.display_name == display_name:
        uploaded_file = file
        print(f"Reusing existing file: {file.name} ({file.display_name})")
        break

    # If file doesn't exist, upload it
    if uploaded_file is None:
      # Create a temporary file with the sample document
      with tempfile.NamedTemporaryFile(
          mode="w", suffix=".md", delete=False
      ) as f:
        f.write(SAMPLE_DOCUMENT)
        temp_file_path = f.name

      try:
        # Upload the file to Gemini Files API
        uploaded_file = client.files.upload(file=temp_file_path)
        print(
            "Uploaded new file:"
            f" {uploaded_file.name} ({uploaded_file.display_name})"
        )
      finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
          os.unlink(temp_file_path)

    # Add Files API file data part
    file_data_parts.append(
        types.Part(
            file_data=types.FileData(
                file_uri=uploaded_file.uri,
                mime_type="text/markdown",
                display_name="Contributing Guide",
            )
        )
    )

    additional_text = (
        " You also have access to the contributing guide document."
    )

  # Create static instruction with mixed content
  parts = [
      types.Part.from_text(
          text=(
              "You are an AI assistant that analyzes images and documents."
              " You have access to the following reference materials:"
          )
      ),
      # Add a sample image as inline_data
      types.Part(
          inline_data=types.Blob(
              data=SAMPLE_IMAGE_DATA,
              mime_type="image/png",
              display_name="sample_chart.png",
          )
      ),
      types.Part.from_text(
          text=f"This is a sample chart showing color data.{additional_text}"
      ),
  ]

  # Add all file_data parts
  parts.extend(file_data_parts)

  # Add instruction text
  if api_variant == GoogleLLMVariant.VERTEX_AI:
    instruction_text = """
When users ask questions, you should:
1. Use the reference chart above to provide context when discussing visual data or charts
2. Reference the Gemma research paper (from GCS) when discussing AI research, model architectures, or technical details
3. Reference the AI research paper (from HTTPS) when discussing research topics
4. Be helpful and informative in your responses
5. Explain how the provided reference materials relate to their questions"""
  else:
    instruction_text = """
When users ask questions, you should:
1. Use the reference chart above to provide context when discussing visual data or charts
2. Reference the contributing guide document when explaining best practices and guidelines
3. Be helpful and informative in your responses
4. Explain how the provided reference materials relate to their questions"""

  instruction_text += """

Remember: The reference materials above are available to help you provide better answers."""

  parts.append(types.Part.from_text(text=instruction_text))

  static_instruction_content = types.Content(parts=parts)

  return static_instruction_content


# Create the root agent with Files API integration
root_agent = Agent(
    model="gemini-2.5-flash",
    name="static_non_text_content_demo_agent",
    description=(
        "Demonstrates static instructions with non-text content (inline_data"
        " and file_data features)"
    ),
    static_instruction=create_static_instruction_with_file_upload(),
    instruction=(
        "Please analyze the user's question and provide helpful insights."
        " Reference the materials provided in your static instructions when"
        " relevant."
    ),
)

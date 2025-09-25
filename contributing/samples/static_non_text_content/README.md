# Static Non-Text Content Sample Agent

This sample demonstrates ADK's static instruction feature with non-text content (images and files).

## Features Demonstrated

- **Static instructions with mixed content**: Text, images, and file references in a single static instruction
- **Reference ID generation**: Non-text parts are automatically given reference IDs (`inline_data_0`, `file_data_1`, etc.)
- **Gemini Files API integration**: Demonstrates uploading documents and using file_data
- **Mixed content types**: inline_data for images, file_data for documents
- **API variant detection**: Different behavior for Gemini API vs Vertex AI
- **GCS file references**: Additional GCS file support when using Vertex AI

## Static Instruction Content

The agent includes:

1. **Text instructions**: Guide the agent on how to behave
2. **Sample image**: A 1x1 yellow pixel PNG (`sample_chart.png`) as inline binary data

**Gemini Developer API:**
3. **Contributing guide**: A sample document uploaded to Gemini Files API and referenced via file_data

**Vertex AI:**
3. **Research paper**: Gemma research paper from Google Cloud Storage via GCS file reference
4. **Contributing guide**: Gemini Cookbook contributing guide from GitHub via HTTPS file reference

## Content Used

**All API variants:**
- **Image**: Base64-encoded 1x1 yellow pixel PNG (embedded in code as `inline_data`)

**Gemini Developer API:**
- **Document**: Sample contributing guide text (uploaded to Gemini Files API as `file_data`)
  - Contains sample guidelines and best practices for development
  - Demonstrates Files API upload and file_data reference functionality
  - Files are automatically cleaned up after 48 hours by the Gemini API

**Vertex AI:**
- **Research Paper**: Gemma research paper (GCS file reference as `file_data`)
  - Public GCS URI: `gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf`
  - Demonstrates GCS file access in Vertex AI
  - PDF format with technical AI research content
- **Contributing Guide**: Gemini Cookbook contributing guide (HTTPS file reference as `file_data`)
  - Public GitHub URL: `https://raw.githubusercontent.com/google-gemini/cookbook/main/CONTRIBUTING.md`
  - Demonstrates HTTPS file access in Vertex AI
  - Markdown format with development guidelines

## Setup

### Setup API Credentials

Create a `.env` file in the project root with your API credentials:

```bash
# Choose Model Backend: 0 -> ML Dev, 1 -> Vertex
GOOGLE_GENAI_USE_VERTEXAI=1

# ML Dev backend config
GOOGLE_API_KEY=your_google_api_key_here

# Vertex backend config
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

The agent will automatically load environment variables on startup.

## Usage

### Default Test Prompts (Recommended)
```bash
cd contributing/samples
python -m static_non_text_content.main
```
This runs 4 test prompts that specifically demonstrate the static content features.

### Interactive Mode  
```bash
cd contributing/samples
adk run static_non_text_content
```
Use ADK's built-in interactive mode for free-form conversation.

### Single Prompt
```bash
cd contributing/samples
python -m static_non_text_content.main --prompt "What reference materials do you have access to?"
```

### With Debug Logging
```bash
cd contributing/samples
python -m static_non_text_content.main --debug --prompt "What is the Gemma research paper about?"
```

## Default Test Prompts

The sample automatically runs test prompts when no `--prompt` is specified:

**All API variants:**
1. "What reference materials do you have access to?"
2. "Can you describe the sample chart that was provided to you?"
3. "What does the contributing guide document say about best practices?"
4. "How do the inline image and file references in your instructions help you answer questions?"

**Vertex AI only (additional prompt):**
5. "What is the Gemma research paper about and what are its key contributions?"

These prompts test `inline_data`, Files API `file_data` (Gemini API), and GCS/HTTPS `file_data` (Vertex AI).

## How It Works

1. **Static Instruction Processing**: The `static_instruction` content is processed during agent initialization
2. **Reference Generation**: Non-text parts get references like `[Reference to inline binary data: inline_data_0 ('sample_chart.png', type: image/png)]` in the system instruction
3. **User Content Creation**: The actual binary data/file references are moved to user contents with proper role attribution
4. **Model Understanding**: The model receives both the descriptive references and the actual content for analysis

## Code Structure

- `agent.py`: Defines the agent with static instruction containing mixed content
- `main.py`: Runnable script with interactive and single-prompt modes
- `__init__.py`: Package initialization following ADK conventions

This sample serves as a test case for the static instruction with non-text parts feature using both `inline_data` and `file_data`.
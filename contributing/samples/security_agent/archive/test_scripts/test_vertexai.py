#!/usr/bin/env python3
"""Test Vertex AI authentication"""

import os
from pathlib import Path

# Get credentials from environment
creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if creds_path and Path(creds_path).exists():
    print(f"✅ Credentials file found: {creds_path}")
else:
    print(f"❌ GOOGLE_APPLICATION_CREDENTIALS not set or file not found")

# Get project and location from environment
project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')

if not project_id:
    print("❌ GOOGLE_CLOUD_PROJECT environment variable not set")
    exit(1)

print(f"Using project: {project_id}, location: {location}")

# Test Vertex AI initialization
try:
    import vertexai
    vertexai.init(project=project_id, location=location)
    print("✅ Vertex AI initialized successfully")
except Exception as e:
    print(f"❌ Vertex AI initialization failed: {e}")

# Test with Google GenAI client
try:
    from google import genai
    client = genai.Client(
        vertexai=True,
        project=project_id, 
        location=location
    )
    print("✅ Google GenAI client created successfully")
except Exception as e:
    print(f"❌ Google GenAI client failed: {e}")

# Test with ADK Agent
try:
    from google.adk import Agent
    test_agent = Agent(
        name="test",
        model="gemini-2.0-flash-exp",
        vertexai=True,
        project=project_id,
        location=location,
        instruction="You are a test agent"
    )
    print("✅ ADK Agent created successfully")
except Exception as e:
    print(f"❌ ADK Agent creation failed: {e}")
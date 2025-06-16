# Configuration settings for the application

# Vertex AI / Gemini Model Configuration
GOOGLE_CLOUD_PROJECT = None # Will be loaded from .env
GOOGLE_CLOUD_LOCATION = None # Will be loaded from .env
MODEL_NAME = "gemini-2.0-flash" # Updated based on user feedback for video functionality

# Audio settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
# Add other audio settings as needed (e.g., chunk size for streaming)

MODEL_NAME = "gemini-2.0-flash"

# Video settings
VIDEO_FPS = 10 # Default FPS for video processing by VideoMonitor
# Add other video settings as needed

# Web server settings
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080

# You can add more configuration variables here as the project grows.

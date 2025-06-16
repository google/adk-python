import os
from dotenv import load_dotenv
from app.agent import MultimodalAgent
from app.config import MODEL_NAME

def main():
    # Load environment variables from .env file
    # This is important for GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"

    if not use_vertex:
        print("GOOGLE_GENAI_USE_VERTEXAI is not set to True in .env file. Exiting.")
        return

    if not project_id or not location:
        print("GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set in .env file. Exiting.")
        return

    print(f"Initializing agent in main() with project: {project_id}, location: {location}, model: {MODEL_NAME}")

    try:
        agent = MultimodalAgent(
            project_id=project_id,
            location=location,
            model_name=MODEL_NAME
        )

        # Example usage (text-only for now)
        agent.start_chat()
        response = agent.send_message(text_prompt="What is the weather like today?")
        print(f"Agent received response: {response}")

        # In a real application, this would be part of a larger loop
        # or triggered by UI events.

    except Exception as e:
        print(f"An error occurred in main: {e}")

if __name__ == "__main__":
    # This allows running main.py directly for testing,
    # assuming .env is in the parent directory of app/
    # For this to work, .env file needs to be created in `streaming-test` directory
    print("Running main.py...")
    print("Ensure .env file exists in the 'streaming-test' directory with:")
    print("GOOGLE_GENAI_USE_VERTEXAI=True")
    print("GOOGLE_CLOUD_PROJECT=<your-project-id>")
    print("GOOGLE_CLOUD_LOCATION=<your-location>")
    main()

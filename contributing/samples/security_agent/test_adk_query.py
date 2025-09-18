import asyncio
import os
import sys


# Add the project root directory to sys.path
# This ensures that 'backend' and 'agents' packages are discoverable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

# Set the database path environment variable
os.environ["DATABASE_PATH"] = "backend/cache/gcp_data.db"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/stuartgano/Desktop/Micron/IT TEAM/ADK/contributing/samples/security_agent/config/mgm-digitalconcierge-8e6bb83a7e22.json"

# Now import adk_wrapper from the backend package
from backend.adk_wrapper import ADKAgentWrapper

async def main():
    print("Testing ADK Agent with database query via FunctionTool...")

    # Example queries that should trigger the query_security_data FunctionTool
    queries = [
        "Show me high severity security findings.",
        "Give me a summary of security statistics.",
        "List all storage buckets."
    ]

    try:
        for query_text in queries:
            print(f"\nSending query: '{query_text}'")
            response = await ADKAgentWrapper.query_agent(query_text)
            print(f"\nAgent Response for '{query_text}':")
            print(response)
    finally:
        await ADKAgentWrapper.cleanup()

if __name__ == "__main__":
    # Load environment variables if .env exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed, assuming environment variables are set.")

    asyncio.run(main())

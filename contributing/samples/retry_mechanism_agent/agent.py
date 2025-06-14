import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

root_agent = LlmAgent(
    model=LiteLlm(
        model="gemini/gemini-2.0-flash",
        max_retries=5,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
        api_key=os.getenv("GOOGLE_API_KEY"),
    ),
    name="retry_test_agent",
    instruction=(
        "You are a helpful assistant for testing retry mechanisms. Please"
        " respond briefly to confirm the retry system is working."
    ),
)

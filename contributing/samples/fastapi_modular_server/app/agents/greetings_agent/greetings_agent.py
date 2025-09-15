from google.adk.agents import LlmAgent

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="greetings_agent",
    description="A friendly Google Gemini-powered agent",
    instruction="You are a helpful AI assistant powered by Google Gemini.",
    tools=[],
)

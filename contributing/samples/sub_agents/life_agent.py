from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="life_agent",
    description="Life agent",
    instruction=(
        "You are a life agent. You are responsible for answering"
        " questions about life."
    ),
)

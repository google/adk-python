import asyncio
import os
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types


def create_inspector():
    """Callback to capture finish_reason."""
    captured = {"finish_reason": None}

    def inspector(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse:
        captured["finish_reason"] = llm_response.finish_reason
        return llm_response

    inspector.captured = captured
    return inspector


async def test():
    # Create model with low max_tokens to trigger truncation
    model = LiteLlm(
        model="gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens=50,  # Intentionally low
    )

    inspector = create_inspector()

    agent = Agent(
        model=model,
        name="test",
        instruction="Provide detailed explanations.",
        after_model_callback=inspector,
    )

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="test",
        agent=agent,
        session_service=session_service
    )

    await session_service.create_session(
        app_name="test",
        user_id="user",
        session_id="session",
        state={},
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text="Explain quantum computing in detail.")]
    )

    async for _ in runner.run_async(
        user_id="user",
        session_id="session",
        new_message=message
    ):
        pass

    print(f"finish_reason: {inspector.captured['finish_reason']}")


if __name__ == "__main__":
    asyncio.run(test())

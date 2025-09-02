import warnings
import litellm
import uuid
import re
import asyncio
from agents import github_agent

from google.adk.runners import InMemoryRunner
from google.adk.cli.utils import logs
from google.adk.sessions.session import Session
from google.genai import types

warnings.filterwarnings('ignore', category=UserWarning)
logs.log_to_tmp_folder()
litellm._turn_on_debug()


async def run_prompt(new_message: str, runner: InMemoryRunner, session: Session) -> Session:
    content = types.Content(
        role='user', parts=[types.Part.from_text(text=new_message)]
    )
    print('** User says:', content.model_dump(exclude_none=True)['parts'][0]['text'])
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
    ):
      if event.content.parts and event.content.parts[0].text:
        full_text = event.content.parts[0].text
        cleaned_response = re.sub(r'<think>.*?</think>\s*', '', full_text, flags=re.S)
        # print(f'** {event.author}: {full_text}\n')
        print(f'** {event.author} (cleaned): {cleaned_response}\n')

    return session


async def main():
  app_name = 'manager_agent_application'

  runner = InMemoryRunner(
    app_name=app_name,
    agent=github_agent
  )

  session = await runner.session_service.create_session(
    app_name=app_name,
    user_id=f"user_{uuid.uuid4()}",
    session_id=f"user_session_{uuid.uuid4()}"
  )

  user_prompt = input("Enter your prompt (exit or quit to stop): ")
  while user_prompt.lower() not in ['exit', 'quit']:
      session = await run_prompt(user_prompt, runner, session)
      user_prompt = input("Enter your prompt (exit or quit to stop): ")


if __name__ == '__main__':
    asyncio.run(main())

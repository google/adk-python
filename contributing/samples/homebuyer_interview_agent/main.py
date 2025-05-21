import asyncio
import uuid
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part
from .agent import root_agent

async def amain():  # Changed to async main
    print("Starting Homebuyer Interview Agent locally (async)...")
    runner = InMemoryRunner(agent=root_agent, app_name="HomebuyerInterviewAppAsync")
    
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # Ensure session exists by calling get_session once.
    # InMemorySessionService.get_session has create_if_not_exists=True by default.
    await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    print(f"Session created/ensured. User ID: {user_id}, Session ID: {session_id}")
    
    # Initial agent message simulation - ADK agents usually respond to user input.
    # The first agent's instruction should make it greet and ask a question.
    # We'll start with an empty message from the user to trigger the first agent.
    print("\nAgent: (Starting conversation...)") # Placeholder for first agent's turn

    current_message_text = "" # Start with empty to let the first agent speak

    try:
        while True:
            agent_spoke = False
            if current_message_text is None: # Indicates user wants to quit
                print("Exiting conversation.")
                break

            user_content = Content(parts=[Part(text=current_message_text)])
            
            print("\nSending to agent...")
            
            full_agent_response = ""
            # Use run_async directly
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content,
            ):
                if event.author != 'user' and event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            # In a real CLI, print this as it arrives.
                            # For non-interactive test, we'll just see it in output.
                            # print(f"Agent ({event.author}): {part.text}")
                            full_agent_response += part.text + " "
                            agent_spoke = True
            
            if full_agent_response:
                print(f"Agent: {full_agent_response.strip()}") # Print consolidated response
            elif not agent_spoke : # if no text response, but agent might have done something
                 print("Agent did not provide a text response. The conversation might have ended or an action was taken.")


            # Heuristic end condition for this specific flow
            if "next steps" in full_agent_response.lower() or "summarize" in full_agent_response.lower() or "final questions" in full_agent_response.lower():
                 print("\nUser: (Type your response or 'quit' to exit)")
                 # In non-interactive mode, we can't use input()
                 # For testing, we'd need to feed predefined inputs.
                 # Here, it will cause EOF.
                 user_input = input("You: ") 
                 if user_input.lower() == 'quit':
                     current_message_text = None 
                 else:
                     current_message_text = user_input
                 
                 if "email is the best" in user_input.lower(): 
                     user_content_final = Content(parts=[Part(text=current_message_text)])
                     async for event in runner.run_async(
                         user_id=user_id,
                         session_id=session_id,
                         new_message=user_content_final
                     ):
                         if event.author != 'user' and event.content and event.content.parts:
                             for part in event.content.parts:
                                 if part.text:
                                     print(f"Agent ({event.author}): {part.text}")
                     print("Agent conversation concluded.")
                     break 

            if current_message_text is None: 
                break

            print("\nUser: (Type your response or 'quit' to exit)")
            # In non-interactive mode, this will cause EOF.
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                current_message_text = None
            else:
                current_message_text = user_input
                
    except EOFError:
        print("\nEOFError: End of input reached (expected in non-interactive test).")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Session ended.")

if __name__ == "__main__":
  try:
      asyncio.run(amain())
  except RuntimeError as e:
      if " asyncio.run() cannot be called from a running event loop" in str(e):
          # This can happen in some environments (like Jupyter)
          # Fallback or specific handling might be needed if this is a common issue.
          print("Note: asyncio.run() issue, this might occur in certain environments.")
      else:
          raise e

from google.genai import types
#!/usr/bin/env python3
"""
Test the fixed InMemorySessionService in the ADK to verify state persistence.
"""
import asyncio
import logging
import os
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents import SequentialAgent, SequentialAgent
from google.adk.runners import Runner

from google.adk.events.event import Event
from google.adk.agents.invocation_context import InvocationContext


# Set DEBUG environment variable for verbose logging
os.environ['DEBUG'] = '1'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SetterAgent(BaseAgent):
    """An agent that sets values in the session state."""
    
    async def _process_event(self, context: InvocationContext, event: Event) -> Event:
        # Set values in the session state
        logger.info(f"SetterAgent processing event: {event.content}")
        
        # Accessing state through context.session.state
        context.session.state["user_input"] = event.content
        context.session.state["test_key"] = "test_value"
        
        # Return response
        return Event(
            content=f"I've stored '{event.content}' as 'user_input' and 'test_value' as 'test_key'.\nState keys: {list(context.session.state.keys())}",
            agent_name="SetterAgent"
        )


class GetterAgent(BaseAgent):
    """An agent that reads values from the session state."""
    
    async def _process_event(self, context: InvocationContext, event: Event) -> Event:
        # Read values from the session state
        logger.info(f"GetterAgent processing event")
        state_keys = list(context.session.state.keys())
        
        response = f"Session state contains {len(state_keys)} keys: {state_keys}\n"
        
        if "user_input" in context.session.state:
            response += f"user_input: {context.session.state['user_input']}\n"
        else:
            response += "user_input: NOT FOUND\n"
            
        if "test_key" in context.session.state:
            response += f"test_key: {context.session.state['test_key']}\n"
        else:
            response += "test_key: NOT FOUND\n"
        
        # Return response
        return Event(
            content=response,
            agent_name="GetterAgent"
        )


class StatePersistenceTest:
    """Test application for session state persistence."""
    
    def __init__(self):
        # Set up session service
        self.session_service = InMemorySessionService(debug_mode=True)
        logger.info("Initialized InMemorySessionService")
        
        # Create agents
        setter_agent = SetterAgent(name="SetterAgent", description="Sets values in the session state")
        getter_agent = GetterAgent(name="GetterAgent", description="Gets values from the session state")
        
        # Create sequential agent
        self.agent = SequentialAgent(
            name="TestSequentialAgent",
            sub_agents=[setter_agent, getter_agent],
            description="Tests state persistence between agents"
        )
        
        # Initialize runner with sequential orchestrator
        self.runner = Runner(
            app_name="TestApp",
            agent=self.agent,
            session_service=self.session_service
        )
        logger.info("Initialized Runner with SequentialAgent")
    
    async def run_test(self):
        # Create a session
        user_id = "test_user"
        session_id = "test_session"
        
        # Create a session
        session = self.session_service.create_session(
            app_name="TestApp",
            user_id=user_id,
            session_id=session_id
        )
        logger.info(f"Created session with ID: {session_id}")
        
        # Log initial state
        state_keys = list(session.state.keys()) if session.state else []
        logger.info(f"Initial session state keys: {state_keys}")
        
        # Send a test message
        test_message = "This is a test message that should be stored in state"
        logger.info(f"Sending test message: {test_message}")
        
        print(f"\nRunning sequential agent with message: {test_message}\n")
        
        # Process the message
        event = Event(author="user", content=types.Content(parts=[types.Part(text=test_message)]))
        result = []
        async for event in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=event.content):
            result.append(event)
        
        # Log the result
        for agent_event in result:
            print(f"Agent [{agent_event.agent_name}]: {agent_event.content}\n")
        
        # Verify final state
        final_session = self.session_service.get_session(
            app_name="TestApp",
            user_id=user_id,
            session_id=session_id
        )
        final_state_keys = list(final_session.state.keys()) if final_session.state else []
        logger.info(f"Final session state keys: {final_state_keys}")
        
        return final_state_keys


async def main():
    test = StatePersistenceTest()
    state_keys = await test.run_test()
    
    # Verify the test results
    if "user_input" in state_keys and "test_key" in state_keys:
        print("✅ TEST PASSED: Session state was correctly persisted between agents")
    else:
        print("❌ TEST FAILED: Session state was not correctly persisted")
        print(f"Missing keys. Found keys: {state_keys}")


if __name__ == "__main__":
    asyncio.run(main()) 
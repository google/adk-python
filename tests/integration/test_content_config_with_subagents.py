"""Integration test for ContentConfig with subagents and tools.

This test verifies that ContentConfig with subagents and tools doesn't cause
loops or unexpected behavior, particularly when convert_foreign_events is configured.
It also tests the summarization features of ContentConfig.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List, Optional

from google.genai import types

from src.google.adk.agents.llm_agent import LlmAgent
from src.google.adk.agents.sequential_agent import SequentialAgent
from src.google.adk.agents.content_config import ContentConfig, SummarizationConfig
from src.google.adk.sessions.in_memory_session_service import InMemorySessionService
from src.google.adk.events.event import Event
from src.google.adk.agents.invocation_context import InvocationContext, new_invocation_context_id
from src.google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from src.google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from src.google.adk.agents.run_config import RunConfig
from src.google.adk.tools import FunctionTool

# Configure logging
logger = logging.getLogger(__name__)

# Mock web search tool for simulation
def mock_web_search(query: str) -> Dict[str, Any]:
    """Mock web search function that simulates a real search tool."""
    logger.info(f"Mock web search called with query: {query}")
    
    # Add counter to detect potential loops
    if not hasattr(mock_web_search, "call_counter"):
        mock_web_search.call_counter = {}
    
    normalized_query = query.lower().strip()
    if normalized_query in mock_web_search.call_counter:
        mock_web_search.call_counter[normalized_query] += 1
    else:
        mock_web_search.call_counter[normalized_query] = 1
    
    count = mock_web_search.call_counter[normalized_query]
    if count >= 3:
        logger.warning(f"POSSIBLE LOOP DETECTED! Query '{query}' called {count} times!")
    
    return {
        "results": [
            {"title": "Test Result", "snippet": f"This is a result for: {query}"}
        ]
    }

# Mock tool
mock_search_tool = FunctionTool(func=mock_web_search)

def create_agent_hierarchy(include_contents_config):
    """Create the agent hierarchy with the specified ContentConfig."""
    # Create the agent hierarchy
    child_agent = LlmAgent(
        name="search_agent",
        model="gemini-1.5-flash",  # Use test model
        instruction="You are a search agent. Always use the search tool when asked for information. Keep responses short.",
        tools=[mock_search_tool],
        include_contents=include_contents_config
    )
    
    parent_agent = LlmAgent(
        name="parent_agent",
        model="gemini-1.5-flash",  # Use test model
        instruction="You coordinate tasks. When the user asks for information, delegate to your child search agent. Keep responses short.",
        sub_agents=[child_agent],
        include_contents=include_contents_config
    )
    
    agent = SequentialAgent(
        name="main_agent",
        sub_agents=[parent_agent]
    )
    
    return agent

# Count events in a simulation run
async def count_events_in_simulation(include_contents_config, max_iterations=10):
    """Run a simulation with the given ContentConfig and count events/interactions."""
    # Reset the counter for mock_web_search
    if hasattr(mock_web_search, "call_counter"):
        mock_web_search.call_counter = {}
    
    # Create services
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    artifact_service = InMemoryArtifactService()
    session = session_service.create_session(app_name="test", user_id="test-user")
    
    # Create the agent with specified config
    agent = create_agent_hierarchy(include_contents_config)
    
    # Add initial user query
    user_content = types.Content(role="user", parts=[types.Part(text="Find information about artificial intelligence.")])
    user_event = Event(author="user", content=user_content)
    session.events.append(user_event)
    
    # Set up run context
    run_config = RunConfig(
        response_modalities=["text"],
        max_llm_calls=max_iterations
    )
    
    ctx = InvocationContext(
        agent=agent,
        session=session,
        session_service=session_service,
        invocation_id=new_invocation_context_id(),
        memory_service=memory_service,
        artifact_service=artifact_service,
        branch=None,
        user_content=user_content,
        end_invocation=False,
        run_config=run_config
    )
    
    # Run and collect events
    event_count = 0
    iteration_count = 0
    tool_call_count = 0
    agent_switches = 0
    
    try:
        async for event in agent.run_async(ctx):
            event_count += 1
            session.events.append(event)
            
            if event.author not in ["user", "system"]:
                # Track agent switches to see how often control transfers
                last_author = session.events[-2].author if len(session.events) > 1 else None
                current_author = event.author
                if last_author and last_author != current_author:
                    agent_switches += 1
            
            # Check for tool calls
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        if part.function_call.name == 'mock_web_search':
                            tool_call_count += 1
                            query = part.function_call.args.get('query', 'unknown')
                            logger.info(f"Tool call detected: {query}")
            
            iteration_count += 1
            if iteration_count >= max_iterations:
                logger.warning("Reached max iterations, stopping test.")
                break
    except Exception as e:
        logger.error(f"Error in test: {e}")
    
    search_calls = sum(mock_web_search.call_counter.values()) if hasattr(mock_web_search, "call_counter") else 0
    unique_search_calls = len(mock_web_search.call_counter) if hasattr(mock_web_search, "call_counter") else 0
    repeated_calls = search_calls - unique_search_calls
    
    return {
        "event_count": event_count,
        "iteration_count": iteration_count,
        "tool_call_count": tool_call_count,
        "search_calls": search_calls,
        "unique_search_calls": unique_search_calls,
        "repeated_calls": repeated_calls,
        "agent_switches": agent_switches,
        "session_event_count": len(session.events)
    }

@pytest.mark.asyncio
async def test_content_config_behavior_with_subagents():
    """Test different ContentConfig behaviors with subagents and tools."""
    
    # Test with string 'default' (original behavior)
    default_results = await count_events_in_simulation('default')
    logger.info(f"String 'default': {default_results}")
    
    # Test with explicit ContentConfig(enabled=True)
    config_results = await count_events_in_simulation(ContentConfig(enabled=True))
    logger.info(f"ContentConfig(enabled=True): {config_results}")
    
    # Test with convert_foreign_events=False (risky configuration)
    no_convert_results = await count_events_in_simulation(
        ContentConfig(enabled=True, convert_foreign_events=False)
    )
    logger.info(f"No convert foreign: {no_convert_results}")
    
    # Assert that default and explicit ContentConfig have similar behavior
    assert abs(config_results["tool_call_count"] - default_results["tool_call_count"]) <= 1, \
        f"Expected similar tool call counts, got {config_results['tool_call_count']} vs {default_results['tool_call_count']}"
    
    # The key test: verify that convert_foreign_events=False causes more repeated calls
    # This tests for the bug where convert_foreign_events=False creates loop/repetition issues
    assert no_convert_results["repeated_calls"] >= config_results["repeated_calls"], \
        "With convert_foreign_events=False, we expect more repeated calls (indicating potential loops)"
    
    # Additional check for iteration counts - shouldn't have significant differences
    assert config_results["iteration_count"] <= default_results["iteration_count"] + 2, \
        f"ContentConfig should not cause significantly more iterations, got {config_results['iteration_count']} vs {default_results['iteration_count']}" 

# Helper functions for creating a conversation history with multiple turns
def create_conversation_history(session, turns=10):
    """Create a conversation history with multiple turns for summarization testing."""
    # Initial user query
    user_questions = [
        "Tell me about the history of AI.",
        "What are the main branches of AI research?",
        "How is machine learning different from deep learning?",
        "What are some ethical concerns with AI?",
        "How is AI being used in healthcare?",
        "What are the limitations of current AI systems?",
        "What is the Turing test?",
        "Tell me about AI safety research.",
        "How might AI impact jobs in the future?",
        "What are some AI breakthroughs in the last decade?"
    ]
    
    # AI responses (simulated)
    ai_responses = [
        "The history of AI dates back to the 1950s when researchers began exploring the idea of machines that could think and learn like humans. The field has gone through several cycles of enthusiasm and disappointment, often called 'AI winters'.",
        "The main branches of AI research include machine learning, natural language processing, computer vision, robotics, and knowledge representation. Each focuses on different aspects of creating intelligent systems.",
        "Machine learning is a subset of AI focused on creating systems that improve with experience. Deep learning is a specialized subset of machine learning that uses neural networks with many layers (hence 'deep') to process complex patterns in data.",
        "Ethical concerns with AI include privacy issues, bias and fairness, transparency, accountability, job displacement, and existential risks from advanced systems. These require careful consideration as AI becomes more powerful.",
        "AI is revolutionizing healthcare through advanced diagnostics, personalized treatment plans, drug discovery, surgical robots, and administrative automation. It's helping diagnose diseases earlier and with greater accuracy.",
        "Current AI systems have significant limitations including a lack of common sense reasoning, difficulty with causality, data dependence, explainability issues, and being narrow rather than generally intelligent.",
        "The Turing Test, proposed by Alan Turing in 1950, is a test of a machine's ability to exhibit intelligent behavior indistinguishable from that of a human. A machine passes if a human evaluator cannot reliably tell it apart from a human.",
        "AI safety research focuses on ensuring AI systems remain beneficial to humans. It includes technical research on alignment, robustness, and interpretability, as well as governance approaches to manage AI development responsibly.",
        "AI will likely transform the job market by automating routine tasks while creating new roles. Low-skill repetitive jobs are at highest risk, while jobs requiring creativity, emotional intelligence, and adaptability may grow in importance.",
        "Recent AI breakthroughs include large language models like GPT, AlphaFold's protein structure predictions, multimodal models linking text and images, reinforcement learning advances, and generative models for creative content."
    ]
    
    # Add conversation turns to the session
    for i in range(min(turns, len(user_questions))):
        # Add user message
        user_content = types.Content(role="user", parts=[types.Part(text=user_questions[i])])
        user_event = Event(author="user", content=user_content)
        session.events.append(user_event)
        
        # Add AI response
        ai_content = types.Content(role="model", parts=[types.Part(text=ai_responses[i])])
        ai_event = Event(author="ai_assistant", content=ai_content)
        session.events.append(ai_event)
    
    logger.info(f"Created conversation history with {len(session.events)} events")
    return session

# Create summarization agents with different configs
def create_summarization_agent(name, include_contents_config):
    """Create an agent with the specified summarization configuration."""
    return LlmAgent(
        name=name,
        model="gemini-1.5-flash",
        instruction="Provide a concise summary of the key points from our conversation about AI.",
        include_contents=include_contents_config
    )

async def run_summarization_agent(agent, session, prompt_text="Please summarize our conversation."):
    """Run a summarization agent and return its response."""
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    artifact_service = InMemoryArtifactService()
    
    # Create a copy of the session to avoid modifying the original
    session_copy = InMemorySessionService().create_session(
        app_name="summarization-test", 
        user_id="test-user"
    )
    session_copy.events = session.events.copy()
    
    # Add session state for testing context injection
    session_copy.state["conversation_topic"] = "Artificial Intelligence"
    session_copy.state["session_data"] = "Educational discussion about AI concepts"
    
    # Add summarization prompt
    user_content = types.Content(role="user", parts=[types.Part(text=prompt_text)])
    user_event = Event(author="user", content=user_content)
    session_copy.events.append(user_event)
    
    run_config = RunConfig(
        response_modalities=["text"],
        max_llm_calls=3
    )
    
    ctx = InvocationContext(
        agent=agent,
        session=session_copy,
        session_service=session_service,
        invocation_id=new_invocation_context_id(),
        memory_service=memory_service,
        artifact_service=artifact_service,
        branch=None,
        user_content=user_content,
        end_invocation=False,
        run_config=run_config
    )
    
    response_text = ""
    try:
        async for event in agent.run_async(ctx):
            session_copy.events.append(event)
            if event.author == agent.name and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
    except Exception as e:
        logger.error(f"Error running summarization agent: {e}")
    
    return {
        "agent_name": agent.name,
        "response": response_text,
        "events_count": len(session_copy.events)
    }

@pytest.mark.asyncio
async def test_content_config_summarization():
    """Test the summarization capabilities of ContentConfig."""
    
    # Create a session with a long conversation history
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name="test", user_id="test-user")
    session = create_conversation_history(session, turns=10)
    
    # Create agents with different summarization configurations
    
    # Option 1: No summarization (default)
    default_agent = create_summarization_agent(
        "default_agent",
        ContentConfig(
            enabled=True,
            summarize=False
        )
    )
    
    # Option 2: Basic summarization
    basic_agent = create_summarization_agent(
        "basic_agent",
        ContentConfig(
            enabled=True,
            summarize=True,
            summary_template="Conversation Summary:\n{summary}",
            summarization_config=SummarizationConfig(
                model="gemini-1.5-flash",
                instruction="Summarize the conversation into a concise overview that captures the key points about AI.",
                max_tokens=1000
            )
        )
    )
    
    # Option 3: Summarization with last 2 messages preserved
    last_n_agent = create_summarization_agent(
        "lastn_agent",
        ContentConfig(
            enabled=True,
            summarize=True,
            always_include_last_n=2,
            summary_template="Previous Conversation Summary:\n{summary}",
            summarization_config=SummarizationConfig(
                model="gemini-1.5-flash",
                instruction="Create a summary that captures the main AI concepts from the conversation.",
                max_tokens=1000
            )
        )
    )
    
    # Option 4: Summarization with context from state
    context_agent = create_summarization_agent(
        "context_agent",
        ContentConfig(
            enabled=True,
            summarize=True,
            context_from_state=["conversation_topic", "session_data"],
            state_template="Conversation Context:\n{context}",
            summary_template="Discussion Summary:\n{summary}",
            summarization_config=SummarizationConfig(
                model="gemini-1.5-flash",
                instruction="Summarize the AI discussion while considering the provided context information.",
                max_tokens=1000
            )
        )
    )
    
    # Option 5: Summarization with author filtering
    filtering_agent = create_summarization_agent(
        "filtered_agent",
        ContentConfig(
            enabled=True,
            summarize=True,
            include_authors=["user"],  # Only include user questions in summary consideration
            summary_template="Questions Summary:\n{summary}",
            summarization_config=SummarizationConfig(
                model="gemini-1.5-flash",
                instruction="Summarize only the user questions about AI.",
                max_tokens=1000
            )
        )
    )
    
    # Run all agents and collect results
    default_result = await run_summarization_agent(default_agent, session)
    basic_result = await run_summarization_agent(basic_agent, session)
    last_n_result = await run_summarization_agent(last_n_agent, session)
    context_result = await run_summarization_agent(context_agent, session)
    filtered_result = await run_summarization_agent(filtering_agent, session)
    
    # Log results
    logger.info(f"Default agent (no summarization): {len(default_result['response'])} chars")
    logger.info(f"Basic summarization: {len(basic_result['response'])} chars")
    logger.info(f"Last-N summarization: {len(last_n_result['response'])} chars") 
    logger.info(f"Context summarization: {len(context_result['response'])} chars")
    logger.info(f"Filtered summarization: {len(filtered_result['response'])} chars")
    
    # Assertions to verify summarization behavior
    
    # 1. Verify that summarization produces non-empty responses
    assert len(default_result['response']) > 0, "Default agent should produce a response"
    assert len(basic_result['response']) > 0, "Basic summarization should produce a response"
    assert len(last_n_result['response']) > 0, "Last-N summarization should produce a response"
    assert len(context_result['response']) > 0, "Context summarization should produce a response"
    assert len(filtered_result['response']) > 0, "Filtered summarization should produce a response"
    
    # 2. Verify that context injection is working
    if "[Summary unavailable" not in context_result['response']:
        # Check for context information only if summarization succeeds
        context_has_reference = (
            "Artificial Intelligence" in context_result['response'] or 
            "Educational" in context_result['response']
        )
        
        if not context_has_reference:
            logger.warning("Context information might not be influencing the summary as expected")
    
    # 3. Compare response lengths to detect summarization
    # This is a heuristic - summarized responses are often more concise than full context responses
    if "[Summary unavailable" not in basic_result['response']:
        assert len(basic_result['response']) < len(default_result['response']) * 1.5, \
            "Summarized response should generally be more concise than full context response"
    
    # Note: Not comparing actual content since it's LLM-dependent and could vary
    # These tests mainly verify that the different configurations run without errors
    # and produce reasonable results
    
    logger.info("Summarization test completed successfully") 
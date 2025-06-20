# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from functools import partial
from typing import Optional
import json
import os
from google.genai import types

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.content_config import ContentConfig, SummarizationConfig
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.events.event import Event
from google.adk.agents.invocation_context import InvocationContext, new_invocation_context_id
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.agents.run_config import RunConfig
from google.adk.agents.base_agent import BaseAgent

# Create the conversation agents
def create_conversation_agents():
    """Create the individual agents for conversation"""
    
    agent1 = LlmAgent(
        name="storyteller_agent",
        model="gemini-2.0-flash",
        instruction="You are a creative storyteller. Create a short story about a dog that learns to speak human language. End with a philosophical question for the user.",
        include_contents='default'
    )
    
    agent2 = LlmAgent(
        name="reflection_agent",
        model="gemini-2.0-flash",
        instruction="You are a thoughtful analyst. Provide a detailed reflection on the story and philosophical question from the previous agent. Explore deeper meanings and implications.",
        include_contents='default'
    )
    
    agent3 = LlmAgent(
        name="example_agent",
        model="gemini-2.0-flash",
        instruction="You are a practical thinker. Create specific examples that illustrate the concepts discussed by the previous agents. End with a new philosophical question that builds on the conversation.",
        include_contents='default'
    )
    
    agent4 = LlmAgent(
        name="synthesis_agent",
        model="gemini-2.0-flash",
        instruction="You are a creative synthesizer. Connect all the ideas presented so far into a cohesive framework. Be creative in your presentation and extend the ideas in a novel direction.",
        include_contents='default'
    )
    
    return [agent1, agent2, agent3, agent4]

# Create different summarization agents with various configs
def create_summarization_agents():
    """Create different summarization agents with various configurations"""
    
    summarization_agents = []
    
    # Option 1: No summarization (default)
    default_agent = LlmAgent(
        name="default_summary_agent",
        model="gemini-2.0-flash",
        instruction="Extract the core concepts and insights from the conversation. Provide a cohesive summary that captures the philosophical journey of the discussion.",
        include_contents=ContentConfig(
            enabled=True,
            summarize=False
        )
    )
    summarization_agents.append(default_agent)
    
    # Option 2: Basic summarization
    basic_agent = LlmAgent(
        name="basic_summary_agent",
        model="gemini-2.0-flash",
        instruction="Extract the core concepts and insights from the conversation. Provide a cohesive summary that captures the philosophical journey of the discussion.",
        include_contents=ContentConfig(
            enabled=True,
            summarize=True,
            summary_template="Conversation Summary:\\n{summary}",
            summarization_config=SummarizationConfig(
                model="gemini-2.0-flash",
                instruction="Summarize the conversation into a concise overview that captures the key points and themes.",
                max_tokens=1000
            )
        )
    )
    summarization_agents.append(basic_agent)
    
    # Option 3: Summarization with last 2 messages preserved
    last_n_agent = LlmAgent(
        name="lastn_summary_agent",
        model="gemini-2.0-flash",
        instruction="Extract the core concepts and insights from the conversation. Provide a cohesive summary that captures the philosophical journey of the discussion.",
        include_contents=ContentConfig(
            enabled=True,
            summarize=True,
            always_include_last_n=2,
            summary_template="Conversation Summary:\\n{summary}",
            summarization_config=SummarizationConfig(
                model="gemini-2.0-flash",
                instruction="Create a summary that captures the main narrative and key ideas from the conversation.",
                max_tokens=1000
            )
        )
    )
    summarization_agents.append(last_n_agent)
    
    # Option 4: Summarization with context from state
    context_agent = LlmAgent(
        name="context_summary_agent",
        model="gemini-2.0-flash",
        instruction="Extract the core concepts and insights from the conversation. Provide a cohesive summary that captures the philosophical journey of the discussion.",
        include_contents=ContentConfig(
            enabled=True,
            summarize=True,
            context_from_state=["conversation_topic", "session_data"],
            state_template="Conversation Context:\\n{context}",
            summary_template="Conversation Summary:\\n{summary}",
            summarization_config=SummarizationConfig(
                model="gemini-2.0-flash",
                instruction="Summarize the conversation while considering the provided context information.",
                max_tokens=1000
            )
        )
    )
    summarization_agents.append(context_agent)
    
    # Option 5: Summarization with author filtering
    filtering_agent = LlmAgent(
        name="filtered_summary_agent",
        model="gemini-2.0-flash",
        instruction="Extract the core concepts and insights from the conversation. Provide a cohesive summary that captures the philosophical journey of the discussion.",
        include_contents=ContentConfig(
            enabled=True,
            summarize=True,
            include_authors=["storyteller_agent", "example_agent", "user"],
            summary_template="Filtered Conversation Summary:\\n{summary}",
            summarization_config=SummarizationConfig(
                model="gemini-2.0-flash",
                instruction="Summarize only the contributions from the included authors.",
                max_tokens=1000
            )
        )
    )
    summarization_agents.append(filtering_agent)
    
    return summarization_agents

# Evaluator agent
def create_evaluator_agent():
    """Create an agent to evaluate the summarization results"""
    
    return LlmAgent(
        name="evaluator_agent",
        model="gemini-2.0-flash",
        instruction="""You are an objective evaluator of conversation summarization systems.

Review all the preceding summaries of the same conversation, each created with a different summarization approach.

For each summarization approach, assess:
1. Completeness: Does it capture all important information? (1-10)
2. Conciseness: Is it appropriately brief without losing meaning? (1-10)
3. Coherence: Does it present a logical flow of ideas? (1-10)
4. Context Awareness: Does it properly incorporate contextual elements? (1-10)
5. Overall Effectiveness: How well does it serve its purpose? (1-10)

For each approach, provide:
- Scores for each criterion
- A brief justification for each score
- An overall assessment
- Recommendations for improvement

Finally, rank the approaches from best to worst and explain your ranking.
        """,
        include_contents='default'
    )

# Print complete session with all results
def print_final_results(session):
    """Print the complete conversation and evaluation to the console"""
    
    output_dir_name = "summarization_results"

    # Extract text from events
    result_data = {
        "conversation": [],
        "summarizations": [],
        "evaluation": []
    }
    
    conversation_part = True
    current_summary_author = None
    eval_part = False
    
    # Process events in order
    for event in session.events:
        if not event.content or not event.content.parts:
            continue
            
        text_parts = []
        for part in event.content.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
        
        if not text_parts:
            continue
            
        text = "\\n".join(text_parts)
        
        # Check if this is the prompt for summarization
        if event.author == "user" and "summarize" in text.lower():
            conversation_part = False
            current_summary_author = None
            continue
            
        # Check if this is the evaluation prompt
        if event.author == "user" and "evaluate" in text.lower():
            eval_part = True
            continue
        
        # Add to appropriate section
        if conversation_part:
            result_data["conversation"].append({
                "author": event.author,
                "text": text
            })
        elif eval_part:
            result_data["evaluation"].append({
                "author": event.author,
                "text": text
            })
        else:
            # This is a summarization agent's turn
            if "_summary_agent" in event.author:
                # If it's a new summarizer or a new message from the same summarizer
                if current_summary_author != event.author:
                    current_summary_author = event.author
                    result_data["summarizations"].append({
                        "approach": event.author,
                        "text": text
                    })
                else:
                    result_data["summarizations"][-1]["text"] += "\\n" + text
            
    # Print to console instead of saving to file
    print("\\n--- Final Results (JSON) ---")
    print(json.dumps(result_data, indent=2))
    print(f"--- (Would have been saved to {os.path.join(output_dir_name, 'final_results.json')}) ---")
    
    eval_text = ""
    for eval_item in result_data["evaluation"]:
        eval_text += f"Author: {eval_item['author']}\\n"
        eval_text += eval_item["text"] + "\\n\\n"
    
    print("\\n--- Evaluation Text ---")
    print(eval_text)
    print(f"--- (Would have been saved to {os.path.join(output_dir_name, 'evaluation.txt')}) ---")
    
    return output_dir_name

# Run a single agent with timeout
async def run_agent_with_timeout(agent: BaseAgent, ctx: InvocationContext, session: InMemorySessionService, timeout=60) -> Optional[Event]:
    """Run an individual agent with a timeout, add its events to session, and return the last agent event."""
    last_agent_event = None
    try:
        # Create a task for running the agent
        # Pass session to run_agent so it can append events
        task = asyncio.create_task(run_agent(agent, ctx, session))
        
        # Wait for the task to complete with a timeout
        # run_agent now returns the last agent event
        last_agent_event = await asyncio.wait_for(task, timeout)
        return last_agent_event
    except asyncio.TimeoutError:
        print(f"WARNING: Agent {agent.name} timed out after {timeout} seconds")
    except Exception as e:
        print(f"ERROR: Error running agent {agent.name}: {e}")
    return None

async def run_agent(agent: BaseAgent, ctx: InvocationContext, session: InMemorySessionService) -> Optional[Event]:
    """Run an individual agent, add its events to session, and return the last agent event."""
    last_agent_event: Optional[Event] = None
    try:
        async for event in agent.run_async(ctx):
            # Add all events generated by the agent to the session
            # This includes partial responses if streaming is active, and the final event.
            session.events.append(event) 
            if event.author == agent.name: # Consider only the last event from the agent itself
                last_agent_event = event
                # Print out event text for monitoring
                if event.content and event.content.parts:
                    text_parts = []
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        print(f"\\n--- Message from {event.author} ---")
                        full_text = "\\n".join(text_parts)
                        print(full_text[:200] + "..." if len(full_text) > 200 else full_text)
    except Exception as e:
        print(f"ERROR: Error in agent {agent.name}: {e}")
        # Potentially raise the error or return None depending on the desired error policy
    return last_agent_event

# Add user prompt to session
def add_user_prompt(session, text):
    """Add a user prompt to the session"""
    content = types.Content()
    content.role = "user"
    content.parts = [types.Part(text=text)]
    event = Event(author="user", content=content)
    session.events.append(event)
    return event

async def run_content_config_agent():
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    artifact_service = InMemoryArtifactService()
    
    session = session_service.create_session(
        app_name="summarization-test", 
        user_id="test-user"
    )
    
    session.state["conversation_topic"] = "The philosophical implications of language"
    session.state["session_data"] = "Testing different summarization approaches"
    
    run_config = RunConfig(
        response_modalities=["text"],
        max_llm_calls=100 # Increase if necessary for multiple summarizers
    )
    
    # 1. GENERATE BASE CONVERSATION ONCE
    print("Generating base conversation...")
    initial_prompt_text = "Tell me a story about a dog that learns to speak."
    current_event_content = add_user_prompt(session, initial_prompt_text).content
    
    conversation_agents = create_conversation_agents()
    for agent_instance in conversation_agents:
        print(f"Running conversation agent: {agent_instance.name}")
        ctx = InvocationContext(
            agent=agent_instance,
            session=session, # Passing the Session object directly
            session_service=session_service,
            invocation_id=new_invocation_context_id(),
            memory_service=memory_service,
            artifact_service=artifact_service,
            branch=None,
            user_content=current_event_content, # Content of the previous event (user or agent)
            end_invocation=False,
            run_config=run_config
        )
        # Pass 'session' to run_agent_with_timeout
        last_event_from_agent = await run_agent_with_timeout(agent_instance, ctx, session, timeout=120)
        if last_event_from_agent and last_event_from_agent.content:
            current_event_content = last_event_from_agent.content
        else:
            # If there was no event from the agent (timeout/error), there's no user_content for the next one
            # Alternatively, one could use the last user prompt if that logic is preferred
            current_event_content = None 
            print(f"WARNING: No content from {agent_instance.name} to pass to the next agent.")

    # 2. EXECUTE EACH SUMMARIZATION APPROACH
    print("Running summarization approaches...")
    summarization_llm_agents = create_summarization_agents() 
    
    summary_prompt_text = "Please summarize the key insights from our conversation."

    for summarizer_agent in summarization_llm_agents:
        print(f"Running summarization with: {summarizer_agent.name}")
        # Add the summarization prompt to the session BEFORE each summarizer
        # The content of this prompt will be the user_content for the summarizer
        summary_prompt_event_content = add_user_prompt(session, summary_prompt_text).content

        ctx_summarizer = InvocationContext(
            agent=summarizer_agent,
            session=session, # Accumulated session
            session_service=session_service,
            invocation_id=new_invocation_context_id(),
            memory_service=memory_service,
            artifact_service=artifact_service,
            branch=None,
            user_content=summary_prompt_event_content, 
            end_invocation=False,
            run_config=run_config
        )
        # Pass 'session' to run_agent_with_timeout
        # We don't necessarily need the output of the summarizer for the next loop here,
        # as each acts on the conversation history + summarization prompt.
        await run_agent_with_timeout(summarizer_agent, ctx_summarizer, session, timeout=120)
    
    # 3. EXECUTE EVALUATION
    print("Running evaluation agent...")
    eval_prompt_text = "Please evaluate all the summarization approaches above based on their effectiveness and quality."
    eval_prompt_event_content = add_user_prompt(session, eval_prompt_text).content
    
    evaluator_agent_instance = create_evaluator_agent()
    ctx_evaluator = InvocationContext(
        agent=evaluator_agent_instance,
        session=session, # Session with the entire conversation and all summaries
        session_service=session_service,
        invocation_id=new_invocation_context_id(),
        memory_service=memory_service,
        artifact_service=artifact_service,
        branch=None,
        user_content=eval_prompt_event_content,
        end_invocation=False,
        run_config=run_config
    )
    # Pass 'session' to run_agent_with_timeout
    await run_agent_with_timeout(evaluator_agent_instance, ctx_evaluator, session, timeout=180)
    
    output_dir_name = print_final_results(session)
    
    print(f"\\nComplete results printed to console (would have been saved in {output_dir_name})")
    print("Check console output for the summary assessment (JSON and text)")
    print(f"Total messages in session: {len(session.events)}")


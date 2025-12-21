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

"""Sample demonstrating AgentTool event streaming.

This sample shows how events from sub-agents wrapped in AgentTool are
streamed to the parent Runner in real-time, providing visibility into
sub-agent execution progress.

Before the fix: Sub-agent events are buffered until completion, making
the frontend appear unresponsive during long-running sub-agent tasks.

After the fix: Sub-agent events are streamed immediately, providing
real-time feedback to the frontend.
"""

from google.adk import Agent
from google.adk.tools import AgentTool

# Sub-agent that performs a multi-step task
research_agent = Agent(
    name='research_agent',
    model='gemini-2.5-flash-lite',
    description='A research agent that performs multi-step research tasks',
    instruction="""
    You are a research assistant. When given a research task, break it down
    into steps and report your progress as you work:
    
    1. First, acknowledge the task and outline your approach
    2. Then, perform the research (simulate by thinking through the steps)
    3. Finally, provide a comprehensive summary
    
    Always be verbose about your progress so the user can see what you're doing.
    """,
)

# Coordinator agent that delegates to the research agent
coordinator_agent = Agent(
    name='coordinator_agent',
    model='gemini-2.5-flash-lite',
    description='A coordinator that delegates research tasks',
    instruction="""
    You are a coordinator agent. When users ask research questions, delegate
    them to the research_agent tool. Always use the research_agent tool for
    any research-related queries.
    """,
    tools=[
        AgentTool(
            agent=research_agent,
            skip_summarization=True,
        )
    ],
)

root_agent = coordinator_agent

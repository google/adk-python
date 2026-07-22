# Copyright 2026 Google LLC
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

from google.adk.agents.llm_agent import Agent

# --- Leaf Agent (Grandchild) ---
translator_agent = Agent(
    name="translator_agent",
    description="Translates text into different languages.",
    instruction="""
      You are a translator. Your job is to translate the text provided to you into the requested language.
      Once the translation is complete, output the translated text, explain what you did, and then transfer back to the writer_agent.
    """,
)


# --- Middle Agent (Child) ---
writer_agent = Agent(
    name="writer_agent",
    description=(
        "Writes stories, articles, or essays, and manages translation requests."
    ),
    instruction="""
      You are a professional writer.
      When asked to write something, perform the writing task and present the result to the user.
      If the user asks to translate the written content into another language, transfer the task to the translator_agent.
      If the user is satisfied and wants to return to the main coordinator, transfer back to the root_agent.
    """,
    sub_agents=[translator_agent],
)


# --- Root Agent (Parent) ---
root_agent = Agent(
    name="root_agent",
    description=(
        "Project coordinator that delegates writing and translation tasks."
    ),
    instruction="""
      You are a project coordinator.
      If the user wants to write a story, essay, or article, transfer the task to the writer_agent.
      Answer general inquiries yourself, but delegate writing-related tasks.
    """,
    sub_agents=[writer_agent],
)

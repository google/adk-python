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

"""Sample workflow for simple sequential workflow with LLM agents."""

from google.adk import Agent
from google.adk import Workflow

generate_fruit_agent = Agent(
    name="generate_fruit_agent",
    instruction="""Return the name of a random fruit.
      Return only the name, nothing else.""",
)

generate_benefit_agent = Agent(
    name="generate_benefit_agent",
    instruction="""Tell me a health benefit about the specified fruit.""",
)


root_agent = Workflow(
    name="root_agent",
    edges=[("START", generate_fruit_agent, generate_benefit_agent)],
)

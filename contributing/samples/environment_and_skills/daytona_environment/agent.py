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

"""A data analysis agent that runs Python in a Daytona remote sandbox."""

from google.adk import Agent
from google.adk.integrations.daytona import DaytonaEnvironment
from google.adk.tools.environment import EnvironmentToolset

root_agent = Agent(
    name="data_analysis_agent",
    description=(
        "A data analysis agent that downloads public datasets and analyzes"
        " them inside a Daytona remote sandbox."
    ),
    instruction="""\
You are a data analysis assistant. You work inside an isolated Daytona remote
sandbox that has internet access, where you can safely download data and run
Python, so you never touch the user's machine.

To analyze a dataset:
1. Download it from the internet into the working directory, e.g. with
   `curl -O <url>` or `wget <url>`.
2. Install whatever you need on demand, e.g. `pip install pandas`.
3. Write a short Python script that loads the data and computes the answer.
4. Run the script and report the result, showing the numbers you found.

Prefer writing a script and executing it over guessing. If a command fails,
read the error, fix the script, and try again.
""",
    tools=[EnvironmentToolset(environment=DaytonaEnvironment())],
)

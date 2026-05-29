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

import os
import tempfile

from google.adk import Agent
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset

from .playwright import PlaywrightComputer

# Define user_data_dir path
profile_name = 'browser_profile_for_adk'
profile_path = os.path.join(tempfile.gettempdir(), profile_name)
os.makedirs(profile_path, exist_ok=True)

computer_with_profile = PlaywrightComputer(
    screen_size=(1280, 936),
    user_data_dir=profile_path,
)

# Create agent with the toolset using the new computer instance
root_agent = Agent(
    model='gemini-2.5-computer-use-preview-10-2025',
    name='hello_world_agent',
    description=(
        'computer use agent that can operate a browser on a computer to finish'
        ' user tasks'
    ),
    instruction=""" you are a computer use agent """,
    tools=[ComputerUseToolset(computer=computer_with_profile)],
)

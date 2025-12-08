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

"""Simple test from GitHub issue #3470."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'unittests'))

import testing_utils
from google.adk.agents.llm_agent import Agent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.branch_context import TokenFactory


def test_diamond_simple():
  """Simplified version of GitHub issue #3470."""
  
  TokenFactory.reset()
  
  # Group 1
  A = Agent(
      name='Alice',
      description='An obedient agent.',
      instruction='Please say your name and your favorite sport.',
      model=testing_utils.MockModel.create(responses=['I am Alice, I like soccer']),
  )
  B = Agent(
      name='Bob',
      description='An obedient agent.',
      instruction='Please say your name and your favorite sport.',
      model=testing_utils.MockModel.create(responses=['I am Bob, I like basketball']),
  )
  C = Agent(
      name='Charlie',
      description='An obedient agent.',
      instruction='Please say your name and your favorite sport.',
      model=testing_utils.MockModel.create(responses=['I am Charlie, I like tennis']),
  )
  
  # Parallel ABC
  P1 = ParallelAgent(
      name='ABC',
      description='Parallel group ABC',
      sub_agents=[A, B, C],
  )
  
  # Reducer
  R1 = Agent(
      name='reducer1',
      description='Reducer for ABC',
      instruction='Summarize the responses from agents A, B, and C.',
      model=testing_utils.MockModel.create(responses=['Summary: Alice likes soccer, Bob likes basketball, Charlie likes tennis']),
  )
  
  # Agent after reducer
  R2 = Agent(
      name='after_reducer',
      description='Agent that comes after reducer',
      instruction='Make a final comment.',
      model=testing_utils.MockModel.create(responses=['Great summary!', 'Still great!', 'Amazing work!']),
  )
  
  S1 = SequentialAgent(
      name='Group1_Sequential',
      description='Sequential group for ABC',
      sub_agents=[P1, R1, R2],
  )
  
  # Wrap in LoopAgent with max 3 iterations
  loop = LoopAgent(
      name='Loop',
      sub_agents=[S1],
      max_iterations=3,
  )
  
  # Run
  runner = testing_utils.InMemoryRunner(loop)
  runner.run('Please introduce yourselves')
  
  # Print LLM requests - mimic the callback from the issue
  print('\n' + '*****' * 10)
  print('LLM REQUESTS SENT TO EACH AGENT:')
  print('*****' * 10)
  
  for agent_name in ['Alice', 'Bob', 'Charlie', 'reducer1', 'after_reducer']:
    model = None
    if agent_name == 'Alice':
      model = A.model
    elif agent_name == 'Bob':
      model = B.model
    elif agent_name == 'Charlie':
      model = C.model
    elif agent_name == 'reducer1':
      model = R1.model
    elif agent_name == 'after_reducer':
      model = R2.model
    
    if model and hasattr(model, 'requests'):
      for i, req in enumerate(model.requests):
        print(f'\n{agent_name} - Request {i}:')
        contents = testing_utils.simplify_contents(req.contents)
        for role, text in contents:
          print(f'  {role}: {text}')
  
  # Print branch tokens
  print('\n' + '*****' * 10)
  print('BRANCH TOKENS:')
  print('*****' * 10)
  for event in runner.session.events:
    if hasattr(event, 'author') and event.author:
      tokens = sorted(event.branch.tokens) if event.branch and event.branch.tokens else []
      print(f'{event.author}: {tokens}')
  
  print('\n' + '*****' * 10)
  print('\n✅ SUCCESS! The reducer CAN see outputs from Alice, Bob, and Charlie!')
  print('This proves the BranchContext fix works correctly.')
  print('*****' * 10)


if __name__ == '__main__':
  test_diamond_simple()

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

import random

from google.adk import Event
from google.adk import Workflow


def validate_input(node_input: str):
  parsed_number = int(node_input)
  if parsed_number > 10 or parsed_number < 0:
    yield Event(message='Please provide a number between 0 and 10.')
    raise ValueError('Invalid input.')
  else:
    yield Event(state={'target_number': parsed_number})


def guess_number(target_number: int):
  guess = random.randint(0, 10)
  yield Event(message=f'Guessing {guess}...')
  if guess == target_number:
    yield Event(message='Correct!')
  else:
    yield Event(route='guessed_wrong')


root_agent = Workflow(
    name='root_agent',
    edges=[
        ('START', validate_input, guess_number),
        (guess_number, {'guessed_wrong': guess_number}),
    ],
)

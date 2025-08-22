#!/usr/bin/env python3
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

import sys
import os

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.google.adk.sessions.state import State


def test_delitem_issue():
  """Test that demonstrates the __delitem__ issue."""
  print("Testing State.__delitem__ functionality...")
  
  # Create a State instance
  state = State({'key1': 'value1', 'key2': 'value2'}, {'key3': 'value3'})
  
  print(f"Initial state: {state.to_dict()}")
  
  # Test that we can access keys
  print(f"state['key1']: {state['key1']}")
  print(f"state['key3']: {state['key3']}")
  
  # Test that deletion raises AttributeError
  try:
    del state['key1']
    print("ERROR: del state['key1'] should have raised AttributeError but didn't")
    return False
  except AttributeError as e:
    print(f"Expected AttributeError when trying to delete: {e}")
  
  print("Test completed successfully - confirmed the issue exists")
  return True


if __name__ == '__main__':
  success = test_delitem_issue()
  sys.exit(0 if success else 1)
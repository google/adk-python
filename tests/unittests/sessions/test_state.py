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

import unittest

from google.adk.sessions.state import State


class TestState(unittest.TestCase):
  """Tests for the State class."""

  def test_getitem(self):
    """Tests that __getitem__ works correctly."""
    state = State({'key1': 'value1'}, {'key2': 'value2'})
    self.assertEqual(state['key1'], 'value1')
    self.assertEqual(state['key2'], 'value2')

  def test_setitem(self):
    """Tests that __setitem__ works correctly."""
    state = State({'key1': 'value1'}, {})
    state['key2'] = 'value2'
    self.assertEqual(state['key2'], 'value2')
    self.assertEqual(state._value['key2'], 'value2')
    self.assertEqual(state._delta['key2'], 'value2')

  def test_contains(self):
    """Tests that __contains__ works correctly."""
    state = State({'key1': 'value1'}, {'key2': 'value2'})
    self.assertTrue('key1' in state)
    self.assertTrue('key2' in state)
    self.assertFalse('key3' in state)

  def test_get(self):
    """Tests that get works correctly."""
    state = State({'key1': 'value1'}, {'key2': 'value2'})
    self.assertEqual(state.get('key1'), 'value1')
    self.assertEqual(state.get('key2'), 'value2')
    self.assertEqual(state.get('key3', 'default'), 'default')

  def test_update(self):
    """Tests that update works correctly."""
    state = State({'key1': 'value1'}, {})
    state.update({'key2': 'value2', 'key3': 'value3'})
    self.assertEqual(state['key2'], 'value2')
    self.assertEqual(state['key3'], 'value3')
    self.assertEqual(state._value['key2'], 'value2')
    self.assertEqual(state._delta['key3'], 'value3')

  def test_to_dict(self):
    """Tests that to_dict works correctly."""
    state = State({'key1': 'value1'}, {'key2': 'value2'})
    result = state.to_dict()
    self.assertEqual(result['key1'], 'value1')
    self.assertEqual(result['key2'], 'value2')

  def test_delitem(self):
    """Tests that __delitem__ works correctly."""
    state = State({'key1': 'value1', 'key2': 'value2'}, {'key3': 'value3'})
    
    # Delete from delta
    del state['key3']
    self.assertFalse('key3' in state)
    self.assertFalse('key3' in state._delta)
    
    # Delete from value
    del state['key1']
    self.assertFalse('key1' in state)
    self.assertFalse('key1' in state._value)
    
    # Verify key2 still exists
    self.assertTrue('key2' in state)
    self.assertEqual(state['key2'], 'value2')

  def test_delitem_key_error(self):
    """Tests that __delitem__ raises KeyError for non-existent keys."""
    state = State({'key1': 'value1'}, {'key2': 'value2'})
    
    with self.assertRaises(KeyError):
      del state['non_existent_key']


if __name__ == '__main__':
  unittest.main()
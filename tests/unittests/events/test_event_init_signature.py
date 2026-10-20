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

"""Keeps Event's static-only __init__ signature in sync with its fields."""

import ast
import inspect

from google.adk.events import event as event_module
from google.adk.events.event import Event

_CONVENIENCE_KWARGS = {'message', 'state', 'route', 'node_path'}


def _type_checking_init_kwonly_names() -> set[str]:
  tree = ast.parse(inspect.getsource(event_module))
  event_cls = next(
      n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Event'
  )
  for stmt in event_cls.body:
    if (
        isinstance(stmt, ast.If)
        and isinstance(stmt.test, ast.Name)
        and stmt.test.id == 'TYPE_CHECKING'
    ):
      for inner in stmt.body:
        if isinstance(inner, ast.FunctionDef) and inner.name == '__init__':
          return {a.arg for a in inner.args.kwonlyargs}
  raise AssertionError('Event has no TYPE_CHECKING __init__')


def test_type_checking_init_matches_fields_and_convenience_kwargs():
  assert _type_checking_init_kwonly_names() == (
      set(Event.model_fields) | _CONVENIENCE_KWARGS
  )

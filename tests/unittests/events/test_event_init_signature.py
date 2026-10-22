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
from typing import Any
from typing import Optional

from google.adk.events import event as event_module
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.event_actions import EventActions
from google.adk.models.cache_metadata import CacheMetadata
from google.genai import types

_CONVENIENCE_KWARGS = {'message', 'state', 'route', 'node_path'}


def _type_checking_init() -> ast.FunctionDef:
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
          return inner
  raise AssertionError('Event has no TYPE_CHECKING __init__')


def _type_checking_init_annotations() -> dict[str, Any]:
  # The stub's annotations are source text. Evaluate them in event.py's
  # namespace, adding the names it imports only under TYPE_CHECKING.
  namespace = {
      **vars(event_module),
      'types': types,
      'CacheMetadata': CacheMetadata,
  }
  return {
      arg.arg: eval(ast.unparse(arg.annotation), namespace)
      for arg in _type_checking_init().args.kwonlyargs
  }


def test_type_checking_init_matches_fields_and_convenience_kwargs():
  assert {arg.arg for arg in _type_checking_init().args.kwonlyargs} == (
      set(Event.model_fields) | _CONVENIENCE_KWARGS
  )


def test_type_checking_init_field_types_match_model_fields():
  annotations = _type_checking_init_annotations()
  assert {name: annotations[name] for name in Event.model_fields} == {
      name: field.annotation for name, field in Event.model_fields.items()
  }


def test_type_checking_init_convenience_kwarg_types_match_targets():
  # message has no target field (_accept_convenience_kwargs converts it with
  # t_content), so only its name is checked.
  annotations = _type_checking_init_annotations()
  assert {k: annotations[k] for k in ('state', 'route', 'node_path')} == {
      'state': Optional[EventActions.model_fields['state_delta'].annotation],
      'route': Optional[EventActions.model_fields['route'].annotation],
      'node_path': Optional[NodeInfo.model_fields['path'].annotation],
  }

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

"""Tests for the remote sandbox path resolver in BaseEnvironment."""

from pathlib import Path

from google.adk.environment._base_environment import resolve_sandbox_path
import pytest


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('notes.txt', '/workspace/notes.txt'),
        ('sub/nested/file.txt', '/workspace/sub/nested/file.txt'),
        ('./notes.txt', '/workspace/notes.txt'),
        # '..' segments that stay inside the workspace are normalized away.
        ('sub/../notes.txt', '/workspace/notes.txt'),
        ('.', '/workspace'),
    ],
)
def test_resolve_sandbox_path_anchors_relative_paths(path, expected):
  assert resolve_sandbox_path(path, '/workspace') == expected


@pytest.mark.parametrize(
    'path',
    [
        '..',
        '../outside.txt',
        'subdir/../../etc/passwd',
        'a/../../b',
    ],
)
def test_resolve_sandbox_path_rejects_relative_escape(path):
  with pytest.raises(ValueError, match='escapes working directory'):
    resolve_sandbox_path(path, '/workspace')


@pytest.mark.parametrize('path', ['/etc/passwd', '/workspace/../../etc/passwd'])
def test_resolve_sandbox_path_passes_absolute_paths_through(path):
  """Absolute paths follow the BaseEnvironment contract and pass through."""
  assert resolve_sandbox_path(path, '/workspace') == path


def test_resolve_sandbox_path_accepts_pathlike():
  assert resolve_sandbox_path(Path('notes.txt'), '/workspace') == (
      '/workspace/notes.txt'
  )

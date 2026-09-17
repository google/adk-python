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

"""Tests for the lazy package accessors."""

from __future__ import annotations

import importlib
import types

from google.adk.utils import _lazy
import pytest


@pytest.fixture(name='package')
def _package(tmp_path, monkeypatch) -> types.ModuleType:
  """A package with a plain submodule and one that needs an absent library."""
  root = tmp_path / 'lazy_fixture_pkg'
  root.mkdir()
  (root / '__init__.py').write_text('')
  (root / 'plain.py').write_text('VALUE = 1\n')
  (root / 'needs_absent.py').write_text('import absent_dependency\n')
  (root / 'exports.py').write_text('Exported = object()\n')
  monkeypatch.syspath_prepend(str(tmp_path))
  return importlib.import_module('lazy_fixture_pkg')


def test_declared_member_resolves_from_its_module(package):
  getattr_, _ = _lazy.accessors(vars(package), {'Exported': '.exports'})

  exports = importlib.import_module('lazy_fixture_pkg.exports')
  assert getattr_('Exported') is exports.Exported


def test_submodule_resolves_as_an_attribute(package):
  getattr_, _ = _lazy.accessors(vars(package), {})

  assert getattr_('plain').VALUE == 1


def test_unknown_name_raises_attribute_error(package):
  getattr_, _ = _lazy.accessors(vars(package), {})

  with pytest.raises(AttributeError, match='no attribute'):
    getattr_('not_a_submodule')


def test_absent_dependency_keeps_its_own_error(package):
  """A library missing inside a submodule must not read as a typo."""
  getattr_, _ = _lazy.accessors(vars(package), {})

  with pytest.raises(ModuleNotFoundError) as error:
    getattr_('needs_absent')

  assert error.value.name == 'absent_dependency'

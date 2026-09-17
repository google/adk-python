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

"""Tests for the optional-dependency helper."""

from __future__ import annotations

from google.adk.utils._dependency import missing_extra
import pytest


@pytest.mark.parametrize(
    ('package', 'extra'),
    [('sqlalchemy', 'db'), ('a2a-sdk', 'a2a')],
)
def test_missing_extra_names_the_package_and_the_install_command(
    package, extra
):
  error = missing_extra(package, extra)

  # Callers surface this straight to the user, so it has to name the missing
  # package and the exact command that installs it.
  assert str(error) == (
      f"The '{package}' package is required to use this feature. Please"
      f' install it by running: pip install google-adk[{extra}]'
  )


def test_missing_extra_returns_the_error_for_the_caller_to_raise():
  # Callers do `raise missing_extra(...) from e`, so the helper must hand
  # back an ImportError rather than raising one itself.
  error = missing_extra('vertexai', 'gcp')

  assert isinstance(error, ImportError)
  with pytest.raises(ImportError, match='vertexai'):
    raise error

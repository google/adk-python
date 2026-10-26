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

"""Helper for optional dependencies and packaging extras."""

from __future__ import annotations


def missing_extra(package: str, extra: str) -> ImportError:
  """Returns an ImportError with a standard message for a missing extra.

  Args:
    package: The name of the package that failed to import (e.g., 'vertexai').
    extra: The name of the extra group required to install it (e.g., 'gcp').
  """
  return ImportError(
      f"The '{package}' package is required to use this feature. "
      f"Please install it by running: pip install google-adk[{extra}]"
  )


def require_agentplatform_attributes(
    obj: object, names: tuple[str, ...], feature: str
) -> None:
  """Raises ImportError if an agentplatform object lacks any of `names`.

  google-cloud-aiplatform releases before 2.2 install their own agentplatform
  package, whose client and types predate the attributes `feature` uses.

  Args:
    obj: An agentplatform client, or the agentplatform.types module.
    names: The attributes `feature` uses.
    feature: The ADK class or command that needs them, named in the error.
  """
  missing = [name for name in names if not hasattr(obj, name)]
  if missing:
    raise ImportError(
        f"{feature} needs google-cloud-agentplatform 2.2 or later, but the"
        f" agentplatform package that loaded has no {', '.join(missing)}."
        " google-cloud-aiplatform releases before 2.2 install their own copy"
        " of that package. Upgrade both with: pip install -U"
        ' "google-cloud-aiplatform>=2.2,<3"'
        ' "google-cloud-agentplatform>=2.2,<3"'
    )

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

"""Does a google.adk.* name exist in the ADK installed right here.

Shared by the config writer and the file writer, which both need to refuse a
reference to something this version does not have. It lives in utils because
those two already import each other; a copy in either would be a cycle.
"""

from __future__ import annotations

import importlib
from typing import Any


def adk_symbol_exists(name: str) -> bool:
  """Whether a google.adk.* name resolves in the installed package.

  Import the longest module prefix, then walk the attributes. Doing it by
  import rather than by matching against a list of known names is what makes
  aliases and re-exports resolve, exactly as the loader will see them.
  """
  parts = name.split(".")
  for split in range(len(parts), 0, -1):
    try:
      obj: Any = importlib.import_module(".".join(parts[:split]))
    except ModuleNotFoundError as e:
      # "exists but an optional extra is not installed" is not the same as
      # "does not exist", and must not be reported as a bad reference.
      if e.name and not ".".join(parts[:split]).startswith(e.name):
        return True
      continue
    except ImportError:
      return True
    except Exception:  # pylint: disable=broad-except
      continue
    for attribute in parts[split:]:
      try:
        obj = getattr(obj, attribute)
      except AttributeError:
        return False
      except Exception:  # pylint: disable=broad-except
        return True  # gated behind an optional extra
    return True
  return False

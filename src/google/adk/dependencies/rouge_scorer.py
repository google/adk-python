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

from __future__ import annotations

import re
import sys
from typing import Any

# NLTK (a subdependency of rouge-score) attempts to import 'regex'.
# If 'regex' is not installed or blocked from cwd on CI runners, provide
# a fallback wrapper that delegates to standard 're' so NLTK operates.
if "regex" not in sys.modules:
  try:
    import regex  # type: ignore # pylint: disable=g-import-not-at-top
  except Exception:

    class _RegexFallback:

      def __getattr__(self, name: str) -> Any:
        return getattr(re, name, 0)

    sys.modules["regex"] = _RegexFallback()  # type: ignore

try:
  from rouge_score import rouge_scorer
except Exception:
  rouge_scorer = None

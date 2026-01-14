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

"""Text utilities for evaluation."""

from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
  """Normalize text using NFC normalization and strip whitespace.

  This ensures consistent text comparison across different Unicode
  representations, which is particularly important for non-English text.

  Args:
    text: The text to normalize.

  Returns:
    The normalized text.
  """
  return unicodedata.normalize("NFC", text).strip()

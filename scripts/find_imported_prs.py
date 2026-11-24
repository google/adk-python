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

"""Find the pull requests a push to main imports, for copybara-pr-handler.yml.

An import writes one `Merge https://github.com/google/adk-python/pull/<N>` line
into the commit message and copies the PR's title and body in beside it. Only
that whole line names the PR, and only when the message holds exactly one.

Usage: find_imported_prs.py <push event JSON>. Prints `prs=<JSON list>` for
`$GITHUB_OUTPUT`.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import json
import re
import sys

_MERGE_LINE = re.compile(
    r"^Merge https://github\.com/google/adk-python/pull/(\d+)$", re.MULTILINE
)


def imported_pr(message: str) -> int | None:
  """The PR a commit message's one import line names, or None."""
  numbers = _MERGE_LINE.findall(message)
  return int(numbers[0]) if len(numbers) == 1 else None


def prs_to_close(
    commits: Sequence[Mapping[str, object]],
) -> list[dict[str, int | str]]:
  """Each PR the pushed commits import, with the commit's short SHA."""
  found: list[dict[str, int | str]] = []
  for commit in commits:
    sha = str(commit.get("id") or "")[:7]
    pr = imported_pr(str(commit.get("message") or ""))
    print(f"{sha}: imports PR {pr}", file=sys.stderr)
    if pr is not None:
      found.append({"prNumber": pr, "commitSha": sha})
  return found


def main(argv: list[str]) -> int:
  with open(argv[1], encoding="utf-8") as f:
    commits = json.load(f).get("commits") or []
  print(f"prs={json.dumps(prs_to_close(commits))}")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))

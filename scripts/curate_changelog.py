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

"""Insert a Highlights section into the newest CHANGELOG.md release.

Run as a post-step after release-please in the "Release: Cut" workflow. It finds
the newest version section, drafts a Highlights block from that version's commit
entries with Gemini, and inserts it at the top of the section.

If no API key is available or the model call fails, it inserts an empty
Highlights template instead, so the release flow never hard-fails on curation.
The release manager edits the result in the PR before merging.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

_HIGHLIGHTS_HEADER = "### Highlights"

# Matches a release header line, e.g. "## [2.4.0](https://...) (2026-06-29)".
_VERSION_RE = re.compile(r"^## \[")

# Inserted verbatim when the model is unavailable, so the release manager has a
# scaffold to fill in by hand. Mirrors the format the model is asked to produce.
_TEMPLATE = """### Highlights

<one sentence describing the theme of this release>

* **<Feature>**: <what it unlocks for the user, in one line>. (<commit>)
* **<Feature>**: <user benefit>. (<commit>)

#### Breaking changes

* **<what changed>**: <how to migrate, in one line>. (<commit>)
"""

_PROMPT = """\
You are drafting the "Highlights" section of an ADK (Agent Development Kit)
Python release changelog.

Below is the auto-generated changelog for the new version, grouped by type
(Features, Bug Fixes, etc.). Each entry ends with a commit hash link.

Write a short Highlights section so a reader can grasp the release at a glance:
- Start with ONE sentence describing the theme of the release.
- Then 2-5 bullets, each leading with the user-facing benefit rather than the
  implementation, formatted as
  "* **<Area>**: <benefit in one line>. (<commit link>)".
- Reuse the exact commit links from the entries you summarize.
- Pick only the few changes that matter most to users. Ignore pure refactors,
  chores, and trivial docs.
- If there are breaking changes, add a "#### Breaking changes" subsection after
  the bullets, each with a one-line migration note.

Output ONLY the markdown body. Do NOT include the "### Highlights" header and do
NOT wrap the output in code fences.

Changelog for the new version:

{changelog}
"""


def _find_latest_section(lines: list[str]) -> tuple[int, int] | None:
  """Returns the [start, end) line span of the newest release section.

  start is the index of the "## [" header; end is the index of the next "## ["
  header or len(lines). Returns None if no release header is present.
  """
  start = None
  for i, line in enumerate(lines):
    if _VERSION_RE.match(line):
      start = i
      break
  if start is None:
    return None
  end = len(lines)
  for j in range(start + 1, len(lines)):
    if _VERSION_RE.match(lines[j]):
      end = j
      break
  return start, end


def _draft_highlights(section_text: str, *, model: str) -> str | None:
  """Drafts the Highlights body with Gemini, or None if unavailable."""
  api_key = os.environ.get("GOOGLE_API_KEY")
  if not api_key:
    print("GOOGLE_API_KEY not set; skipping model drafting.")
    return None
  try:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=_PROMPT.format(changelog=section_text),
    )
    body = (response.text or "").strip()
    return body or None
  # The release must never fail because drafting failed (missing dependency,
  # network/API error, quota); fall back to the template in every case.
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Highlights drafting failed ({e!r}); falling back to template.")
    return None


def _build_block(body: str) -> str:
  """Wraps a model-drafted body in the Highlights header."""
  body = body.strip()
  if body.startswith(_HIGHLIGHTS_HEADER):
    body = body[len(_HIGHLIGHTS_HEADER) :].lstrip("\n")
  return f"{_HIGHLIGHTS_HEADER}\n\n{body}\n"


def curate(text: str, *, model: str) -> str:
  """Returns CHANGELOG text with Highlights inserted into the newest release."""
  lines = text.splitlines(keepends=True)
  span = _find_latest_section(lines)
  if span is None:
    print("No release section found; leaving CHANGELOG unchanged.")
    return text
  start, end = span
  if any(line.strip() == _HIGHLIGHTS_HEADER for line in lines[start:end]):
    print("Highlights already present; leaving CHANGELOG unchanged.")
    return text

  body = _draft_highlights("".join(lines[start:end]), model=model)
  if body is None:
    block = _TEMPLATE
    print("Inserted Highlights template.")
  else:
    block = _build_block(body)
    print("Inserted model-drafted Highlights.")

  # Insert before the first "### " subsection (Features, Bug Fixes, ...), or at
  # the end of the section if it has no categorized entries.
  insert_at = end
  for i in range(start + 1, end):
    if lines[i].startswith("### "):
      insert_at = i
      break
  block_text = block.rstrip("\n") + "\n\n"
  return "".join(lines[:insert_at] + [block_text] + lines[insert_at:])


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--changelog",
      default="CHANGELOG.md",
      help="Path to the changelog file to curate.",
  )
  parser.add_argument(
      "--model",
      default=os.environ.get("CHANGELOG_CURATION_MODEL", "gemini-2.5-flash"),
      help="Gemini model used to draft the Highlights.",
  )
  args = parser.parse_args()

  with open(args.changelog, encoding="utf-8") as f:
    text = f.read()
  updated = curate(text, model=args.model)
  if updated == text:
    return 0
  with open(args.changelog, "w", encoding="utf-8") as f:
    f.write(updated)
  print(f"Updated {args.changelog}.")
  return 0


if __name__ == "__main__":
  sys.exit(main())

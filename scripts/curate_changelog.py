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

"""Curate the newest CHANGELOG.md release section during a release cut.

Runs as a post-step after release-please in the "Release: Cut" workflow and
commits the result back to the release PR branch. It does three things to the
newest version section, in order:

1. Deterministic cleanup of the entries (no model): unescape HTML entities
   (``&gt;=`` -> ``>=``), de-link accidental ``@mentions`` that release-please
   auto-linked from a commit subject, drop duplicate entries (the same change
   landed under several commits), and lowercase the leading word so entries
   read as consistent imperative phrases.
2. Draft a short "Highlights" block with Gemini and place it above the fold, so
   a reader grasps the release in a handful of bullets.
3. For large releases, collapse the full categorized list under a ``<details>``
   fold so the notes read short while remaining a complete record.

Every step is best-effort. If the model is unavailable the Highlights fall back
to a template; the deterministic passes never call the network. The file is only
rewritten when something changed, so it is safe to re-run on each release-please
regenerate (idempotent). The release manager edits the result in the PR before
merging.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys

_HIGHLIGHTS_HEADER = "### Highlights"
_DETAILS_SUMMARY = "<summary>All changes</summary>"

# Matches a release header line, e.g. "## [2.4.0](https://...) (2026-06-29)".
_VERSION_RE = re.compile(r"^## \[")
# Matches a category header, e.g. "### Features", "### Bug Fixes".
_SUBSECTION_RE = re.compile(r"^### ")
# Matches a changelog entry bullet.
_ENTRY_RE = re.compile(r"^\s*\* ")
# Trailing " ([abc1234](url))..." on an entry; stripped only to build the dedupe
# key so two commits with the same subject collapse to one.
_TRAILER_RE = re.compile(r"\s*\(\[[0-9a-f]{6,}\]\(.*$")
# An accidental "[@name](https://github.com/name)" auto-link, produced when a
# commit subject contained a bare "@name" (e.g. "... in @node decorator").
_MENTION_RE = re.compile(r"\[@([\w-]+)\]\(https://github\.com/\1\)")
# "* " then an optional bold "**scope:** " prefix, then the first word and rest.
_LEAD_RE = re.compile(
    r"(?P<head>\s*\* (?:\*\*[^*]+\*\* )?)(?P<first>\w+)(?P<rest>.*)", re.S
)

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


def _latest_section_text(text: str) -> str | None:
  """Returns the text of the newest release section, or None if absent."""
  lines = text.splitlines(keepends=True)
  span = _find_latest_section(lines)
  if span is None:
    return None
  start, end = span
  return "".join(lines[start:end]).strip("\n") + "\n"


def _normalize_entry(line: str) -> str:
  """Applies deterministic, meaning-preserving fixes to a single entry line."""
  s = html.unescape(line)  # &gt;= -> >=, &amp; -> &, etc.
  s = _MENTION_RE.sub(r"`@\1`", s)  # de-link an accidental @mention
  m = _LEAD_RE.match(s)
  if m:
    first = m.group("first")
    # Lowercase a plain leading word ("Fix" -> "fix") but leave acronyms and
    # camelCase/proper nouns intact ("OAuth", "GPU", "iOS", "A2A").
    if not any(c.isupper() for c in first[1:]):
      first = first[0].lower() + first[1:]
    s = f"{m.group('head')}{first}{m.group('rest')}"
  return s


def _dedupe_key(line: str) -> str:
  """Key for detecting the same change landed under multiple commits."""
  core = _TRAILER_RE.sub("", line)  # drop the "([hash](url))" trailer
  return re.sub(r"\s+", " ", core).strip().lower()


def _normalize_body(lines: list[str]) -> list[str]:
  """Normalizes and de-duplicates entry bullets; passes other lines through."""
  seen: set[str] = set()
  out: list[str] = []
  for line in lines:
    if _ENTRY_RE.match(line):
      norm = _normalize_entry(line)
      key = _dedupe_key(norm)
      if key in seen:
        continue
      seen.add(key)
      out.append(norm)
    else:
      out.append(line)
  return out


def _count_entries(lines: list[str]) -> int:
  return sum(1 for line in lines if _ENTRY_RE.match(line))


def _wrap_in_details(body_lines: list[str]) -> str:
  """Collapses the categorized list under a <details> fold."""
  inner = "".join(body_lines).strip("\n")
  return f"<details>\n{_DETAILS_SUMMARY}\n\n{inner}\n\n</details>\n"


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


def curate(text: str, *, model: str, fold_threshold: int) -> str:
  """Returns CHANGELOG text with the newest release section curated."""
  lines = text.splitlines(keepends=True)
  span = _find_latest_section(lines)
  if span is None:
    print("No release section found; leaving CHANGELOG unchanged.")
    return text
  start, end = span

  section = lines[start:end]
  if any(line.strip() == _HIGHLIGHTS_HEADER for line in section) or any(
      _DETAILS_SUMMARY in line for line in section
  ):
    print("Section already curated; leaving CHANGELOG unchanged.")
    return text

  # Split the section into its header (## [..] + blank lines) and the
  # categorized body (### Features ... through the end of the section).
  first_sub = None
  for i in range(start + 1, end):
    if _SUBSECTION_RE.match(lines[i]):
      first_sub = i
      break

  if first_sub is None:
    # No categorized entries (rare): only add Highlights.
    header = section
    body_norm: list[str] = []
    model_input = ""
  else:
    header = lines[start:first_sub]
    body_norm = _normalize_body(lines[first_sub:end])
    model_input = "".join(body_norm)

  drafted = _draft_highlights(model_input, model=model) if model_input else None
  if drafted is None:
    highlights = _TEMPLATE
    print("Inserted Highlights template.")
  else:
    highlights = _build_block(drafted)
    print("Inserted model-drafted Highlights.")

  parts: list[str] = list(header)
  if parts and parts[-1].strip():
    parts.append("\n")
  parts.append(highlights.rstrip("\n") + "\n\n")

  if body_norm:
    if _count_entries(body_norm) > fold_threshold:
      parts.append(_wrap_in_details(body_norm))
      print(f"Folded {_count_entries(body_norm)} entries under <details>.")
    else:
      parts.append("".join(body_norm).strip("\n") + "\n")

  new_section = "".join(parts).rstrip("\n") + "\n\n"
  return "".join(lines[:start]) + new_section + "".join(lines[end:])


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
  parser.add_argument(
      "--fold-threshold",
      type=int,
      default=int(os.environ.get("CHANGELOG_FOLD_THRESHOLD", "12")),
      help=(
          "Collapse the full list under a <details> fold when the release has"
          " more than this many entries. Set very high to never fold."
      ),
  )
  parser.add_argument(
      "--section-out",
      default=None,
      help=(
          "If set, write the curated newest release section to this path, for"
          " use as the PR description body. Written even when the changelog"
          " file is otherwise unchanged."
      ),
  )
  args = parser.parse_args()

  with open(args.changelog, encoding="utf-8") as f:
    text = f.read()
  updated = curate(text, model=args.model, fold_threshold=args.fold_threshold)

  if args.section_out:
    section = _latest_section_text(updated)
    if section is not None:
      with open(args.section_out, "w", encoding="utf-8") as f:
        f.write(section)
      print(f"Wrote latest section to {args.section_out}.")

  if updated == text:
    return 0
  with open(args.changelog, "w", encoding="utf-8") as f:
    f.write(updated)
  print(f"Updated {args.changelog}.")
  return 0


if __name__ == "__main__":
  sys.exit(main())

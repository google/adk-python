#!/usr/bin/env python3
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

"""Check that URLs referenced from llms.txt resolve successfully."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

URL_RE = re.compile(r"https?://[^\s)\]>]+")


def extract_urls(path: Path) -> list[str]:
  """Returns sorted unique URLs from a markdown/text file."""
  urls = set(URL_RE.findall(path.read_text(encoding="utf-8")))
  return sorted(url for url in urls if ".git@" not in url)


def check_url(url: str, timeout: float) -> str | None:
  request = urllib.request.Request(
      url,
      method="GET",
      headers={"User-Agent": "adk-python-llms-link-checker"},
  )
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      if response.status == 200:
        return None
      return f"{response.status} {response.reason}"
  except urllib.error.HTTPError as exc:
    return f"{exc.code} {exc.reason}"
  except urllib.error.URLError as exc:
    return str(exc.reason)


def main() -> int:
  parser = argparse.ArgumentParser(
      description="Validate that every URL in llms.txt returns HTTP 200."
  )
  parser.add_argument(
      "path",
      nargs="?",
      default=Path("llms.txt"),
      type=Path,
      help="Path to llms.txt.",
  )
  parser.add_argument("--timeout", default=20.0, type=float)
  args = parser.parse_args()

  urls = extract_urls(args.path)
  failures: list[tuple[str, str]] = []
  for url in urls:
    failure = check_url(url, args.timeout)
    if failure:
      failures.append((url, failure))

  if failures:
    for url, failure in failures:
      print(f"FAILED {failure}: {url}", file=sys.stderr)
    print(f"FAILED: {len(failures)} broken URLs out of {len(urls)}")
    return 1

  print(f"OK: {len(urls)} URLs checked")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

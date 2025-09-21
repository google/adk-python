#!/usr/bin/env python3
"""
build_llms_txt.py – produce llms.txt and llms-full.txt
                   – skips ```java``` blocks
                   – README can be next to docs/ or inside docs/
                   – includes Python API reference from HTML files
                   – includes adk-python repository README
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import textwrap
from typing import List
from typing import Tuple
import urllib.error
import urllib.request

RE_JAVA = re.compile(r"```java[ \t\r\n][\s\S]*?```", re.I | re.M)
RE_SNIPPET = re.compile(r"^(\s*)--8<--\s+\"([^\"]+?)(?::([^\"]+))?\"$", re.M)


def fetch_adk_python_readme() -> str:
  """Fetch README content from adk-python repository"""
  try:
    url = "https://raw.githubusercontent.com/google/adk-python/main/README.md"
    with urllib.request.urlopen(url) as response:
      return response.read().decode("utf-8")
  except (urllib.error.URLError, urllib.error.HTTPError) as e:
    print(f"Warning: Could not fetch adk-python README: {e}")
    return ""


def strip_java(md: str) -> str:
  return RE_JAVA.sub("", md)


def first_heading(md: str) -> str | None:
  for line in md.splitlines():
    if line.startswith("#"):
      return line.lstrip("#").strip()
  return None


def md_to_text(md: str) -> str:
  import bs4
  import markdown

  html = markdown.markdown(
      md, extensions=["fenced_code", "tables", "attr_list"]
  )
  return bs4.BeautifulSoup(html, "html.parser").get_text("\n")


def html_to_text(html_file: Path) -> str:
  """Extract text content from HTML files (for Python API reference)"""
  import bs4

  try:
    html_content = html_file.read_text(encoding="utf-8")
    soup = bs4.BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
      script.decompose()

    # Get text and clean it up
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text
  except Exception as e:
    print(f"Warning: Could not process {html_file}: {e}")
    return ""


def count_tokens(text: str, model: str = "cl100k_base") -> int:
  try:
    import tiktoken

    return len(tiktoken.get_encoding(model).encode(text))
  except Exception:
    return len(text.split())


def expand_code_snippets(content: str, project_root: Path) -> str:
  """
  Expands code snippets marked with --8<-- "path/to/file.py" or
  --8<-- "path/to/file.py:section_name" into the content.
  """

  def replace_snippet(match):
    indent = match.group(1)  # Capture leading spaces
    snippet_path_str = match.group(
        2
    )  # Capture the file path (e.g., "examples/python/snippets/file.py")
    section_name = match.group(
        3
    )  # Capture the section name if present (e.g., "init")
    snippet_full_path = (
        project_root / snippet_path_str
    )  # Changed from base_path to project_root

    # If not found in project root, try adk-docs directory
    if not snippet_full_path.exists():
      script_dir = Path(__file__).resolve().parent
      adk_docs_path = script_dir / "adk-docs" / snippet_path_str
      if adk_docs_path.exists():
        snippet_full_path = adk_docs_path

    if snippet_full_path.exists():
      try:
        file_content = snippet_full_path.read_text(encoding="utf-8")
        if section_name:
          # Extract content based on section markers
          # Handle both single and double hash markers with optional spacing
          start_marker_patterns = [
              f"# --8<-- [start:{section_name.strip()}]",
              f"## --8<-- [start:{section_name.strip()}]",
          ]
          end_marker_patterns = [
              f"# --8<-- [end:{section_name.strip()}]",
              f"## --8<-- [end:{section_name.strip()}]",
              f"##  --8<-- [end:{section_name.strip()}]",  # Handle extra space
          ]

          start_index = -1
          end_index = -1

          # Find start marker
          for pattern in start_marker_patterns:
            start_index = file_content.find(pattern)
            if start_index != -1:
              start_marker = pattern
              break

          # Find end marker
          for pattern in end_marker_patterns:
            end_index = file_content.find(pattern)
            if end_index != -1:
              break

          if start_index != -1 and end_index != -1 and start_index < end_index:
            # Adjust start_index to begin immediately after the start_marker
            start_of_code = start_index + len(start_marker)
            temp_content = file_content[start_of_code:end_index]
            lines = temp_content.splitlines(keepends=True)
            extracted_lines = []
            for line in lines:
              if (
                  not line.strip().startswith("# --8<--")
                  and not line.strip().startswith("## --8<--")
                  and line.strip() != ""
              ):
                extracted_lines.append(line)
            extracted_content = "".join(extracted_lines).strip("\n")

            return textwrap.indent(extracted_content, indent)
          else:
            print(
                f"Warning: Section '{section_name}' not found or markers"
                f" malformed in {snippet_full_path}"
            )
            return match.group(0)
        else:
          # Read entire file if no section name
          return textwrap.indent(file_content, indent)
      except Exception as e:
        print(f"Warning: Could not read snippet file {snippet_full_path}: {e}")
        return match.group(0)
    else:
      print(f"Warning: Snippet file not found: {snippet_full_path}")
      return match.group(0)

  expanded_content = RE_SNIPPET.sub(replace_snippet, content)
  return expanded_content


# ---------- index (llms.txt) ----------
def build_index(docs: Path) -> str:
  # Locate README
  for cand in (docs / "README.md", docs.parent / "README.md"):
    if cand.exists():
      readme = cand.read_text(encoding="utf-8")
      break
  else:
    sys.exit("README.md not found in docs/ or its parent")

  title = first_heading(readme) or "Documentation"
  summary = md_to_text(readme).split("\n\n")[0]
  lines = [f"# {title}", "", f"> {summary}", ""]

  # Add adk-python repository README content
  adk_readme = fetch_adk_python_readme()
  if adk_readme:
    lines.append("## ADK Python Repository")
    lines.append("")
    # Include the full README content, properly formatted
    adk_text = md_to_text(strip_java(adk_readme))
    lines.append(adk_text)
    lines.append("")
    lines.append(
        f"**Source:** [adk-python"
        f" repository](https://github.com/google/adk-python)"
    )
    lines.append("")

  primary: List[Tuple[str, str]] = []
  secondary: List[Tuple[str, str]] = []

  # Process Markdown files
  for md in sorted(docs.rglob("*.md")):
    # Skip Java API reference files
    if "api-reference" in md.parts and "java" in md.parts:
      continue

    rel = md.relative_to(docs)
    # Construct the correct GitHub URL for the Markdown file
    url = f"https://github.com/google/adk-docs/blob/main/docs/{rel}".replace(
        " ", "%20"
    )
    h = first_heading(strip_java(md.read_text(encoding="utf-8"))) or rel.stem
    (
        secondary
        if "sample" in rel.parts or "tutorial" in rel.parts
        else primary
    ).append((h, url))

  # Add Python API reference
  python_api_dir = docs / "api-reference" / "python"
  if python_api_dir.exists():
    primary.append((
        "Python API Reference",
        "https://github.com/google/adk-docs/blob/main/docs/api-reference/python/",
    ))

  def emit(name: str, items: List[Tuple[str, str]]):
    nonlocal lines
    if items:
      lines.append(f"## {name}")
      lines += [f"- [{h}]({u})" for h, u in items]
      lines.append("")

  emit("Documentation", primary)
  emit("Optional", secondary)
  return "\n".join(lines)


# ---------- full corpus ----------
def build_full(docs: Path) -> str:
  out = []

  script_dir = Path(__file__).resolve().parent
  project_root = script_dir.parents[2]  # Correct project root
  print(f"DEBUG: Project Root: {project_root}")
  print(f"DEBUG: Docs Dir: {docs}")

  # Add adk-python repository README content at the beginning
  adk_readme = fetch_adk_python_readme()
  if adk_readme:
    # Expand snippets in README if any
    expanded_adk_readme = expand_code_snippets(
        strip_java(adk_readme), project_root
    )  # Pass project_root
    out.append("# ADK Python Repository")
    out.append("")
    out.append(expanded_adk_readme)  # Use expanded content
    out.append("")
    out.append("---")
    out.append("")

  # Process Markdown files
  for md in sorted(docs.rglob("*.md")):
    # Skip Java API reference files
    if "api-reference" in md.parts and "java" in md.parts:
      continue

    md_content = md.read_text(encoding="utf-8")
    print(f"DEBUG: Processing markdown file: {md.relative_to(docs)}")
    expanded_md_content = expand_code_snippets(
        strip_java(md_content), project_root
    )  # Changed back to project_root
    out.append(expanded_md_content)  # Use expanded content

  # Process Python API reference HTML files
  python_api_dir = docs / "api-reference" / "python"
  if python_api_dir.exists():
    # Add a separator and header for Python API reference
    out.append("\n\n# Python API Reference\n")

    # Process main HTML files (skip static assets and generated files)
    html_files = [
        python_api_dir / "index.html",
        python_api_dir / "google-adk.html",
        python_api_dir / "genindex.html",
        python_api_dir / "py-modindex.html",
    ]

    for html_file in html_files:
      if html_file.exists():
        text = html_to_text(html_file)
        if text.strip():
          out.append(f"\n## {html_file.stem}\n")
          out.append(text)

  return "\n\n".join(out)


def main() -> None:
  ap = argparse.ArgumentParser(
      description="Generate llms.txt / llms-full.txt",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  ap.add_argument("--docs-dir", required=True, type=Path)
  ap.add_argument("--out-root", default=Path("."), type=Path)
  ap.add_argument("--index-limit", type=int, default=50_000)
  ap.add_argument("--full-limit", type=int, default=500_000)
  args = ap.parse_args()

  idx, full = build_index(args.docs_dir), build_full(args.docs_dir)
  if (tok := count_tokens(idx)) > args.index_limit:
    sys.exit(f"Index too big: {tok:,}")
  if (tok := count_tokens(full)) > args.full_limit:
    sys.exit(f"Full text too big: {tok:,}")

  (args.out_root / "llms.txt").write_text(idx, encoding="utf-8")
  (args.out_root / "llms-full.txt").write_text(full, encoding="utf-8")
  print("✅ Generated llms.txt and llms-full.txt successfully")
  print(f"llms.txt tokens: {count_tokens(idx)}")
  print(f"llms-full.txt tokens: {count_tokens(full)}")


if __name__ == "__main__":
  main()

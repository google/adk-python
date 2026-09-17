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

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

SKIP_ANNOTATION = "<!-- verify-snippets: ignore -->"
SNIPPET_TIMEOUT = 120  # seconds; adjust if snippets legitimately need longer

# Must match COV_SECTION_HEADER in run.py exactly — used to split coverage output
# from the main execution log when parsing run.py's stdout.
COV_SECTION_HEADER = "📊 Phase 4: Code Coverage Report"

# Structured exit codes from run.py — kept in sync with run.py:EXIT_* constants.
# Using exit codes (not string/emoji matching) makes classification robust to
# future changes in run.py's human-readable output text.
EXIT_SUCCESS = 0  # All phases passed
EXIT_LOAD_FAILURE = 1  # Failed to compile / load the snippet
EXIT_RUN_FAILURE = 2  # Loaded OK but the ADK component failed at runtime
EXIT_NO_COMPONENT = 3  # Loaded OK, no runnable ADK component found (load-only)


def extract_snippets(md_path: Path):
  """Parses a markdown file and extracts python code blocks along with their preceding headings.

  A code block immediately preceded by the HTML comment
  ``<!-- verify-snippets: ignore -->`` is recorded but marked as skipped so
  that illustrative / pseudo-code examples are excluded from execution.
  """
  with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

  lines = content.splitlines()
  snippets = []
  current_heading = "Top Level"
  in_code_block = False
  code_lines = []
  skip_next_block = False

  for line in lines:
    # If we are inside a code block, handle it first to preserve comments starting with '#'
    if in_code_block:
      stripped = line.strip()
      # Close only on a bare closing fence (``` with no language specifier).
      # A fenced block of another language (e.g. ```bash) appearing *inside*
      # the Python block will not trigger this branch because it carries a
      # language tag, so it is appended to code_lines as literal content.
      if stripped == "```":
        in_code_block = False
        code_text = "\n".join(code_lines)
        snippets.append({
            "heading": current_heading,
            "code": code_text,
            "skip": skip_next_block,
        })
        skip_next_block = False
      else:
        code_lines.append(line)
      continue

    # If we are outside a code block, check for headings or code block starts
    if line.startswith("#"):
      # Clean up heading markers (e.g., "## Get started" -> "Get started")
      current_heading = line.lstrip("#").strip()
      # A heading between the annotation and the fence cancels the skip.
      skip_next_block = False
      continue

    if line.strip() == SKIP_ANNOTATION:
      skip_next_block = True
      continue

    if line.strip().startswith("```python"):
      in_code_block = True
      code_lines = []
      continue

    # Any other non-empty line (prose, blank-line-separated text, etc.) between
    # the annotation and the fence cancels the skip.
    if line.strip():
      skip_next_block = False

  return snippets


def run_snippet(run_py_path: Path, snippet_path: Path):
  """Executes run.py on the isolated snippet and returns the result."""
  # Run using the same Python interpreter as this script (which will be the venv's python)
  cmd = [sys.executable, str(run_py_path), str(snippet_path)]

  # Ensure GEMINI_API_KEY is preferred if both keys are set in the environment
  env = os.environ.copy()
  if "GOOGLE_API_KEY" in env and "GEMINI_API_KEY" in env:
    env.pop("GOOGLE_API_KEY", None)

  try:
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=SNIPPET_TIMEOUT
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
  except subprocess.TimeoutExpired:
    return {
        "exit_code": EXIT_RUN_FAILURE,
        "stdout": (
            "❌ Run Failure: Snippet execution timed out after"
            f" {SNIPPET_TIMEOUT} seconds."
        ),
        "stderr": (
            "TimeoutExpired: The snippet process did not complete within the"
            f" {SNIPPET_TIMEOUT}-second limit."
        ),
    }


def extract_error_detail(stdout: str, stderr: str) -> str:
  """Extracts the most relevant error line from run.py's output.

  Searches in order:
  1. Last line in stderr that looks like a Python exception (``<Name>Error:``
     or ``<Name>Exception:``). Scoping to stderr avoids matching runner prose
     in stdout (e.g. "❌ Run Failure: ...") which contains words like "Failure"
     but is not an exception line.
  2. Last line in stdout with the same pattern, as a fallback for runtimes that
     write tracebacks to stdout instead of stderr.
  3. Last line in stderr matching the generic ``<ClassName>: <detail>`` format
     (custom exception classes that don't end in Error/Exception).
  4. Fallback string if nothing matches.
  """
  # Matches standard Python exception class names: ends in 'Error' or 'Exception',
  # followed by a colon and detail text.  Anchored to the start of the stripped line
  # so runner prose ("❌ Run Failure: ...") is not matched.
  _exception_re = re.compile(r"^[A-Za-z]\w*(?:Error|Exception|Warning):\s*.+")

  for source in (stderr, stdout):
    for line in reversed(source.splitlines()):
      if _exception_re.match(line.strip()):
        return f"`{line.strip()}`"

  # Pass 3: generic '<ClassName>: <detail>' in stderr only
  for line in reversed(stderr.splitlines()):
    if re.match(r"^[A-Za-z]\w*:.+", line.strip()):
      return f"`{line.strip()}`"

  return "Failed to compile/load."


def clean_name(name: str):
  """Sanitizes a string to be a safe filename."""
  name = name.lower().replace(" ", "_")
  return re.sub(r"[^a-z0-9_]", "", name)


def md_cell(value: str) -> str:
  """Escapes pipe characters so the value is safe inside a Markdown table cell."""
  return value.replace("|", r"\|")


def safe_fence(content: str, language: str = "") -> str:
  """Returns a Markdown fenced code block that safely wraps *content*.

  Picks the shortest fence (minimum three backticks) that is strictly longer
  than any contiguous run of backticks found inside *content*, so the fence
  cannot be prematurely closed by content that itself contains backtick runs.
  This is the approach recommended by the CommonMark spec.

  Example::

      safe_fence("x = ```foo```", "python")
      # returns:
      # ````python
      # x = ```foo```
      # ````
  """
  # Find the longest run of backticks inside the content
  max_run = max(
      (len(m.group()) for m in re.finditer(r"`+", content)), default=0
  )
  # The outer fence must be strictly longer, and at least 3 characters
  fence_len = max(3, max_run + 1)
  fence = "`" * fence_len
  tag = f"{fence}{language}\n" if language else f"{fence}\n"
  return f"{tag}{content}\n{fence}"


def main():
  parser = argparse.ArgumentParser(description="Markdown Snippet Verifier")
  parser.add_argument(
      "file", type=str, help="Path to the markdown file to verify"
  )
  args = parser.parse_args()

  md_path = Path(args.file).resolve()
  if not md_path.exists():
    print(f"❌ Error: Markdown file '{md_path}' does not exist.")
    sys.exit(1)

  # Locate run.py bundled inside the same scripts folder as verify_md.py (portable mode!)
  run_py_path = Path(__file__).parent / "run.py"
  if not run_py_path.exists():
    print(f"❌ Error: Bundled runner 'run.py' not found at '{run_py_path}'.")
    sys.exit(1)

  print(f"🔬 Analyzing Markdown: {md_path.name}")

  # 1. Extract snippets
  snippets = extract_snippets(md_path)
  if not snippets:
    print(f"⚠️  No python code blocks found in '{md_path.name}'.")
    sys.exit(0)

  print(f"📋 Found {len(snippets)} python code snippets to verify.")

  # Create a unique temp directory to avoid collisions with concurrent runs
  temp_dir = Path(tempfile.mkdtemp(prefix="verify_snippets_"))

  results = []

  # 2. Execute each snippet, then write the report — both inside the try so
  # the finally cleanup only runs after the report is fully written.
  try:
    for i, snippet in enumerate(snippets, start=1):
      heading = snippet["heading"]
      code = snippet["code"]
      is_skipped = snippet.get("skip", False)

      # Create a unique, sanitized filename for the snippet
      safe_heading = clean_name(heading)
      temp_file_name = f"snippet_{i}_{safe_heading}.py"
      temp_file_path = temp_dir / temp_file_name

      if is_skipped:
        print(
            f"⏭️  Skipping Snippet {i}/{len(snippets)} under heading"
            f" '{heading}' (marked ignore)."
        )
        results.append({
            "index": i,
            "heading": heading,
            "code": code,
            "temp_file": temp_file_name,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "skipped": True,
        })
        continue

      # Write snippet to file
      with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(code)

      print(
          f"🧪 Testing Snippet {i}/{len(snippets)} under heading '{heading}'..."
      )

      # Run the snippet
      run_res = run_snippet(run_py_path, temp_file_path)

      results.append({
          "index": i,
          "heading": heading,
          "code": code,
          "temp_file": temp_file_name,
          "exit_code": run_res["exit_code"],
          "stdout": run_res["stdout"],
          "stderr": run_res["stderr"],
          "skipped": False,
      })

    # 3. Generate Markdown Report — inside the try so finally runs after this completes.
    # Use clean_name on the stem so the report path is safe on all filesystems.
    # If clean_name strips everything (e.g. a fully non-ASCII filename), fall
    # back to a hash of the original stem so two such files in the same
    # directory never produce the same report path.
    safe_stem = clean_name(md_path.stem) or f"report_{abs(hash(md_path.stem))}"
    report_path = md_path.parent / f"{safe_stem}_REPORT.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w", encoding="utf-8") as f:
      f.write(f"# 🔬 ADK Markdown Snippet Verification Report\n\n")
      f.write(f"* **Source File**: [{md_path.name}](file://{md_path})\n")
      f.write(f"* **Verified On**: `{timestamp}`\n\n")

      # Write summary table
      f.write("## 📈 Executive Summary\n\n")
      f.write(
          "| Snippet | Preceding Heading | Load Phase | Run Phase | Coverage |"
          " Details |\n"
      )
      f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")

      for r in results:
        # Handle explicitly skipped snippets
        if r.get("skipped"):
          f.write(
              f"| **Snippet {r['index']}** | `{md_cell(r['heading'])}` | ⏭️"
              f" **SKIPPED** | ⏭️ **SKIPPED** | — | Marked `{SKIP_ANNOTATION}`"
              " — intentionally ignored. |\n"
          )
          continue

        # Determine Phase 1 (Load) and Phase 3 (Run) statuses from the
        # structured exit code emitted by run.py — no emoji/string matching.
        exit_code = r["exit_code"]
        load_status = "✅ **PASS**"
        run_status = "✅ **PASS**"
        coverage_pct = "—"

        stdout_and_stderr = r["stdout"] + "\n" + r["stderr"]

        if exit_code == EXIT_LOAD_FAILURE:
          load_status = "❌ **FAIL**"
          run_status = "➖ **SKIPPED**"
        elif exit_code == EXIT_NO_COMPONENT:
          run_status = "➖ **NO ADK COMPONENT**"
        elif exit_code == EXIT_RUN_FAILURE:
          run_status = "❌ **FAIL**"
        # EXIT_SUCCESS (0): both statuses remain ✅ **PASS**

        # 3. Parse Coverage — anchor to line start to avoid matching prose.
        # Handles both branch (5 numeric cols) and non-branch (3 cols) formats.
        total_match = re.search(
            r"^TOTAL(?:\s+\d+)+\s+(\d+)%", r["stdout"], re.MULTILINE
        )
        if total_match and load_status != "❌ **FAIL**":
          coverage_pct = f"`{total_match.group(1)}`"

        # 4. Formulate details and handle transient 503s
        details = "All checks passed successfully."
        if load_status == "❌ **FAIL**":
          details = extract_error_detail(r["stdout"], r["stderr"])
        elif run_status == "➖ **NO ADK COMPONENT**":
          details = (
              "No module-level `Workflow`, `Agent`, or `App` instance found."
              " Assign one to a top-level variable to enable runnability"
              " testing."
          )
        elif run_status == "❌ **FAIL**":
          if "503" in stdout_and_stderr and "UNAVAILABLE" in stdout_and_stderr:
            details = (
                "⚠️ **Transient 503 from Gemini API (overloaded)**. Code"
                " structure is correct."
            )
          else:
            details = extract_error_detail(r["stdout"], r["stderr"])

        # Store statuses for reuse in the detailed section
        r["load_status"] = load_status
        r["run_status"] = run_status

        f.write(
            f"| **Snippet {r['index']}** | `{md_cell(r['heading'])}` |"
            f" {load_status} | {run_status} | {coverage_pct} |"
            f" {md_cell(details)} |\n"
        )

      f.write("\n---\n\n## 🔍 Detailed Snippet Reports\n\n")

      for r in results:
        if r.get("skipped"):
          f.write(
              f"### ⏭️ Snippet {r['index']}: `{r['heading']}` *(ignored)*\n\n"
          )
          f.write("#### 📝 Code Block\n")
          f.write(safe_fence(r["code"], "python"))
          f.write("\n\n")
          f.write(
              "> This snippet was skipped because it is annotated with"
              f" `{SKIP_ANNOTATION}`.\n\n"
          )
          f.write("---\n\n")
          continue

        l_stat = r.get("load_status", "✅ **PASS**")
        r_stat = r.get("run_status", "✅ **PASS**")
        if l_stat == "❌ **FAIL**" or r_stat == "❌ **FAIL**":
          status_icon = "❌"
        elif r_stat == "➖ **NO ADK COMPONENT**":
          status_icon = "➖"
        else:
          status_icon = "✅"
        f.write(f"### {status_icon} Snippet {r['index']}: `{r['heading']}`\n\n")

        f.write("#### 📝 Code Block\n")
        f.write(safe_fence(r["code"], "python"))
        f.write("\n\n")

        # Write stdout / stderr logs
        # Split run.py stdout into main log and coverage section using the
        # shared COV_SECTION_HEADER constant (kept in sync with run.py).
        stdout_clean = r["stdout"]
        cov_section_match = re.search(
            rf"({re.escape(COV_SECTION_HEADER)}.*)", r["stdout"], re.DOTALL
        )
        cov_text = cov_section_match.group(1) if cov_section_match else None

        if cov_text:
          stdout_clean = r["stdout"].replace(cov_text, "").strip()

        log_content = stdout_clean
        if r["stderr"]:
          log_content += "\n\n=== STDERR/TRACEBACK ===\n" + r["stderr"].strip()

        f.write("#### 🖥️ Loadability & Runnability Logs\n")
        f.write(safe_fence(log_content))
        f.write("\n\n")

        # Write coverage report if available
        if cov_text:
          f.write("#### 📊 Coverage Report\n")
          f.write(safe_fence(cov_text))
          f.write("\n\n")

        f.write("---\n\n")

    print(f"🎉 Verification complete! Report generated at: {report_path}")

  finally:
    # Always clean up the temp directory, even on Ctrl+C or unexpected errors.
    # This runs after report generation completes (or if it raises), ensuring
    # temp files are never left behind.
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
  main()

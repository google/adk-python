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

"""ADK Release Analyzer Agent - Multi-agent architecture for analyzing releases.

This agent uses a SequentialAgent + LoopAgent pattern to handle large releases
without context overflow:

1. PlannerAgent: Collects changed files and creates analysis groups
2. LoopAgent + FileGroupAnalyzer: Processes one group at a time
3. SummaryAgent: Compiles all findings and creates the GitHub issue

State keys used:
- start_tag, end_tag: Release tags being compared
- compare_url: GitHub compare URL
- file_groups: List of file groups to analyze
- current_group_index: Index of current group being processed
- recommendations: Accumulated recommendations from all groups
"""

import os
import sys
from typing import Any

SAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if SAMPLES_DIR not in sys.path:
  sys.path.append(SAMPLES_DIR)

from adk_documentation.settings import CODE_OWNER
from adk_documentation.settings import CODE_REPO
from adk_documentation.settings import DOC_OWNER
from adk_documentation.settings import DOC_REPO
from adk_documentation.settings import IS_INTERACTIVE
from adk_documentation.settings import LOCAL_REPOS_DIR_PATH
from adk_documentation.tools import clone_or_pull_repo
from adk_documentation.tools import create_issue
from adk_documentation.tools import get_changed_files_summary
from adk_documentation.tools import get_file_diff_for_release
from adk_documentation.tools import list_directory_contents
from adk_documentation.tools import list_releases
from adk_documentation.tools import read_local_git_repo_file_content
from adk_documentation.tools import search_local_git_repo
from google.adk import Agent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models import Gemini
from google.adk.tools.exit_loop_tool import exit_loop
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Retry configuration for handling API rate limits and overload
_RETRY_OPTIONS = types.HttpRetryOptions(
    initial_delay=10,
    attempts=8,
    exp_base=2,
    max_delay=300,
    http_status_codes=[429, 503],
)

# Use gemini-3-pro-preview for planning and summary (better quality)
GEMINI_PRO_WITH_RETRY = Gemini(
    model="gemini-3-pro-preview",
    retry_options=_RETRY_OPTIONS,
)

# Maximum number of files per analysis group to avoid context overflow
MAX_FILES_PER_GROUP = 5

if IS_INTERACTIVE:
  APPROVAL_INSTRUCTION = (
      "Ask for user approval or confirmation for creating or updating the"
      " issue."
  )
else:
  APPROVAL_INSTRUCTION = (
      "**Do not** wait or ask for user approval or confirmation for creating"
      " or updating the issue."
  )


# =============================================================================
# Tool functions for state management
# =============================================================================


def get_next_file_group(tool_context: ToolContext) -> dict[str, Any]:
  """Gets the next group of files to analyze from the state.

  This tool retrieves the next file group from state["file_groups"]
  and increments the current_group_index.

  Args:
      tool_context: The tool context providing access to state.

  Returns:
      A dictionary with the next file group or indication that all groups
      are processed.
  """
  file_groups = tool_context.state.get("file_groups", [])
  current_index = tool_context.state.get("current_group_index", 0)

  if current_index >= len(file_groups):
    return {
        "status": "complete",
        "message": "All file groups have been processed.",
        "total_groups": len(file_groups),
        "processed": current_index,
    }

  current_group = file_groups[current_index]
  tool_context.state["current_group_index"] = current_index + 1

  return {
      "status": "success",
      "group_index": current_index,
      "total_groups": len(file_groups),
      "remaining": len(file_groups) - current_index - 1,
      "files": current_group,
  }


def save_group_recommendations(
    tool_context: ToolContext,
    group_index: int,
    recommendations: list[dict[str, str]],
) -> dict[str, Any]:
  """Saves recommendations for a file group to state.

  Args:
      tool_context: The tool context providing access to state.
      group_index: The index of the group these recommendations belong to.
      recommendations: List of recommendation dicts with keys:
          - summary: Brief summary of the change
          - doc_file: Path to the doc file to update
          - current_state: Current content in the doc
          - proposed_change: What should be changed
          - reasoning: Why this change is needed
          - reference: Reference to the code file

  Returns:
      A dictionary confirming the save operation.
  """
  all_recommendations = tool_context.state.get("recommendations", [])
  all_recommendations.extend(recommendations)
  tool_context.state["recommendations"] = all_recommendations

  return {
      "status": "success",
      "group_index": group_index,
      "new_recommendations": len(recommendations),
      "total_recommendations": len(all_recommendations),
  }


def get_all_recommendations(tool_context: ToolContext) -> dict[str, Any]:
  """Retrieves all accumulated recommendations from state.

  Args:
      tool_context: The tool context providing access to state.

  Returns:
      A dictionary with all recommendations and metadata.
  """
  recommendations = tool_context.state.get("recommendations", [])
  start_tag = tool_context.state.get("start_tag", "unknown")
  end_tag = tool_context.state.get("end_tag", "unknown")
  compare_url = tool_context.state.get("compare_url", "")

  return {
      "status": "success",
      "start_tag": start_tag,
      "end_tag": end_tag,
      "compare_url": compare_url,
      "total_recommendations": len(recommendations),
      "recommendations": recommendations,
  }


def save_release_info(
    tool_context: ToolContext,
    start_tag: str,
    end_tag: str,
    compare_url: str,
    file_groups: list[list[dict[str, Any]]],
    release_summary: str,
    all_changed_files: list[str],
) -> dict[str, Any]:
  """Saves release info and file groups to state for processing.

  Args:
      tool_context: The tool context providing access to state.
      start_tag: The starting release tag.
      end_tag: The ending release tag.
      compare_url: The GitHub compare URL.
      file_groups: List of file groups, where each group is a list of file
          info dicts.
      release_summary: A high-level summary of all changes in this release,
          including the main themes (e.g., "new feature X", "refactoring Y",
          "bug fixes in Z"). This helps individual analyzers understand the
          bigger picture.
      all_changed_files: List of all changed file paths (for cross-reference).

  Returns:
      A dictionary confirming the save operation.
  """
  tool_context.state["start_tag"] = start_tag
  tool_context.state["end_tag"] = end_tag
  tool_context.state["compare_url"] = compare_url
  tool_context.state["file_groups"] = file_groups
  tool_context.state["current_group_index"] = 0
  tool_context.state["recommendations"] = []
  tool_context.state["release_summary"] = release_summary
  tool_context.state["all_changed_files"] = all_changed_files

  return {
      "status": "success",
      "start_tag": start_tag,
      "end_tag": end_tag,
      "total_groups": len(file_groups),
      "total_files": sum(len(group) for group in file_groups),
  }


def get_release_context(tool_context: ToolContext) -> dict[str, Any]:
  """Gets the global release context for cross-group awareness.

  This allows individual file group analyzers to understand:
  - The overall theme of the release
  - What other files were changed (for identifying related changes)
  - What recommendations have already been made (to avoid duplicates)

  Args:
      tool_context: The tool context providing access to state.

  Returns:
      A dictionary with global release context.
  """
  return {
      "status": "success",
      "start_tag": tool_context.state.get("start_tag", "unknown"),
      "end_tag": tool_context.state.get("end_tag", "unknown"),
      "release_summary": tool_context.state.get("release_summary", ""),
      "all_changed_files": tool_context.state.get("all_changed_files", []),
      "existing_recommendations": tool_context.state.get("recommendations", []),
      "current_group_index": tool_context.state.get("current_group_index", 0),
      "total_groups": len(tool_context.state.get("file_groups", [])),
  }


# =============================================================================
# Agent 1: Planner Agent
# =============================================================================

planner_agent = Agent(
    model=GEMINI_PRO_WITH_RETRY,
    name="release_planner",
    description=(
        "Plans the analysis by fetching release info and organizing files into"
        " groups for incremental processing."
    ),
    instruction=f"""
# 1. Identity
You are the Release Planner, responsible for setting up the analysis of ADK
Python releases. You gather information about changes and organize them for
efficient processing.

# 2. Workflow
1. First, call `clone_or_pull_repo` for both repositories:
   - ADK Python codebase: owner={CODE_OWNER}, repo={CODE_REPO}, path={LOCAL_REPOS_DIR_PATH}/{CODE_REPO}
   - ADK Docs: owner={DOC_OWNER}, repo={DOC_REPO}, path={LOCAL_REPOS_DIR_PATH}/{DOC_REPO}

2. Call `list_releases` to find the release tags for {CODE_OWNER}/{CODE_REPO}.
   - By default, compare the two most recent releases.
   - If the user specifies tags, use those instead.

3. Call `get_changed_files_summary` to get the list of changed files WITHOUT
   the full patches (to save context space).
   - **IMPORTANT**: Pass these parameters:
     - `local_repo_path="{LOCAL_REPOS_DIR_PATH}/{CODE_REPO}"` to avoid 300-file limit
     - `path_filter="src/google/adk/"` to only get ADK source files (reduces token usage)

4. Further filter the returned files:
   - **EXCLUDE** test files and `__init__.py` files
   - **IMPORTANT**: Do NOT exclude any file just because it has few changes.
     Even single-line changes to public APIs need documentation updates.
   - **PRIORITIZE** by importance:
     a) New files (status: "added") - ALWAYS include these
     b) CLI files (cli/) - often contain user-facing flags and options
     c) Tool files (tools/) - may contain new tools or tool parameters
     d) Core files (agents/, models/, sessions/, memory/, a2a/, flows/,
        plugins/, evaluation/)
     e) Files with many changes (high additions + deletions)

5. **Create a high-level release summary** based on the changed files:
   - Identify the main themes (e.g., "new tool X added", "refactoring of Y")
   - Note any files that appear related (e.g., same feature area)
   - This summary will be shared with individual file analyzers so they
     understand the bigger picture.

6. Group the filtered files into groups of at most {MAX_FILES_PER_GROUP} files each.
   - **IMPORTANT**: Group RELATED files together (same directory or feature)
   - Files that are part of the same feature should be in the same group
   - Each group should be independently analyzable

7. Call `save_release_info` to save:
   - start_tag, end_tag
   - compare_url
   - file_groups (the organized groups)
   - release_summary (the high-level summary you created)
   - all_changed_files (list of all file paths for cross-reference)

# 3. Output
Provide a summary of:
- Which releases are being compared
- The high-level themes of this release
- How many files changed in total
- How many files are relevant for doc analysis
- How many groups were created
""",
    tools=[
        clone_or_pull_repo,
        list_releases,
        get_changed_files_summary,
        save_release_info,
    ],
    output_key="planner_output",
)


# =============================================================================
# Agent 2: File Group Analyzer (runs inside LoopAgent)
# =============================================================================


def file_analyzer_instruction(readonly_context: ReadonlyContext) -> str:
  """Dynamic instruction that includes current state info."""
  start_tag = readonly_context.state.get("start_tag", "unknown")
  end_tag = readonly_context.state.get("end_tag", "unknown")
  release_summary = readonly_context.state.get("release_summary", "")

  return f"""
# 1. Identity
You are the File Group Analyzer, responsible for analyzing a group of changed
files and finding related documentation that needs updating.

# 2. Context
- Comparing releases: {start_tag} to {end_tag}
- Code repository: {CODE_OWNER}/{CODE_REPO}
- Docs repository: {DOC_OWNER}/{DOC_REPO}
- Docs local path: {LOCAL_REPOS_DIR_PATH}/{DOC_REPO}
- Code local path: {LOCAL_REPOS_DIR_PATH}/{CODE_REPO}

## Release Summary (from Planner)
{release_summary}

# 3. Workflow
1. Call `get_next_file_group` to get the next group of files to analyze.
   - If status is "complete", call the `exit_loop` tool to exit the loop.

2. **FIRST**, call `get_release_context` to understand:
   - The overall release themes (to understand how your files fit in)
   - What other files were changed (to identify related changes)
   - What recommendations already exist (to AVOID DUPLICATES)

3. For each file in the group:
   a) Call `get_file_diff_for_release` to get the patch content for that file.
   b) Analyze the changes THOROUGHLY. Look for:
      **API Changes:**
      - New functions, classes, methods (especially public ones)
      - New parameters added to existing functions
      - New CLI arguments or flags (look for argparse, click decorators)
      - New environment variables (look for os.environ, getenv)
      - New tools or features being added
      - Renamed or deprecated functionality
      **Behavior Changes (even without API changes):**
      - Default values changed
      - Error handling or exception types changed
      - Return value format or content changed
      - Side effects added or removed
      - Performance characteristics changed
      - Edge case handling changed
      - Validation rules changed
   c) Consider how this file relates to OTHER changed files in this release.
   d) Generate MULTIPLE search patterns based on:
      - Class/function names that changed
      - Feature names mentioned in the file path
      - Keywords from the patch content (e.g., "local_storage", "allow_origins")
      - Tool names, parameter names, environment variable names

4. For EACH significant change, call `search_local_git_repo` to find related docs
   in {LOCAL_REPOS_DIR_PATH}/{DOC_REPO}/docs/
   - Search for the feature name, class name, or related keywords
   - If no docs found, recommend creating new documentation

5. Call `read_local_git_repo_file_content` to read the relevant doc files
   and check if they need updating.

6. For each documentation update needed, create a recommendation with:
   - summary: Brief summary of what needs to change
   - doc_file: Relative path in the docs repo (e.g., docs/tools/google-search.md)
   - current_state: What the doc currently says
   - proposed_change: What it should say instead
   - reasoning: Why this update is needed
   - reference: The source code file path
   - related_files: Other changed files that are part of the same change (if any)

7. Call `save_group_recommendations` with all recommendations for this group.

8. After saving, output a brief summary of what you found for this group.

# 4. Rules
- **BE THOROUGH**: Check EVERY change in the diff that could affect users.
  This includes API changes AND behavior changes (default values, error handling,
  return formats, side effects, etc.).
- Focus on changes that users need to know about
- Include behavior changes even if the API signature stays the same
- If a change only affects auto-generated API reference docs, note that
  regeneration is needed instead of manual updates
- **AVOID DUPLICATES**: Check existing_recommendations before adding new ones
- **CROSS-REFERENCE**: If files in your group relate to files in other groups,
  mention this in your recommendation so the Summary agent can consolidate
- **DON'T MISS ITEMS**: Better to have too many recommendations than too few.
  If unsure whether something needs documentation, include it.
- For new features with no existing docs, recommend creating a new page
"""


file_group_analyzer = Agent(
    model=GEMINI_PRO_WITH_RETRY,
    name="file_group_analyzer",
    description=(
        "Analyzes a group of changed files and generates recommendations."
    ),
    instruction=file_analyzer_instruction,
    tools=[
        get_next_file_group,
        get_release_context,  # Get global context to avoid duplicates
        get_file_diff_for_release,
        search_local_git_repo,
        read_local_git_repo_file_content,
        list_directory_contents,
        save_group_recommendations,
        exit_loop,  # Call this when all groups are processed
    ],
    output_key="analyzer_output",
)

# Loop agent that processes file groups one at a time
file_analysis_loop = LoopAgent(
    name="file_analysis_loop",
    sub_agents=[file_group_analyzer],
    max_iterations=50,  # Safety limit
)


# =============================================================================
# Agent 3: Summary Agent
# =============================================================================


def summary_instruction(readonly_context: ReadonlyContext) -> str:
  """Dynamic instruction with release info."""
  start_tag = readonly_context.state.get("start_tag", "unknown")
  end_tag = readonly_context.state.get("end_tag", "unknown")

  return f"""
# 1. Identity
You are the Summary Agent, responsible for compiling all recommendations into
a well-formatted GitHub issue.

# 2. Workflow
1. Call `get_all_recommendations` to retrieve all accumulated recommendations.

2. Organize the recommendations:
   - Group by importance: Feature changes > Bug fixes > Other
   - Within each group, sort by number of affected files
   - Remove duplicates or merge similar recommendations

3. Format the issue body using this template for each recommendation:
   ```
   ### N. **Summary of the change**

   **Doc file**: path/to/doc.md

   **Current state**:
   > Current content in the doc

   **Proposed Change**:
   > What it should say instead

   **Reasoning**:
   Explanation of why this change is necessary.

   **Reference**: src/google/adk/path/to/file.py
   ```

4. Create the GitHub issue:
   - Title: "Found docs updates needed from ADK python release {start_tag} to {end_tag}"
   - Include the compare link at the top
   - {APPROVAL_INSTRUCTION}

5. Call `create_issue` for {DOC_OWNER}/{DOC_REPO} with the formatted content.

# 3. Output
Present a summary of:
- Total recommendations created
- Issue URL if created
- Any notes about the analysis
"""


summary_agent = Agent(
    model=GEMINI_PRO_WITH_RETRY,
    name="summary_agent",
    description="Compiles recommendations and creates the GitHub issue.",
    instruction=summary_instruction,
    tools=[
        get_all_recommendations,
        create_issue,
    ],
    output_key="summary_output",
)


# =============================================================================
# Pipeline Agent: Sequential orchestration of the analysis
# =============================================================================

analysis_pipeline = SequentialAgent(
    name="analysis_pipeline",
    description=(
        "Executes the release analysis pipeline: planning, file analysis, and"
        " summary generation."
    ),
    sub_agents=[
        planner_agent,
        file_analysis_loop,
        summary_agent,
    ],
)


# =============================================================================
# Root Agent: Entry point that understands user requests
# =============================================================================

root_agent = Agent(
    model=GEMINI_PRO_WITH_RETRY,
    name="adk_release_analyzer",
    description=(
        "Analyzes ADK Python releases and generates documentation update"
        " recommendations."
    ),
    instruction=f"""
# 1. Identity
You are the ADK Release Analyzer, a helper bot that analyzes changes between
ADK Python releases and identifies documentation updates needed in the ADK
Docs repository.

# 2. Capabilities
You can help users in several ways:

## A. Full Release Analysis (delegate to analysis_pipeline)
When users want a complete analysis of releases, delegate to the
`analysis_pipeline` sub-agent. This will:
- Clone/update repositories
- Analyze all changed files
- Generate recommendations
- Create a GitHub issue

Use this when users say things like:
- "Analyze the latest releases"
- "Check what docs need updating for v1.15.0"
- "Run a full analysis"

## B. Quick Queries (use your tools directly)
For targeted questions, use your tools directly WITHOUT delegating:

- **"How should I modify doc1.md?"** → Use `search_local_git_repo` to find
  mentions of doc1.md in the codebase, then use `get_changed_files_summary`
  to see what changed, and provide specific guidance.

- **"What changed in the tools module?"** → Use `get_changed_files_summary`
  and filter for tools/ directory.

- **"Show me the recommendations from the last analysis"** → Use
  `get_all_recommendations` to retrieve stored recommendations.

- **"What releases are available?"** → Use `list_releases` directly.

# 3. Workflow Decision
1. First, understand what the user is asking:
   - Full analysis request → delegate to analysis_pipeline
   - Specific question about a file/module → use tools directly
   - Query about previous results → use get_all_recommendations

2. For quick queries, ensure repos are cloned first using `clone_or_pull_repo`
   if needed.

3. Always explain what you're doing and provide clear, actionable answers.

# 4. Available Tools
- `clone_or_pull_repo`: Ensure local repos are up to date
- `list_releases`: See available release tags
- `get_changed_files_summary`: Get list of changed files (lightweight)
- `get_file_diff_for_release`: Get patch for a specific file
- `search_local_git_repo`: Search for patterns in repos
- `read_local_git_repo_file_content`: Read file contents
- `get_all_recommendations`: Retrieve recommendations from previous analysis

# 5. Repository Info
- Code repo: {CODE_OWNER}/{CODE_REPO} at {LOCAL_REPOS_DIR_PATH}/{CODE_REPO}
- Docs repo: {DOC_OWNER}/{DOC_REPO} at {LOCAL_REPOS_DIR_PATH}/{DOC_REPO}
""",
    tools=[
        clone_or_pull_repo,
        list_releases,
        get_changed_files_summary,
        get_file_diff_for_release,
        search_local_git_repo,
        read_local_git_repo_file_content,
        get_all_recommendations,
    ],
    sub_agents=[analysis_pipeline],
)

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

"""Helper script for ADK PR Triage verification and remote update."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def run_command(args: list[str]) -> tuple[int, str, str]:
  """Runs a shell command and returns its exit code, stdout, and stderr."""
  try:
    res = subprocess.run(args, capture_output=True, text=True, check=False)
    return res.returncode, res.stdout.strip(), res.stderr.strip()
  except Exception as e:
    return -1, "", str(e)


def fetch_pr_data(pr_number: str) -> dict | None:
  """Fetches all PR metadata in one shot from GitHub."""
  print(f"[*] Fetching PR #{pr_number} metadata from GitHub...")
  cmd = [
      "gh",
      "pr",
      "view",
      pr_number,
      "--repo",
      "google/adk-python",
      "--json",
      ",".join([
          "number",
          "title",
          "body",
          "state",
          "url",
          "author",
          "additions",
          "deletions",
          "changedFiles",
          "labels",
          "statusCheckRollup",
          "assignees",
          "closingIssuesReferences",
      ]),
  ]
  code, stdout, stderr = run_command(cmd)
  if code != 0:
    print(
        f"Error: Failed to fetch PR details from GitHub: {stderr}",
        file=sys.stderr,
    )
    return None
  try:
    return json.loads(stdout)
  except json.JSONDecodeError:
    print("Error: Failed to parse GitHub API JSON response.", file=sys.stderr)
    return None


def verify_cla(pr_data: dict) -> bool:
  """Verifies if the Google CLA is signed using cached PR data."""
  status_checks = pr_data.get("statusCheckRollup") or []
  cla_check = None
  for check in status_checks:
    if check.get("name") == "cla/google":
      cla_check = check
      break

  if not cla_check:
    print("\n" + "=" * 80)
    print("🚨  CRITICAL COMPLIANCE REFUSAL: GOOGLE CLA NOT SIGNED/VERIFIED  🚨")
    print("=" * 80)
    print(
        "Error: The mandatory 'cla/google' status check is completely missing"
        " on GitHub."
    )
    print(
        "The contributor HAS NOT signed the Google Contributor License"
        " Agreement."
    )
    print(
        "Legal policy strictly prohibits triaging, downloading, or reviewing"
        " this PR."
    )
    print("=" * 80 + "\n")
    return False

  conclusion = cla_check.get("conclusion")
  if conclusion != "SUCCESS":
    print("\n" + "=" * 80)
    print("🚨  CRITICAL COMPLIANCE REFUSAL: GOOGLE CLA NOT SIGNED/VERIFIED  🚨")
    print("=" * 80)
    print(
        "Error: The 'cla/google' status check has the status:"
        f" '{conclusion or 'UNKNOWN'}'."
    )
    print(
        "The contributor HAS NOT successfully signed or verified the Google"
        " CLA."
    )
    print(
        "Legal policy strictly prohibits triaging, downloading, or reviewing"
        " this PR."
    )
    print("=" * 80 + "\n")
    return False

  print("✅ Google CLA is verified and signed (status SUCCESS).")
  return True


def get_current_user() -> str | None:
  """Fetches the login name of the current authenticated GitHub user."""
  cmd = ["gh", "api", "user", "-q", ".login"]
  code, stdout, stderr = run_command(cmd)
  if code != 0:
    return None
  return stdout.strip()


def verify_pr_assignment(pr_data: dict, pr_number: str) -> bool:
  """Checks if the PR is assigned to the current user using cached PR data."""
  print(f"\n[*] Verifying assignment for PR #{pr_number}...")

  # Fetch the current logged in user
  current_user = get_current_user()
  if not current_user:
    print(
        "Warning: Could not determine current GitHub user. Skipping assignment"
        " check."
    )
    return True

  print(f"[*] Current GitHub user: {current_user}")

  assignees = pr_data.get("assignees") or []
  assignee_logins = [a.get("login") for a in assignees if a.get("login")]

  if current_user in assignee_logins:
    print(f"✅ Pull Request #{pr_number} is assigned to you.")
    return True

  assignees_str = ", ".join(assignee_logins) if assignee_logins else "None"
  print(
      f"⚠️  WARNING: Pull Request #{pr_number} is NOT assigned to you!"
      f" Current assignees: {assignees_str}"
  )
  print("\n[!] ACTION REQUIRED: The Pull Request is not assigned to you.")
  print("    Please ask the user if they want to take over the PR.")
  return False


def update_pr_branch(pr_number: str) -> None:
  """Updates the remote PR branch with the latest changes from the base branch."""
  print(
      f"\n[*] Attempting to update PR #{pr_number} branch via remote REBASE..."
  )
  rebase_cmd = [
      "gh",
      "pr",
      "update-branch",
      pr_number,
      "--rebase",
      "--repo",
      "google/adk-python",
  ]
  code, stdout, stderr = run_command(rebase_cmd)
  if code == 0:
    print(
        "✅ Successfully updated PR branch on GitHub by rebasing onto base"
        " branch!"
    )
    if stdout:
      print(stdout)
    return

  print(f"Warning: Remote rebase-update failed: {stderr}")
  print("[*] Falling back to standard remote MERGE commit update...")

  merge_cmd = [
      "gh",
      "pr",
      "update-branch",
      pr_number,
      "--repo",
      "google/adk-python",
  ]
  code, stdout, stderr = run_command(merge_cmd)
  if code == 0:
    print(
        "✅ Successfully updated PR branch on GitHub via standard merge commit!"
    )
    if stdout:
      print(stdout)
    return

  print(
      "\n[!] Warning: Remote branch update failed completely on GitHub server:"
      f" {stderr}"
  )
  print("    This is typical if edits are disabled on the contributor's fork.")
  print(
      "    No worries! We will automatically rebase locally after checking out."
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Triage PR verification and sync helper."
  )
  parser.add_argument(
      "pr_number", help="The GitHub Pull Request number (e.g. 5875)."
  )
  parser.add_argument(
      "--skip-update",
      action="store_true",
      help="Skip updating the remote PR branch on GitHub.",
  )
  args = parser.parse_args()

  # Step 0: Fetch PR data in one-shot
  pr_data = fetch_pr_data(args.pr_number)
  if not pr_data:
    sys.exit(1)

  # Step 1: Verify CLA using cached PR data
  if not verify_cla(pr_data):
    sys.exit(2)  # Exit code 2 indicates compliance refusal

  # Step 2: Output the PR metadata JSON directly to standard output
  print("\n[PR_METADATA_JSON]")
  print(json.dumps(pr_data, indent=2))
  print("[/PR_METADATA_JSON]")

  # Step 3: Verify PR Assignment using cached PR data
  if not verify_pr_assignment(pr_data, args.pr_number):
    sys.exit(3)  # Exit code 3 indicates assignment block

  # Step 4: Update branch
  if not args.skip_update:
    update_pr_branch(args.pr_number)

  print("\n[*] Verification complete. Safe to proceed with checkout.")
  sys.exit(0)


if __name__ == "__main__":
  main()

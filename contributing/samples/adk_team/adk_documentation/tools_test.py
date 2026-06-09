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

"""Tests that the file/search tools stay confined to LOCAL_REPOS_DIR_PATH.

These tools are driven by an LLM that processes untrusted input (issue bodies,
release diffs), so a path that escapes the managed repositories directory must
be rejected even though it is absolute.
"""

import os
import tempfile
import unittest
from unittest import mock

# The tools must operate only under this sandbox; set it before importing so the
# module-level constant resolves to the test directory.
_SANDBOX = tempfile.mkdtemp(prefix="adk_repos_")
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ["LOCAL_REPOS_DIR_PATH"] = _SANDBOX

from adk_documentation import tools  # noqa: E402


class PathConfinementTest(unittest.TestCase):

  def setUp(self):
    self.repo = os.path.join(_SANDBOX, "adk-docs")
    os.makedirs(self.repo, exist_ok=True)
    self.inside_file = os.path.join(self.repo, "index.md")
    with open(self.inside_file, "w", encoding="utf-8") as f:
      f.write("hello")

  def test_read_inside_sandbox_succeeds(self):
    res = tools.read_local_git_repo_file_content(self.inside_file)
    self.assertEqual(res["status"], "success")

  def test_read_outside_sandbox_is_denied(self):
    for path in ("/etc/passwd", "/proc/self/environ"):
      res = tools.read_local_git_repo_file_content(path)
      self.assertEqual(res["status"], "error", path)
      self.assertIn("Access denied", res["error_message"])

  def test_read_symlink_escape_is_denied(self):
    link = os.path.join(self.repo, "sneaky")
    if not os.path.lexists(link):
      os.symlink("/etc/passwd", link)
    res = tools.read_local_git_repo_file_content(link)
    self.assertEqual(res["status"], "error")
    self.assertIn("Access denied", res["error_message"])

  def test_list_and_search_outside_sandbox_are_denied(self):
    self.assertEqual(tools.list_directory_contents("/etc")["status"], "error")
    self.assertEqual(
        tools.search_local_git_repo("/etc", "root")["status"], "error"
    )

  def test_create_pr_rejects_path_traversal_in_changes(self):
    # Reach the file-writing step with the git/network calls stubbed out, then
    # assert a traversing change key is rejected before any write.
    with (
        mock.patch.object(tools, "_run_git_command"),
        mock.patch.object(tools, "post_request"),
    ):
      res = tools.create_pull_request_from_changes(
          repo_owner="google",
          repo_name="adk-docs",
          local_path=self.repo,
          base_branch="main",
          changes={"../../../../tmp/evil.txt": "owned"},
          commit_message="m",
          pr_title="t",
          pr_body="b",
      )
    self.assertEqual(res["status"], "error")
    self.assertIn("escapes the repository", res["error_message"])
    self.assertFalse(os.path.exists("/tmp/evil.txt"))


if __name__ == "__main__":
  unittest.main()

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

"""Unit tests for skill utilities."""

import asyncio
import builtins
import io
import pathlib
import struct
import sys
import threading
import tracemalloc
from unittest import mock
import zipfile
import zlib

from google.adk.skills import _utils
from google.adk.skills import list_skills_in_dir
from google.adk.skills import list_skills_in_dir_async as _list_skills_in_dir_async
from google.adk.skills import list_skills_in_gcs_dir as _list_skills_in_gcs_dir
from google.adk.skills import list_skills_in_gcs_dir_async as _list_skills_in_gcs_dir_async
from google.adk.skills import load_skill_from_dir as _load_skill_from_dir
from google.adk.skills import load_skill_from_dir_async as _load_skill_from_dir_async
from google.adk.skills import load_skill_from_gcs_dir as _load_skill_from_gcs_dir
from google.adk.skills import load_skill_from_gcs_dir_async as _load_skill_from_gcs_dir_async
from google.adk.skills import load_skills_from_dir as _load_skills_from_dir
from google.adk.skills import load_skills_from_dir_async as _load_skills_from_dir_async
from google.adk.skills._utils import _load_skill_from_zip_bytes
from google.adk.skills._utils import _MAX_ZIP_ENTRIES
from google.adk.skills._utils import _MAX_ZIP_UNCOMPRESSED_BYTES
from google.adk.skills._utils import _read_skill_properties
from google.adk.skills._utils import _validate_skill_dir
import pytest

# The first bytes of a PNG file: valid binary content that is not valid UTF-8.
_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def test__load_skill_from_dir(tmp_path):
  """Tests loading a skill from a directory."""
  skill_dir = tmp_path / "test-skill"
  skill_dir.mkdir()

  skill_md_content = """---
name: test-skill
description: Test description
---
Test instructions
"""
  (skill_dir / "SKILL.md").write_text(skill_md_content)

  # Create references
  ref_dir = skill_dir / "references"
  ref_dir.mkdir()
  (ref_dir / "ref1.md").write_text("ref1 content")

  # Create assets
  assets_dir = skill_dir / "assets"
  assets_dir.mkdir()
  (assets_dir / "asset1.txt").write_text("asset1 content")

  # Create scripts
  scripts_dir = skill_dir / "scripts"
  scripts_dir.mkdir()
  (scripts_dir / "script1.sh").write_text("echo hello")

  skill = _load_skill_from_dir(skill_dir)

  assert skill.name == "test-skill"
  assert skill.description == "Test description"
  assert skill.instructions == "Test instructions"
  assert skill.resources.get_reference("ref1.md") == "ref1 content"
  assert skill.resources.get_asset("asset1.txt") == "asset1 content"
  assert skill.resources.get_script("script1.sh").src == "echo hello"


def _write_nested_skill(tmp_path):
  """Writes a skill whose resources live in subdirectories."""
  skill_dir = tmp_path / "nested-skill"
  skill_dir.mkdir()
  (skill_dir / "SKILL.md").write_text("""---
name: nested-skill
description: Test description
---
Test instructions
""")

  scripts_dir = skill_dir / "scripts" / "runtime"
  scripts_dir.mkdir(parents=True)
  (scripts_dir / "helper.py").write_text("helper source")

  ref_dir = skill_dir / "references" / "deep" / "deeper"
  ref_dir.mkdir(parents=True)
  (ref_dir / "note.md").write_text("nested note")

  assets_dir = skill_dir / "assets" / "templates"
  assets_dir.mkdir(parents=True)
  (assets_dir / "tmpl.txt").write_text("template body")

  return skill_dir


def test__load_skill_from_dir_nested_resources_use_forward_slash_keys(tmp_path):
  """Resources in subdirectories are keyed with forward slashes."""
  skill = _load_skill_from_dir(_write_nested_skill(tmp_path))

  assert skill.resources.get_script("runtime/helper.py").src == "helper source"
  assert skill.resources.get_reference("deep/deeper/note.md") == "nested note"
  assert skill.resources.get_asset("templates/tmpl.txt") == "template body"


def test__load_skill_from_dir_nested_resources_on_windows_paths(tmp_path):
  """Windows-style separators still produce forward-slash keys.

  Regression test for the Windows-only defect where `_load_dir` keyed resources
  with `str(relative_path)`. On Windows that is backslash-separated, while
  callers such as `load_skill_resource` look resources up with forward slashes,
  so every resource in a subdirectory was unreachable.

  The bug cannot reproduce on a POSIX test runner, where `str()` already yields
  forward slashes, so the Windows flavour of `relative_to` is simulated here.

  Args:
    tmp_path: pytest fixture providing a temporary directory.
  """
  skill_dir = _write_nested_skill(tmp_path)
  real_relative_to = pathlib.Path.relative_to

  def windows_relative_to(self, *args, **kwargs):
    return pathlib.PureWindowsPath(real_relative_to(self, *args, **kwargs))

  with mock.patch.object(pathlib.Path, "relative_to", windows_relative_to):
    skill = _load_skill_from_dir(skill_dir)

  assert list(skill.resources.scripts) == ["runtime/helper.py"]
  assert skill.resources.get_script("runtime/helper.py").src == "helper source"
  assert skill.resources.get_reference("deep/deeper/note.md") == "nested note"
  assert skill.resources.get_asset("templates/tmpl.txt") == "template body"


def test__load_skill_from_dir_keeps_binary_resources(tmp_path):
  """Tests that non-UTF-8 references and assets are loaded as bytes."""
  skill_dir = tmp_path / "test-skill"
  skill_dir.mkdir()
  (skill_dir / "SKILL.md").write_text(
      "---\nname: test-skill\ndescription: Test description\n---\nBody"
  )

  ref_dir = skill_dir / "references"
  ref_dir.mkdir()
  (ref_dir / "ref1.md").write_text("ref1 content")
  (ref_dir / "diagram.png").write_bytes(_PNG_HEADER)

  assets_dir = skill_dir / "assets"
  assets_dir.mkdir()
  (assets_dir / "logo.png").write_bytes(_PNG_HEADER)

  skill = _load_skill_from_dir(skill_dir)

  assert skill.resources.get_reference("ref1.md") == "ref1 content"
  assert skill.resources.get_reference("diagram.png") == _PNG_HEADER
  assert skill.resources.get_asset("logo.png") == _PNG_HEADER


def test__load_skill_from_dir_skips_binary_scripts(tmp_path):
  """Tests that non-UTF-8 scripts are skipped, since Script.src is text."""
  skill_dir = tmp_path / "test-skill"
  skill_dir.mkdir()
  (skill_dir / "SKILL.md").write_text(
      "---\nname: test-skill\ndescription: Test description\n---\nBody"
  )

  scripts_dir = skill_dir / "scripts"
  scripts_dir.mkdir()
  (scripts_dir / "script1.sh").write_text("echo hello")
  (scripts_dir / "helper").write_bytes(_PNG_HEADER)

  skill = _load_skill_from_dir(skill_dir)

  assert skill.resources.list_scripts() == ["script1.sh"]


def test_allowed_tools_yaml_key(tmp_path):
  """Tests that allowed-tools YAML key loads correctly."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
allowed-tools: "some-tool-*"
---
Instructions here
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  skill = _load_skill_from_dir(skill_dir)
  assert skill.frontmatter.allowed_tools == "some-tool-*"


def test_name_directory_mismatch(tmp_path):
  """Tests that name-directory mismatch raises ValueError."""
  skill_dir = tmp_path / "wrong-dir"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
---
Body
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  with pytest.raises(ValueError, match="does not match directory"):
    _load_skill_from_dir(skill_dir)


def test_validate_skill_dir_valid(tmp_path):
  """Tests validate_skill_dir with a valid skill."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
---
Body
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  problems = _validate_skill_dir(skill_dir)
  assert problems == []


def test_validate_skill_dir_missing_dir(tmp_path):
  """Tests validate_skill_dir with missing directory."""
  problems = _validate_skill_dir(tmp_path / "nonexistent")
  assert len(problems) == 1
  assert "does not exist" in problems[0]


def test_validate_skill_dir_missing_skill_md(tmp_path):
  """Tests validate_skill_dir with missing SKILL.md."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  problems = _validate_skill_dir(skill_dir)
  assert len(problems) == 1
  assert "SKILL.md not found" in problems[0]


def test_validate_skill_dir_name_mismatch(tmp_path):
  """Tests validate_skill_dir catches name-directory mismatch."""
  skill_dir = tmp_path / "wrong-dir"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
---
Body
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  problems = _validate_skill_dir(skill_dir)
  assert any("does not match" in p for p in problems)


def test_validate_skill_dir_unknown_fields(tmp_path):
  """Tests validate_skill_dir detects unknown frontmatter fields."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A skill
unknown-field: something
---
Body
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  problems = _validate_skill_dir(skill_dir)
  assert any("Unknown frontmatter" in p for p in problems)


def test__read_skill_properties(tmp_path):
  """Tests read_skill_properties basic usage."""
  skill_dir = tmp_path / "my-skill"
  skill_dir.mkdir()

  skill_md = """---
name: my-skill
description: A cool skill
license: MIT
---
Body content
"""
  (skill_dir / "SKILL.md").write_text(skill_md)

  fm = _read_skill_properties(skill_dir)
  assert fm.name == "my-skill"
  assert fm.description == "A cool skill"
  assert fm.license == "MIT"


@mock.patch("google.cloud.storage.Client")
def test__list_skills_in_gcs_dir(mock_client_class):

  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  mock_iterator = mock.MagicMock()
  mock_iterator.prefixes = ["skills/my-skill/"]
  mock_bucket.list_blobs.return_value = mock_iterator

  mock_blob = mock.MagicMock()
  mock_blob.exists.return_value = True
  mock_blob.download_as_text.return_value = (
      "---\nname: my-skill\ndescription: A skill\n---\nBody"
  )
  mock_bucket.blob.return_value = mock_blob

  skills = _list_skills_in_gcs_dir("my-bucket", "skills/")
  assert "my-skill" in skills
  assert skills["my-skill"].name == "my-skill"


@mock.patch("google.cloud.storage.Client")
@mock.patch("logging.warning")
def test__list_skills_in_gcs_dir_skips_invalid(
    mock_logging_warning, mock_client_class
):
  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  mock_iterator = mock.MagicMock()
  mock_iterator.prefixes = ["skills/invalid-skill/", "skills/valid-skill/"]
  mock_bucket.list_blobs.return_value = mock_iterator

  def mock_blob_side_effect(path):
    m = mock.MagicMock()
    m.exists.return_value = True
    if "invalid-skill" in path:
      m.download_as_text.return_value = "invalid yaml content"
    else:
      m.download_as_text.return_value = (
          "---\nname: valid-skill\ndescription: A skill\n---\nBody"
      )
    return m

  mock_bucket.blob.side_effect = mock_blob_side_effect

  skills = _list_skills_in_gcs_dir("my-bucket", "skills/")
  assert "valid-skill" in skills
  assert "invalid-skill" not in skills

  # Verify warning was logged for the invalid skill
  mock_logging_warning.assert_called_once()
  args, _ = mock_logging_warning.call_args
  assert "Skipping invalid skill" in args[0]
  assert args[1] == "invalid-skill"
  assert args[2] == "my-bucket"


@mock.patch("google.cloud.storage.Client")
def test__load_skill_from_gcs_dir(mock_client_class):

  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  def mock_blob_side_effect(path):
    m = mock.MagicMock()
    if path.endswith("SKILL.md"):
      m.exists.return_value = True
      m.download_as_text.return_value = (
          "---\nname: my-skill\ndescription: Test description\n---\nTest"
          " instructions"
      )
    else:
      m.exists.return_value = False
    return m

  mock_bucket.blob.side_effect = mock_blob_side_effect

  # For resources
  def list_blobs_side_effect(prefix=None):
    if prefix.endswith("references/"):
      m = mock.MagicMock()
      m.name = prefix + "ref1.md"
      m.download_as_text.return_value = "ref1 content"
      return [m]
    return []

  mock_bucket.list_blobs.side_effect = list_blobs_side_effect

  skill = _load_skill_from_gcs_dir("my-bucket", "skills/my-skill/")

  assert skill.name == "my-skill"
  assert skill.description == "Test description"
  assert skill.instructions == "Test instructions"
  # Using dict access for reference
  assert skill.resources.get_reference("ref1.md") == "ref1 content"


@mock.patch("google.cloud.storage.Client")
def test__load_skill_from_gcs_dir_binary_resources(mock_client_class):
  """Tests that non-UTF-8 GCS blobs are loaded as bytes, and scripts skipped."""

  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  def mock_blob_side_effect(path):
    m = mock.MagicMock()
    m.exists.return_value = path.endswith("SKILL.md")
    m.download_as_text.return_value = (
        "---\nname: my-skill\ndescription: Test description\n---\nTest"
        " instructions"
    )
    return m

  mock_bucket.blob.side_effect = mock_blob_side_effect

  def binary_blob(name):
    m = mock.MagicMock()
    m.name = name
    m.download_as_text.side_effect = UnicodeDecodeError(
        "utf-8", _PNG_HEADER, 0, 1, "invalid start byte"
    )
    m.download_as_bytes.return_value = _PNG_HEADER
    return m

  def list_blobs_side_effect(prefix=None):
    if prefix.endswith("assets/"):
      return [binary_blob(prefix + "logo.png")]
    if prefix.endswith("scripts/"):
      return [binary_blob(prefix + "helper")]
    return []

  mock_bucket.list_blobs.side_effect = list_blobs_side_effect

  skill = _load_skill_from_gcs_dir("my-bucket", "skills/my-skill/")

  assert skill.resources.get_asset("logo.png") == _PNG_HEADER
  assert not skill.resources.list_scripts()


def test_list_skills_in_dir(tmp_path):
  """Tests listing skills in a directory."""
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()

  # Valid skill 1
  skill1_dir = skills_dir / "skill1"
  skill1_dir.mkdir()
  (skill1_dir / "SKILL.md").write_text(
      "---\nname: skill1\ndescription: desc1\n---\nbody"
  )

  # Valid skill 2
  skill2_dir = skills_dir / "skill2"
  skill2_dir.mkdir()
  (skill2_dir / "SKILL.md").write_text(
      "---\nname: skill2\ndescription: desc2\n---\nbody"
  )

  # Invalid skill: missing SKILL.md
  (skills_dir / "invalid-no-md").mkdir()

  # Invalid skill: invalid YAML
  invalid_yaml_dir = skills_dir / "invalid-yaml"
  invalid_yaml_dir.mkdir()
  (invalid_yaml_dir / "SKILL.md").write_text("---\ninvalid: yaml: :\n---\nbody")

  # Invalid skill: name mismatch
  mismatch_dir = skills_dir / "mismatch"
  mismatch_dir.mkdir()
  (mismatch_dir / "SKILL.md").write_text(
      "---\nname: other-name\ndescription: desc\n---\nbody"
  )

  skills = list_skills_in_dir(skills_dir)

  assert len(skills) == 2
  assert "skill1" in skills
  assert "skill2" in skills
  assert skills["skill1"].name == "skill1"
  assert skills["skill2"].name == "skill2"


def test_list_skills_in_dir_missing_base_path(tmp_path):
  """Tests list_skills_in_dir with missing base directory."""

  skills = list_skills_in_dir(tmp_path / "nonexistent")
  assert skills == {}


def test__load_skill_from_zip_bytes():
  """Tests loading a skill directly from in-memory zip file bytes."""

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr(
        "SKILL.md",
        "---\nname: my-skill\ndescription: A skill\n---\nBody instructions",
    )
    z.writestr("references/ref1.md", "ref1 content")
    z.writestr("scripts/script1.sh", "echo hello")

  skill = _load_skill_from_zip_bytes(zip_buffer.getvalue())
  assert skill.frontmatter.name == "my-skill"
  assert skill.frontmatter.description == "A skill"
  assert skill.instructions == "Body instructions"
  assert skill.resources.get_reference("ref1.md") == "ref1 content"
  assert skill.resources.get_script("script1.sh").src == "echo hello"


def test__load_skill_from_zip_bytes_keeps_binary_resources():
  """Tests that non-UTF-8 archive members are loaded as bytes."""

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr(
        "SKILL.md",
        "---\nname: my-skill\ndescription: A skill\n---\nBody instructions",
    )
    z.writestr("references/ref1.md", "ref1 content")
    z.writestr("references/diagram.png", _PNG_HEADER)
    z.writestr("assets/logo.png", _PNG_HEADER)
    z.writestr("scripts/script1.sh", "echo hello")
    z.writestr("scripts/helper", _PNG_HEADER)

  skill = _load_skill_from_zip_bytes(zip_buffer.getvalue())

  assert skill.resources.get_reference("ref1.md") == "ref1 content"
  assert skill.resources.get_reference("diagram.png") == _PNG_HEADER
  assert skill.resources.get_asset("logo.png") == _PNG_HEADER
  assert skill.resources.list_scripts() == ["script1.sh"]


def test__load_skill_from_zip_bytes_rejects_oversized_archive():
  """Tests that an archive declaring too much decompressed data is refused."""

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr(
        "SKILL.md",
        "---\nname: my-skill\ndescription: A skill\n---\nBody instructions",
    )
    # Stream the payload so the test never holds the whole thing in memory.
    chunk = b"a" * (1024 * 1024)
    chunks = _MAX_ZIP_UNCOMPRESSED_BYTES // len(chunk) + 1
    with z.open("references/big.md", "w") as f:
      for _ in range(chunks):
        f.write(chunk)

  with pytest.raises(ValueError, match="decompressed"):
    _load_skill_from_zip_bytes(zip_buffer.getvalue())


def test__load_skill_from_zip_bytes_rejects_too_many_entries():
  """Tests that an archive with too many entries is refused."""

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr(
        "SKILL.md",
        "---\nname: my-skill\ndescription: A skill\n---\nBody instructions",
    )
    for i in range(_MAX_ZIP_ENTRIES):
      z.writestr(f"references/ref{i}.md", "x")

  with pytest.raises(ValueError, match="too many entries"):
    _load_skill_from_zip_bytes(zip_buffer.getvalue())


def test__load_skill_from_zip_bytes_accepts_archive_at_the_limits():
  """Tests that an archive exactly at both ceilings is still accepted."""

  skill_md = "---\nname: my-skill\ndescription: A skill\n---\nBody"
  padding = "x" * 64
  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("SKILL.md", skill_md)
    z.writestr("references/pad.md", padding)

  # Two entries, and exactly as many bytes as the ceiling allows.
  with (
      mock.patch("google.adk.skills._utils._MAX_ZIP_ENTRIES", 2),
      mock.patch(
          "google.adk.skills._utils._MAX_ZIP_UNCOMPRESSED_BYTES",
          len(skill_md) + len(padding),
      ),
  ):
    skill = _load_skill_from_zip_bytes(zip_buffer.getvalue())

  assert skill.resources.get_reference("pad.md") == padding


_UNDERSTATED_REAL_BYTES = 64 * 1024 * 1024


def _zip_understating_big_member(
    real_size: int, declared_size: int, *, matching_crc: bool
) -> bytes:
  """Builds an archive whose central directory under-reports a member's size.

  ``references/big.md`` really expands to ``real_size`` bytes while the
  directory claims ``declared_size``, the way a hostile archive would. With
  ``matching_crc`` the checksum is rewritten to cover only the declared
  prefix, so the archive is internally consistent about the lie.
  """
  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr(
        "SKILL.md",
        "---\nname: my-skill\ndescription: A skill\n---\nBody instructions",
    )
    # Stream the payload so the test never holds the whole thing in memory.
    chunk = b"a" * (1024 * 1024)
    with z.open("references/big.md", "w") as f:
      for _ in range(real_size // len(chunk)):
        f.write(chunk)
  raw = bytearray(zip_buffer.getvalue())

  # Walk the central directory and rewrite the big member's declared size.
  eocd = raw.rfind(b"PK\x05\x06")
  entry_count = struct.unpack("<H", raw[eocd + 10 : eocd + 12])[0]
  offset = struct.unpack("<I", raw[eocd + 16 : eocd + 20])[0]
  for _ in range(entry_count):
    name_len, extra_len, comment_len = struct.unpack(
        "<HHH", raw[offset + 28 : offset + 34]
    )
    if bytes(raw[offset + 46 : offset + 46 + name_len]) == b"references/big.md":
      raw[offset + 24 : offset + 28] = struct.pack("<I", declared_size)
      if matching_crc:
        raw[offset + 16 : offset + 20] = struct.pack(
            "<I", zlib.crc32(b"a" * declared_size)
        )
    offset += 46 + name_len + extra_len + comment_len
  return bytes(raw)


def test__load_skill_from_zip_bytes_rejects_member_understating_its_size():
  """Tests that a member expanding past the size it declares is refused."""

  zip_bytes = _zip_understating_big_member(
      _UNDERSTATED_REAL_BYTES, 100, matching_crc=False
  )

  with pytest.raises(ValueError, match="malformed"):
    _load_skill_from_zip_bytes(zip_bytes)


def test__load_skill_from_zip_bytes_bounds_bytes_read_from_a_lying_member():
  """Tests that reading costs what a member holds, not what it hides."""

  # Nothing in this archive gives the lie away: the declared sizes are small
  # enough to pass every up-front check and the checksum matches the declared
  # prefix, so the read itself has to stay bounded.
  zip_bytes = _zip_understating_big_member(
      _UNDERSTATED_REAL_BYTES, 100, matching_crc=True
  )

  tracemalloc.start()
  try:
    skill = _load_skill_from_zip_bytes(zip_bytes)
    peak = tracemalloc.get_traced_memory()[1]
  finally:
    tracemalloc.stop()

  assert skill.resources.get_reference("big.md") == "a" * 100
  # Decompressing the member in one call peaks at roughly the 64 MB it really
  # holds; decompressing it in steps peaks near the size of the archive.
  assert peak < _UNDERSTATED_REAL_BYTES // 4


def test__list_skills_in_gcs_dir_import_error():
  """Tests list_skills_in_gcs_dir raises ImportError when storage missing."""
  real_import = builtins.__import__

  def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "google.cloud" and "storage" in (fromlist or ()):
      raise ImportError("No module named 'google.cloud.storage'")
    return real_import(name, globals, locals, fromlist, level)

  with mock.patch("builtins.__import__", mock_import):
    with pytest.raises(ImportError, match="google-cloud-storage is required"):
      _list_skills_in_gcs_dir("my-bucket", "skills/")


def test__load_skill_from_gcs_dir_import_error():
  """Tests load_skill_from_gcs_dir raises ImportError when storage missing."""
  real_import = builtins.__import__

  def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "google.cloud" and "storage" in (fromlist or ()):
      raise ImportError("No module named 'google.cloud.storage'")
    return real_import(name, globals, locals, fromlist, level)

  with mock.patch("builtins.__import__", mock_import):
    with pytest.raises(ImportError, match="google-cloud-storage is required"):
      _load_skill_from_gcs_dir("my-bucket", "skills/my-skill/")


def test__load_skills_from_dir(tmp_path):
  """Tests loading multiple skills from a directory."""
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()

  # Skill 1
  skill1_dir = skills_dir / "skill1"
  skill1_dir.mkdir()
  (skill1_dir / "SKILL.md").write_text(
      "---\nname: skill1\ndescription: desc1\n---\nbody1"
  )

  # Skill 2
  skill2_dir = skills_dir / "skill2"
  skill2_dir.mkdir()
  (skill2_dir / "SKILL.md").write_text(
      "---\nname: skill2\ndescription: desc2\n---\nbody2"
  )

  # Non-skill directory (no SKILL.md) should be ignored
  (skills_dir / "__pycache__").mkdir()

  skills = _load_skills_from_dir(skills_dir)
  assert len(skills) == 2
  skill_names = [s.name for s in skills]
  assert "skill1" in skill_names
  assert "skill2" in skill_names


def test__load_skills_from_dir_errors(tmp_path):
  """Tests errors in load_skills_from_dir."""
  with pytest.raises(FileNotFoundError, match="does not exist"):
    _load_skills_from_dir(tmp_path / "nonexistent")

  file_path = tmp_path / "some_file.txt"
  file_path.write_text("hello")
  with pytest.raises(ValueError, match="not a directory"):
    _load_skills_from_dir(file_path)


# --- Async wrappers --------------------------------------------------------

# Guards the deadlock-style test below: with a correct (off-thread)
# implementation the handshake completes in milliseconds, so this only ever
# elapses when the event loop is genuinely blocked.
_BLOCKED_LOOP_TIMEOUT_SEC = 10

# Each async wrapper and the blocking function it must offload, plus the
# minimal positional args needed to call it.
_ASYNC_WRAPPERS = [
    ("_load_skill_from_dir_async", "_load_skill_from_dir", ("skill-dir",)),
    ("_load_skills_from_dir_async", "_load_skills_from_dir", ("skills-dir",)),
    ("_list_skills_in_dir_async", "_list_skills_in_dir", ("skills-dir",)),
    (
        "_load_skill_from_gcs_dir_async",
        "_load_skill_from_gcs_dir",
        ("my-bucket", "my-skill"),
    ),
    (
        "_list_skills_in_gcs_dir_async",
        "_list_skills_in_gcs_dir",
        ("my-bucket",),
    ),
]


@pytest.mark.parametrize("async_name, sync_name, args", _ASYNC_WRAPPERS)
async def test_async_wrapper_runs_blocking_call_off_event_loop(
    monkeypatch, async_name, sync_name, args
):
  """Each async wrapper must run its blocking counterpart in a worker thread.

  This is the property that distinguishes these wrappers from a plain
  ``async def f(): return _sync_f(...)``, which would satisfy every other test
  in this file while still stalling the caller's event loop.
  """
  calls = []

  def _record_thread(*call_args, **call_kwargs):
    calls.append((threading.get_ident(), call_args, call_kwargs))
    return "sentinel-result"

  monkeypatch.setattr(_utils, sync_name, _record_thread)

  result = await getattr(_utils, async_name)(*args)

  assert len(calls) == 1
  thread_id, call_args, _ = calls[0]
  assert thread_id != threading.get_ident(), (
      f"{sync_name} ran on the event loop thread; {async_name} must offload it"
      " to a worker thread"
  )
  # The wrapper must forward its arguments through unchanged.
  assert call_args[: len(args)] == args
  assert result == "sentinel-result"


async def test_async_wrapper_keeps_event_loop_responsive(monkeypatch):
  """The event loop must keep scheduling tasks while a wrapper is in flight.

  The blocking stand-in can only be released by a coroutine running on the
  event loop, so an implementation that blocks the loop deadlocks here and
  fails on the timeout instead of passing silently.
  """
  entered = threading.Event()
  release = threading.Event()

  def _blocking_loader(*args, **kwargs):
    entered.set()
    if not release.wait(timeout=_BLOCKED_LOOP_TIMEOUT_SEC):
      raise AssertionError(
          "event loop never resumed while the blocking call was in flight"
      )
    return "loaded"

  monkeypatch.setattr(_utils, "_load_skill_from_dir", _blocking_loader)

  async def _release_once_entered():
    # Only makes progress if the event loop was not blocked by the wrapper.
    while not entered.is_set():
      await asyncio.sleep(0.001)
    release.set()

  results = await asyncio.wait_for(
      asyncio.gather(
          _load_skill_from_dir_async("skill-dir"), _release_once_entered()
      ),
      timeout=_BLOCKED_LOOP_TIMEOUT_SEC,
  )

  assert results[0] == "loaded"


async def test_async_wrappers_run_concurrently(monkeypatch):
  """Independent loads must overlap rather than serialize on the event loop."""
  barrier = threading.Barrier(3, timeout=_BLOCKED_LOOP_TIMEOUT_SEC)

  def _rendezvous(skill_dir):
    # Each call blocks until all three are running at once. A serialized
    # implementation can never reach the barrier count and times out.
    barrier.wait()
    return skill_dir

  monkeypatch.setattr(_utils, "_load_skill_from_dir", _rendezvous)

  results = await asyncio.wait_for(
      asyncio.gather(*(_load_skill_from_dir_async(f"s{i}") for i in range(3))),
      timeout=_BLOCKED_LOOP_TIMEOUT_SEC,
  )

  assert results == ["s0", "s1", "s2"]


async def test_load_skill_from_dir_async(tmp_path):
  """Tests loading a skill from a directory asynchronously."""
  skill_dir = tmp_path / "test-skill"
  skill_dir.mkdir()

  skill_md_content = """---
name: test-skill
description: Test description
---
Test instructions
"""
  (skill_dir / "SKILL.md").write_text(skill_md_content)

  # Create references
  ref_dir = skill_dir / "references"
  ref_dir.mkdir()
  (ref_dir / "ref1.md").write_text("ref1 content")

  skill = await _load_skill_from_dir_async(skill_dir)

  assert skill.name == "test-skill"
  assert skill.description == "Test description"
  assert skill.instructions == "Test instructions"
  assert skill.resources.get_reference("ref1.md") == "ref1 content"


async def test_load_skill_from_dir_async_propagates_errors(tmp_path):
  """Errors raised in the worker thread must surface to the caller."""
  with pytest.raises(FileNotFoundError):
    await _load_skill_from_dir_async(tmp_path / "nonexistent")


async def test_load_skills_from_dir_async(tmp_path):
  """Tests loading every skill in a directory asynchronously."""
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()

  for name in ("skill-a", "skill-b"):
    skill_dir = skills_dir / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: desc {name}\n---\nbody {name}"
    )
  # Directories without a SKILL.md are skipped, matching the sync version.
  (skills_dir / "not-a-skill").mkdir()

  skills = await _load_skills_from_dir_async(skills_dir)

  assert [skill.name for skill in skills] == ["skill-a", "skill-b"]
  assert skills[0].instructions == "body skill-a"


async def test_load_skills_from_dir_async_propagates_errors(tmp_path):
  """Errors raised in the worker thread must surface to the caller."""
  with pytest.raises(FileNotFoundError, match="does not exist"):
    await _load_skills_from_dir_async(tmp_path / "nonexistent")


async def test_list_skills_in_dir_async(tmp_path):
  """Tests listing skills in a directory asynchronously."""
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()

  # Valid skill 1
  skill1_dir = skills_dir / "skill1"
  skill1_dir.mkdir()
  (skill1_dir / "SKILL.md").write_text(
      "---\nname: skill1\ndescription: desc1\n---\nbody"
  )

  skills = await _list_skills_in_dir_async(skills_dir)

  assert len(skills) == 1
  assert "skill1" in skills
  assert skills["skill1"].name == "skill1"


@mock.patch("google.cloud.storage.Client")
async def test_load_skill_from_gcs_dir_async(mock_client_class):
  """Tests loading a skill from GCS asynchronously."""
  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  def mock_blob_side_effect(path):
    m = mock.MagicMock()
    if path.endswith("SKILL.md"):
      m.exists.return_value = True
      m.download_as_text.return_value = (
          "---\nname: my-skill\ndescription: Test description\n---\nTest"
          " instructions"
      )
    else:
      m.exists.return_value = False
    return m

  mock_bucket.blob.side_effect = mock_blob_side_effect

  # For resources
  def list_blobs_side_effect(prefix=None):
    if prefix.endswith("references/"):
      m = mock.MagicMock()
      m.name = prefix + "ref1.md"
      m.download_as_text.return_value = "ref1 content"
      return [m]
    return []

  mock_bucket.list_blobs.side_effect = list_blobs_side_effect

  skill = await _load_skill_from_gcs_dir_async(
      "my-bucket", "my-skill", "skills"
  )

  assert skill.name == "my-skill"
  assert skill.description == "Test description"
  assert skill.instructions == "Test instructions"
  assert skill.resources.get_reference("ref1.md") == "ref1 content"
  mock_bucket.blob.assert_any_call("skills/my-skill/SKILL.md")


@mock.patch("google.cloud.storage.Client")
async def test_list_skills_in_gcs_dir_async(mock_client_class):
  """Tests listing skills in GCS asynchronously."""
  mock_client = mock.MagicMock()
  mock_client_class.return_value = mock_client
  mock_bucket = mock.MagicMock()
  mock_client.bucket.return_value = mock_bucket

  mock_iterator = mock.MagicMock()
  mock_iterator.prefixes = ["skills/my-skill/"]
  mock_bucket.list_blobs.return_value = mock_iterator

  mock_blob = mock.MagicMock()
  mock_blob.exists.return_value = True
  mock_blob.download_as_text.return_value = (
      "---\nname: my-skill\ndescription: A skill\n---\nBody"
  )
  mock_bucket.blob.return_value = mock_blob

  skills = await _list_skills_in_gcs_dir_async("my-bucket", "skills/")
  assert "my-skill" in skills
  assert skills["my-skill"].name == "my-skill"

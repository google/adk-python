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

"""Utility functions for Agent Skills."""

from __future__ import annotations

import asyncio
import io
import logging
import pathlib
from typing import Dict
from typing import Union
import zipfile

from google.auth import credentials as auth
from pydantic import ValidationError
import yaml

from . import models

# Bounds on a skill archive, which may come from a remote registry and is
# untrusted until it has been loaded. They are generous relative to any
# realistic skill; the toolset already warns about payloads over 16 MB.
_MAX_ZIP_ENTRIES = 2000
_MAX_ZIP_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
# How much of a member is decompressed per step. Reading in steps keeps the
# transient buffer this size however much the member really expands.
_ZIP_READ_CHUNK_BYTES = 64 * 1024

_ALLOWED_FRONTMATTER_KEYS = frozenset({
    "name",
    "description",
    "license",
    "allowed-tools",
    "allowed_tools",
    "metadata",
    "compatibility",
})


def _load_dir(directory: pathlib.Path) -> dict[str, Union[str, bytes]]:
  """Recursively load files from a directory into a dictionary.

  Args:
    directory: Path to the directory to load.

  Returns:
    Dictionary mapping relative file paths to their content: `str` for UTF-8
    text, `bytes` for everything else.
  """
  files: dict[str, Union[str, bytes]] = {}
  if directory.exists() and directory.is_dir():
    for file_path in directory.rglob("*"):
      if "__pycache__" in file_path.parts:
        continue
      if file_path.is_file():
        relative_path = file_path.relative_to(directory)
        try:
          files[relative_path.as_posix()] = file_path.read_text(
              encoding="utf-8"
          )
        except UnicodeDecodeError:
          files[relative_path.as_posix()] = file_path.read_bytes()
  return files


def _build_scripts(
    raw_scripts: dict[str, Union[str, bytes]],
) -> dict[str, models.Script]:
  """Wrap raw script sources in `Script` models.

  Args:
    raw_scripts: Mapping of relative path to raw script content.

  Returns:
    Mapping of relative path to `Script`, omitting any script that is not
    UTF-8 text, since `Script.src` holds source code.
  """
  scripts = {}
  for name, src in raw_scripts.items():
    if isinstance(src, bytes):
      try:
        src = src.decode("utf-8")
      except UnicodeDecodeError:
        logging.warning("Skipping non-UTF-8 skill script '%s'.", name)
        continue
    scripts[name] = models.Script(src=src)
  return scripts


def _parse_skill_md_content(content: str) -> tuple[dict, str]:
  """Parse SKILL.md from raw content string.

  Args:
    content: The string content of SKILL.md.

  Returns:
    Tuple of (parsed_frontmatter_dict, body_string).

  Raises:
    ValueError: If SKILL.md is invalid.
  """
  if not content.startswith("---"):
    raise ValueError("SKILL.md must start with YAML frontmatter (---)")

  parts = content.split("---", 2)
  if len(parts) < 3:
    raise ValueError("SKILL.md frontmatter not properly closed with ---")

  frontmatter_str = parts[1]
  body = parts[2].strip()

  try:
    parsed = yaml.safe_load(frontmatter_str)
  except yaml.YAMLError as e:
    raise ValueError(f"Invalid YAML in frontmatter: {e}") from e

  if not isinstance(parsed, dict):
    raise ValueError("SKILL.md frontmatter must be a YAML mapping")

  return parsed, body


def _parse_skill_md(
    skill_dir: pathlib.Path,
) -> tuple[dict, str, pathlib.Path]:
  """Parse SKILL.md from a skill directory.

  Args:
    skill_dir: Resolved path to the skill directory.

  Returns:
    Tuple of (parsed_frontmatter_dict, body_string, skill_md_path).

  Raises:
    FileNotFoundError: If the directory or SKILL.md is not found.
    ValueError: If SKILL.md is invalid.
  """
  if not skill_dir.is_dir():
    raise FileNotFoundError(f"Skill directory '{skill_dir}' not found.")

  skill_md = None
  for name in ("SKILL.md", "skill.md"):
    path = skill_dir / name
    if path.exists():
      skill_md = path
      break

  if skill_md is None:
    raise FileNotFoundError(f"SKILL.md not found in '{skill_dir}'.")

  content = skill_md.read_text(encoding="utf-8")
  parsed, body = _parse_skill_md_content(content)

  return parsed, body, skill_md


def _load_skill_from_dir(skill_dir: Union[str, pathlib.Path]) -> models.Skill:
  """Load a complete skill from a directory.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    Skill object with all components loaded.

  Raises:
    FileNotFoundError: If the skill directory or SKILL.md is not found.
    ValueError: If SKILL.md is invalid or the skill name does not match
      the directory name.
  """
  skill_dir = pathlib.Path(skill_dir).resolve()

  parsed, body, skill_md = _parse_skill_md(skill_dir)

  # Use model_validate to handle aliases like allowed-tools
  frontmatter = models.Frontmatter.model_validate(parsed)

  # Validate that skill name matches the directory name
  if skill_dir.name != frontmatter.name:
    raise ValueError(
        f"Skill name '{frontmatter.name}' does not match directory"
        f" name '{skill_dir.name}'."
    )

  references = _load_dir(skill_dir / "references")
  assets = _load_dir(skill_dir / "assets")
  scripts = _build_scripts(_load_dir(skill_dir / "scripts"))

  resources = models.Resources(
      references=references,
      assets=assets,
      scripts=scripts,
  )

  return models.Skill(
      frontmatter=frontmatter,
      instructions=body,
      resources=resources,
  )


def _load_skills_from_dir(
    skills_dir: Union[str, pathlib.Path],
) -> list[models.Skill]:
  """Load all skills from subdirectories within a directory.

  Args:
    skills_dir: Path to the directory containing skill folders.

  Returns:
    List of Skill objects loaded from valid skill directories.

  Raises:
    FileNotFoundError: If skills_dir does not exist.
    ValueError: If skills_dir is not a directory, or if any skill fails
      validation.
  """
  skills_dir = pathlib.Path(skills_dir).resolve()
  if not skills_dir.exists():
    raise FileNotFoundError(f"Skills directory '{skills_dir}' does not exist.")
  if not skills_dir.is_dir():
    raise ValueError(f"'{skills_dir}' is not a directory.")

  skills: list[models.Skill] = []
  for subdir in sorted(skills_dir.iterdir()):
    if not subdir.is_dir():
      continue
    if (
        not (subdir / "SKILL.md").exists()
        and not (subdir / "skill.md").exists()
    ):
      continue
    skills.append(_load_skill_from_dir(subdir))

  return skills


def _read_zip_member(
    z: zipfile.ZipFile,
    member: Union[str, zipfile.ZipInfo],
    budget: int,
) -> tuple[bytes, int]:
  """Read one archive member in fixed steps, against a byte budget.

  A member can expand to far more than its central-directory entry declares,
  but zipfile truncates the read to the declared size, so the caller's cap on
  the declared total is what bounds the bytes returned. Reading in fixed steps
  keeps the decompressor's transient buffer small while that happens; the
  budget is defense in depth behind the declared-size cap.

  Args:
    z: The open archive.
    member: The name or entry to read.
    budget: How many more bytes may be decompressed from this archive.

  Returns:
    The member's bytes, and the budget remaining after reading it.

  Raises:
    KeyError: If the archive has no such member.
    ValueError: If the member expands past the budget, or the archive is
      malformed.
  """
  chunks = []
  try:
    with z.open(member) as f:
      while True:
        chunk = f.read(_ZIP_READ_CHUNK_BYTES)
        if not chunk:
          break
        budget -= len(chunk)
        if budget < 0:
          raise ValueError(
              "Skill archive is too large decompressed: it expands past the"
              f" limit of {_MAX_ZIP_UNCOMPRESSED_BYTES} bytes."
          )
        chunks.append(chunk)
  except zipfile.BadZipFile as e:
    raise ValueError(f"Skill archive is malformed: {e}") from e
  return b"".join(chunks), budget


def _load_skill_from_zip_bytes(zip_bytes: bytes) -> models.Skill:
  """Load a complete skill directly from in-memory zip file bytes.

  Args:
    zip_bytes: The raw bytes of the zip file containing the skill.

  Returns:
    Skill object with all components loaded.

  Raises:
    FileNotFoundError: If SKILL.md is not found in the archive.
    ValueError: If SKILL.md is invalid, the archive contains dangerous paths,
      the archive is malformed, or it expands past the entry or decompressed
      size limits.
  """
  try:
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
  except zipfile.BadZipFile as e:
    raise ValueError(f"Skill archive is malformed: {e}") from e

  with archive as z:
    # zipfile truncates each member's read to the size its central-directory
    # entry declares, so capping the declared total is what bounds the bytes
    # decompressed out of the archive.
    entry_count = len(z.infolist())
    if entry_count > _MAX_ZIP_ENTRIES:
      raise ValueError(
          f"Skill archive has too many entries: {entry_count} exceeds the"
          f" limit of {_MAX_ZIP_ENTRIES}."
      )
    declared_size = sum(info.file_size for info in z.infolist())
    if declared_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
      raise ValueError(
          f"Skill archive is too large decompressed: {declared_size} bytes"
          f" exceeds the limit of {_MAX_ZIP_UNCOMPRESSED_BYTES} bytes."
      )
    budget = _MAX_ZIP_UNCOMPRESSED_BYTES

    # Security check for zip slip
    for member in z.infolist():
      filename = member.filename
      if (
          filename.startswith("/")
          or filename.startswith("../")
          or "/../" in filename
      ):
        raise ValueError(f"Dangerous zip entry ignored: {filename}")

    # Find SKILL.md or skill.md
    skill_md_content = None
    for name in ("SKILL.md", "skill.md"):
      try:
        skill_md_bytes, budget = _read_zip_member(z, name, budget)
      except KeyError:
        continue
      skill_md_content = skill_md_bytes.decode("utf-8")
      break

    if skill_md_content is None:
      raise FileNotFoundError("SKILL.md not found in zipped filesystem.")

    parsed, body = _parse_skill_md_content(skill_md_content)
    skill_name = parsed.get("name")
    if not skill_name:
      raise ValueError("SKILL.md frontmatter must contain 'name'")
    if (
        not isinstance(skill_name, str)
        or pathlib.Path(skill_name).name != skill_name
    ):
      raise ValueError(f"Invalid skill name in SKILL.md: {skill_name}")

    frontmatter = models.Frontmatter.model_validate(parsed)

    # Helper to load files under a directory prefix inside the zip
    def _load_zip_dir(prefix: str) -> dict[str, Union[str, bytes]]:
      nonlocal budget
      result: dict[str, Union[str, bytes]] = {}
      if not prefix.endswith("/"):
        prefix += "/"
      for info in z.infolist():
        if info.is_dir():
          continue
        if info.filename.startswith(prefix):
          # Avoid cache files or similar
          if "__pycache__" in info.filename:
            continue
          relative_path = info.filename[len(prefix) :]
          if not relative_path:
            continue
          data, budget = _read_zip_member(z, info, budget)
          try:
            result[relative_path] = data.decode("utf-8")
          except UnicodeDecodeError:
            result[relative_path] = data
      return result

    references = _load_zip_dir("references")
    assets = _load_zip_dir("assets")
    scripts = _build_scripts(_load_zip_dir("scripts"))

    resources = models.Resources(
        references=references,
        assets=assets,
        scripts=scripts,
    )

    return models.Skill(
        frontmatter=frontmatter,
        instructions=body,
        resources=resources,
    )


def _validate_skill_dir(
    skill_dir: Union[str, pathlib.Path],
) -> list[str]:
  """Validate a skill directory without fully loading it.

  Checks that the directory exists, contains a valid SKILL.md with correct
  frontmatter, and that the skill name matches the directory name.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    List of problem strings. Empty list means the skill is valid.
  """
  problems: list[str] = []
  skill_dir = pathlib.Path(skill_dir).resolve()

  if not skill_dir.exists():
    return [f"Directory '{skill_dir}' does not exist."]
  if not skill_dir.is_dir():
    return [f"'{skill_dir}' is not a directory."]

  skill_md = None
  for name in ("SKILL.md", "skill.md"):
    path = skill_dir / name
    if path.exists():
      skill_md = path
      break
  if skill_md is None:
    return [f"SKILL.md not found in '{skill_dir}'."]

  try:
    parsed, _, _ = _parse_skill_md(skill_dir)
  except (FileNotFoundError, ValueError) as e:
    return [str(e)]

  unknown = set(parsed.keys()) - _ALLOWED_FRONTMATTER_KEYS
  if unknown:
    problems.append(f"Unknown frontmatter fields: {sorted(unknown)}")

  try:
    frontmatter = models.Frontmatter.model_validate(parsed)
  except ValidationError as e:
    problems.append(f"Frontmatter validation error: {e}")
    return problems

  if skill_dir.name != frontmatter.name:
    problems.append(
        f"Skill name '{frontmatter.name}' does not match directory"
        f" name '{skill_dir.name}'."
    )

  return problems


def _read_skill_properties(
    skill_dir: Union[str, pathlib.Path],
) -> models.Frontmatter:
  """Read only the frontmatter properties from a skill directory.

  This is a lightweight alternative to ``load_skill_from_dir`` when you
  only need the skill metadata without loading instructions or resources.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    Frontmatter object with the skill's metadata.

  Raises:
    FileNotFoundError: If the directory or SKILL.md is not found.
    ValueError: If the frontmatter is invalid.
  """
  skill_dir = pathlib.Path(skill_dir).resolve()
  parsed, _, _ = _parse_skill_md(skill_dir)
  return models.Frontmatter.model_validate(parsed)


def _list_skills_in_dir(
    skills_base_path: Union[str, pathlib.Path],
) -> dict[str, models.Frontmatter]:
  """List skills in a local directory.

  Args:
    skills_base_path: Path to the base directory containing skills.

  Returns:
    Dictionary mapping skill IDs to their frontmatter.
  """
  skills_base_path = pathlib.Path(skills_base_path).resolve()
  skills = {}

  if not skills_base_path.is_dir():
    logging.warning(
        "Skills base path '%s' is not a directory.", skills_base_path
    )
    return skills

  for skill_dir in sorted(skills_base_path.iterdir()):
    if not skill_dir.is_dir():
      continue

    skill_id = skill_dir.name
    try:
      frontmatter = _read_skill_properties(skill_dir)
      if skill_id != frontmatter.name:
        raise ValueError(
            f"Skill name '{frontmatter.name}' does not match directory"
            f" name '{skill_id}'."
        )
      skills[skill_id] = frontmatter
    except (FileNotFoundError, ValueError, ValidationError) as e:
      # log invalid skills during listing and skip them
      logging.warning(
          "Skipping invalid skill '%s' in directory '%s': %s",
          skill_id,
          skills_base_path,
          e,
      )
  return skills


def _list_skills_in_gcs_dir(
    bucket_name: str,
    skills_base_path: str = "",
    project_id: str | None = None,
    credentials: auth.Credentials | None = None,
) -> Dict[str, models.Frontmatter]:
  """List skills in a GCS directory.

  Args:
    bucket_name: Name of the GCS bucket.
    skills_base_path: Base directory within the bucket (e.g., 'path/to/skills').

  Returns:
    Dictionary mapping skill IDs to their frontmatter.
  """
  try:
    from google.cloud import storage
  except ImportError as e:
    raise ImportError(
        "google-cloud-storage is required to list skills in GCS. Install it"
        " with `pip install google-cloud-storage` or `pip install"
        " google-adk[gcp]`."
    ) from e

  client = storage.Client(project=project_id, credentials=credentials)
  bucket = client.bucket(bucket_name)

  base_prefix = skills_base_path.strip("/")
  if base_prefix:
    base_prefix += "/"

  iterator = bucket.list_blobs(prefix=base_prefix, delimiter="/")
  # We must consume the iterator to populate the prefixes attribute
  for _ in iterator:
    pass
  logging.info("Found %s skills in GCS.", iterator.prefixes)

  skills = {}
  for skill_prefix in sorted(iterator.prefixes):
    manifest_blob = bucket.blob(f"{skill_prefix}SKILL.md")

    if manifest_blob.exists():
      content = manifest_blob.download_as_text()
      skill_id = skill_prefix.rstrip("/").split("/")[-1]
      try:
        parsed, _ = _parse_skill_md_content(content)
        frontmatter = models.Frontmatter.model_validate(parsed)
        skills[skill_id] = frontmatter
      except (ValueError, ValidationError) as e:
        # log invalid skills during listing and skip them
        logging.warning(
            "Skipping invalid skill '%s' in bucket '%s': %s",
            skill_id,
            bucket_name,
            e,
        )
  return skills


def _load_skill_from_gcs_dir(
    bucket_name: str,
    skill_id: str,
    skills_base_path: str = "",
    project_id: str | None = None,
    credentials: auth.Credentials | None = None,
) -> models.Skill:
  """Load a complete skill from a GCS directory.

  Args:
    bucket_name: Name of the GCS bucket.
    skill_id: The ID of the skill (directory name).
    skills_base_path: Base directory within the bucket (e.g., 'path/to/skills').
    project_id: Project ID to use for GCS client.
    credentials: Credentials to use for GCS client.

  Returns:
    Skill object with all components loaded.

  Raises:
    FileNotFoundError: If the skill directory or SKILL.md is not found.
    ValueError: If SKILL.md is invalid or the skill name does not match
      the directory name.
  """
  try:
    from google.cloud import storage
  except ImportError as e:
    raise ImportError(
        "google-cloud-storage is required to load skills from GCS. Install it"
        " with `pip install google-cloud-storage` or `pip install"
        " google-adk[gcp]`."
    ) from e

  client = storage.Client(project=project_id, credentials=credentials)
  bucket = client.bucket(bucket_name)

  base_prefix = skills_base_path.strip("/")
  if base_prefix:
    base_prefix += "/"

  skill_dir_prefix = f"{base_prefix}{skill_id}/"
  manifest_blob = bucket.blob(f"{skill_dir_prefix}SKILL.md")

  if not manifest_blob.exists():
    raise FileNotFoundError(
        f"SKILL.md not found at gs://{bucket_name}/{skill_dir_prefix}SKILL.md"
    )

  content = manifest_blob.download_as_text()
  parsed, body = _parse_skill_md_content(content)
  frontmatter = models.Frontmatter.model_validate(parsed)

  # Validate that skill name matches the directory name
  skill_name_expected = skill_id.strip("/").split("/")[-1]
  if skill_name_expected != frontmatter.name:
    raise ValueError(
        f"Skill name '{frontmatter.name}' does not match directory"
        f" name '{skill_name_expected}'."
    )

  def _load_files_in_dir(subdir: str) -> Dict[str, Union[str, bytes]]:
    prefix = f"{skill_dir_prefix}{subdir}/"
    blobs = bucket.list_blobs(prefix=prefix)
    result = {}

    for blob in blobs:
      relative_path = blob.name[len(prefix) :]
      if not relative_path:
        continue

      try:
        result[relative_path] = blob.download_as_text()
      except UnicodeDecodeError:
        result[relative_path] = blob.download_as_bytes()
    return result

  references = _load_files_in_dir("references")
  assets = _load_files_in_dir("assets")
  scripts = _build_scripts(_load_files_in_dir("scripts"))

  resources = models.Resources(
      references=references,
      assets=assets,
      scripts=scripts,
  )

  return models.Skill(
      frontmatter=frontmatter,
      instructions=body,
      resources=resources,
  )


async def _load_skill_from_dir_async(
    skill_dir: str | pathlib.Path,
) -> models.Skill:
  """Load a complete skill from a directory asynchronously.

  Runs the blocking :func:`_load_skill_from_dir` in a worker thread so the
  calling event loop stays responsive.

  Args:
    skill_dir: Path to the skill directory.

  Returns:
    Skill object with all components loaded.

  Raises:
    FileNotFoundError: If the skill directory or SKILL.md is not found.
    ValueError: If SKILL.md is invalid or the skill name does not match
      the directory name.
  """
  return await asyncio.to_thread(_load_skill_from_dir, skill_dir)


async def _load_skills_from_dir_async(
    skills_dir: str | pathlib.Path,
) -> list[models.Skill]:
  """Load all skills from subdirectories within a directory asynchronously.

  Runs the blocking :func:`_load_skills_from_dir` in a worker thread so the
  calling event loop stays responsive. The whole directory walk happens in a
  single worker thread rather than one thread per skill, so ordering and error
  behavior match the synchronous version exactly.

  Args:
    skills_dir: Path to the directory containing skill folders.

  Returns:
    List of Skill objects loaded from valid skill directories.

  Raises:
    FileNotFoundError: If skills_dir does not exist.
    ValueError: If skills_dir is not a directory, or if any skill fails
      validation.
  """
  return await asyncio.to_thread(_load_skills_from_dir, skills_dir)


async def _load_skill_from_gcs_dir_async(
    bucket_name: str,
    skill_id: str,
    skills_base_path: str = "",
    project_id: str | None = None,
    credentials: auth.Credentials | None = None,
) -> models.Skill:
  """Load a complete skill from a GCS directory asynchronously.

  Runs the blocking :func:`_load_skill_from_gcs_dir` in a worker thread so the
  calling event loop stays responsive.

  Args:
    bucket_name: Name of the GCS bucket.
    skill_id: The ID of the skill (directory name).
    skills_base_path: Base directory within the bucket (e.g., 'path/to/skills').
    project_id: Project ID to use for GCS client.
    credentials: Credentials to use for GCS client.

  Returns:
    Skill object with all components loaded.

  Raises:
    ImportError: If google-cloud-storage is not installed.
    FileNotFoundError: If the skill directory or SKILL.md is not found.
    ValueError: If SKILL.md is invalid or the skill name does not match
      the directory name.
  """
  return await asyncio.to_thread(
      _load_skill_from_gcs_dir,
      bucket_name,
      skill_id,
      skills_base_path,
      project_id,
      credentials,
  )


async def _list_skills_in_dir_async(
    skills_base_path: str | pathlib.Path,
) -> dict[str, models.Frontmatter]:
  """List skills in a local directory asynchronously.

  Runs the blocking :func:`_list_skills_in_dir` in a worker thread so the
  calling event loop stays responsive.

  Args:
    skills_base_path: Path to the base directory containing skills.

  Returns:
    Dictionary mapping skill IDs to their frontmatter. Invalid skills are
    logged and skipped.
  """
  return await asyncio.to_thread(_list_skills_in_dir, skills_base_path)


async def _list_skills_in_gcs_dir_async(
    bucket_name: str,
    skills_base_path: str = "",
    project_id: str | None = None,
    credentials: auth.Credentials | None = None,
) -> dict[str, models.Frontmatter]:
  """List skills in a GCS directory asynchronously.

  Runs the blocking :func:`_list_skills_in_gcs_dir` in a worker thread so the
  calling event loop stays responsive.

  Args:
    bucket_name: Name of the GCS bucket.
    skills_base_path: Base directory within the bucket (e.g., 'path/to/skills').
    project_id: Project ID to use for GCS client.
    credentials: Credentials to use for GCS client.

  Returns:
    Dictionary mapping skill IDs to their frontmatter. Invalid skills are
    logged and skipped.

  Raises:
    ImportError: If google-cloud-storage is not installed.
  """
  return await asyncio.to_thread(
      _list_skills_in_gcs_dir,
      bucket_name,
      skills_base_path,
      project_id,
      credentials,
  )

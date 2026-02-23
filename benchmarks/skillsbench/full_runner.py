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

"""Full SkillsBench runner — Docker-based with real pytest scoring.

Builds Docker images per-task from each task's environment/Dockerfile,
runs agent scripts inside the container (where /root/ data files exist),
then runs the real pytest tests inside the same container for binary
pass/fail scoring matching SkillsBench methodology.

Usage:
    python -u -m benchmarks.skillsbench.full_runner
    python -u -m benchmarks.skillsbench.full_runner --filter citation-check
    python -u -m benchmarks.skillsbench.full_runner --build-only
    python -u -m benchmarks.skillsbench.full_runner --skip-tests

Environment variables:
    GOOGLE_API_KEY              — API key for authentication
    GOOGLE_GENAI_USE_VERTEXAI   — Set to 1 for Vertex AI backend
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import pathlib
import re
import sys
import tarfile
import time
import tomllib
from typing import Any
import uuid

import docker
from google.adk import Agent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors.code_execution_utils import CodeExecutionInput
from google.adk.code_executors.code_execution_utils import CodeExecutionResult
from google.adk.evaluation.eval_case import get_all_tool_calls
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.events.event import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.skills import load_skill_from_dir
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Resources
from google.adk.skills.models import Script
from google.adk.skills.models import Skill
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.utils.context_utils import Aclosing
from google.genai import types as genai_types
from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_TASKS_DIR = pathlib.Path("/tmp/skillsbench/tasks")
_MODEL_NAME = "gemini-3-flash-preview"

# ── Timeouts (seconds) ──────────────────────────────────────────────
_DEFAULT_AGENT_TIMEOUT = 900.0
_DEFAULT_BUILD_TIMEOUT = 600.0
_DEFAULT_VERIFIER_TIMEOUT = 900.0


# ── helpers ──────────────────────────────────────────────────────────


def _slugify(name: str) -> str:
  """Convert a name to lowercase kebab-case.

  Lowercases, replaces underscores and spaces with hyphens,
  strips non-alphanumeric/non-hyphen chars, collapses runs
  of hyphens, and trims leading/trailing hyphens.
  """
  s = name.lower()
  s = s.replace("_", "-").replace(" ", "-")
  s = re.sub(r"[^a-z0-9-]", "", s)
  s = re.sub(r"-{2,}", "-", s)
  return s.strip("-")


def _load_dir(directory: pathlib.Path) -> dict[str, str]:
  """Recursively load files from a directory into a dict.

  Mirrors google.adk.skills._utils._load_dir so the runner
  does not depend on a private import.
  """
  files: dict[str, str] = {}
  if directory.exists() and directory.is_dir():
    for file_path in directory.rglob("*"):
      if "__pycache__" in file_path.parts:
        continue
      if file_path.is_file():
        relative = file_path.relative_to(directory)
        try:
          files[str(relative)] = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
          continue
  return files


# ── Docker prereq ────────────────────────────────────────────────────


def ensure_docker_running() -> docker.DockerClient:
  """Return a Docker client or fail fast with a clear message."""
  try:
    client = docker.from_env()
    client.ping()
    return client
  except Exception as exc:
    print(
        "\nERROR: Docker is not running or not accessible.\n"
        "Please start Docker Desktop and try again.\n"
        f"  Detail: {exc}\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ── TaskContainerExecutor ────────────────────────────────────────────


class TaskContainerExecutor(BaseCodeExecutor):
  """Code executor that runs Python inside a Docker container.

  Unlike ADK's ContainerCodeExecutor (which uses atexit for
  cleanup), this gives direct lifecycle control for test
  injection after the agent finishes.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  stateful: bool = Field(default=False, frozen=True, exclude=True)
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  _container: object = None
  _client: object = None

  def start(
      self,
      image_tag: str,
      client: docker.DockerClient,
  ) -> None:
    """Start a detached container from *image_tag*."""
    self._client = client
    self._container = client.containers.run(
        image_tag,
        command="sleep infinity",
        detach=True,
        tty=True,
        working_dir="/root",
    )

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """Run ``python3 -c <code>`` inside the container."""
    if self._container is None:
      return CodeExecutionResult(
          stdout="",
          stderr="Container not started",
      )
    code = code_execution_input.code
    rc, output = self._container.exec_run(
        ["python3", "-c", code],
        workdir="/root",
        demux=True,
    )
    stdout = (output[0] or b"").decode("utf-8", errors="replace")
    stderr = (output[1] or b"").decode("utf-8", errors="replace")
    return CodeExecutionResult(
        stdout=stdout,
        stderr=stderr,
        output_files=[],
    )

  def copy_to_container(
      self,
      local_path: pathlib.Path,
      container_path: str,
  ) -> None:
    """Copy a local directory into the container via put_archive."""
    if self._container is None:
      raise RuntimeError("Container not started")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
      tar.add(str(local_path), arcname=".")
    buf.seek(0)
    self._container.put_archive(container_path, buf)

  def exec_in_container(
      self,
      cmd: list[str],
      workdir: str = "/root",
      timeout: float | None = None,
  ) -> tuple[int, str]:
    """Run an arbitrary command inside the container.

    Returns (exit_code, combined_output).
    """
    if self._container is None:
      raise RuntimeError("Container not started")
    rc, output = self._container.exec_run(
        cmd,
        workdir=workdir,
        demux=True,
    )
    stdout = (output[0] or b"").decode("utf-8", errors="replace")
    stderr = (output[1] or b"").decode("utf-8", errors="replace")
    combined = stdout
    if stderr:
      combined = combined + "\n" + stderr if combined else stderr
    return rc, combined

  def read_file_from_container(self, path: str) -> str | None:
    """Read a single file from the container, or None."""
    if self._container is None:
      return None
    try:
      bits, _ = self._container.get_archive(path)
      buf = io.BytesIO()
      for chunk in bits:
        buf.write(chunk)
      buf.seek(0)
      with tarfile.open(fileobj=buf, mode="r") as tar:
        for member in tar.getmembers():
          if member.isfile():
            f = tar.extractfile(member)
            if f:
              return f.read().decode("utf-8", errors="replace")
    except Exception:
      return None
    return None

  def stop(self) -> None:
    """Stop and remove the container."""
    if self._container is not None:
      try:
        self._container.stop(timeout=5)
      except Exception:
        pass
      try:
        self._container.remove(force=True)
      except Exception:
        pass
      self._container = None


# ── Docker image builder ─────────────────────────────────────────────


def build_task_image(
    task_dir: pathlib.Path,
    client: docker.DockerClient,
    *,
    rebuild: bool = False,
    build_timeout: float = _DEFAULT_BUILD_TIMEOUT,
) -> str:
  """Build (or reuse) a Docker image for one task.

  Returns the image tag.
  """
  tag = f"skillsbench-{task_dir.name}:latest"
  if not rebuild:
    try:
      client.images.get(tag)
      return tag
    except docker.errors.ImageNotFound:
      pass

  build_ctx = task_dir / "environment"
  if not (build_ctx / "Dockerfile").exists():
    raise FileNotFoundError(f"No Dockerfile in {build_ctx}")

  logger.info("Building image %s …", tag)
  client.images.build(
      path=str(build_ctx),
      tag=tag,
      rm=True,
      timeout=int(build_timeout),
  )
  return tag


# ── Lenient skill loader ─────────────────────────────────────────────


def _load_skill_lenient(skill_dir: pathlib.Path) -> Skill | None:
  """Load a skill, falling back to lenient parsing on failure.

  Tries the strict ``load_skill_from_dir`` first.  On any error,
  manually parses SKILL.md, fixes the frontmatter fields, and
  builds a ``Skill`` directly.

  Returns None only for truly unrecoverable cases (no SKILL.md,
  unparseable YAML, no description).
  """
  # ── strict path ──
  try:
    return load_skill_from_dir(skill_dir)
  except Exception:
    pass

  # ── lenient path ──
  skill_md = None
  for name in ("SKILL.md", "skill.md"):
    p = skill_dir / name
    if p.exists():
      skill_md = p
      break
  if skill_md is None:
    return None

  try:
    content = skill_md.read_text(encoding="utf-8")
  except Exception:
    return None

  if not content.startswith("---"):
    return None

  parts = content.split("---", 2)
  if len(parts) < 3:
    return None

  try:
    parsed = yaml.safe_load(parts[1])
  except yaml.YAMLError:
    return None

  if not isinstance(parsed, dict):
    return None

  body = parts[2].strip()

  # Fix name: slugify and force to match dir name
  parsed["name"] = _slugify(skill_dir.name)

  # Fix allowed-tools: list → comma-separated string
  for key in ("allowed-tools", "allowed_tools"):
    val = parsed.get(key)
    if isinstance(val, list):
      parsed[key] = ", ".join(str(v) for v in val)

  # Fix compatibility: dict → JSON string
  if isinstance(parsed.get("compatibility"), dict):
    parsed["compatibility"] = json.dumps(parsed["compatibility"])

  # Fix metadata values: ensure all are strings
  meta = parsed.get("metadata")
  if isinstance(meta, dict):
    parsed["metadata"] = {
        str(k): str(v) if not isinstance(v, str) else v for k, v in meta.items()
    }

  # Ensure description exists
  if not parsed.get("description"):
    parsed["description"] = f"Skill from {skill_dir.name}"

  try:
    frontmatter = Frontmatter.model_validate(parsed)
  except Exception:
    return None

  # Load resources
  references = _load_dir(skill_dir / "references")
  assets = _load_dir(skill_dir / "assets")
  raw_scripts = _load_dir(skill_dir / "scripts")
  scripts = {n: Script(src=c) for n, c in raw_scripts.items()}
  resources = Resources(
      references=references,
      assets=assets,
      scripts=scripts,
  )

  return Skill(
      frontmatter=frontmatter,
      instructions=body,
      resources=resources,
  )


def load_task_skills(
    task_dir: pathlib.Path,
) -> tuple[list[Skill], list[str]]:
  """Load all skills from a task's environment/skills/.

  Uses lenient loading to handle malformed frontmatter.

  Returns:
    (loaded_skills, error_messages)
  """
  skills_dir = task_dir / "environment" / "skills"
  if not skills_dir.exists() or not skills_dir.is_dir():
    return [], ["No environment/skills/ directory found"]

  skills: list[Skill] = []
  errors: list[str] = []
  for sd in sorted(skills_dir.iterdir()):
    if not sd.is_dir():
      continue
    skill = _load_skill_lenient(sd)
    if skill is not None:
      skills.append(skill)
    else:
      errors.append(f"{sd.name}: could not load")
  return skills, errors


# ── task.toml parser ─────────────────────────────────────────────────


def parse_task_config(
    task_dir: pathlib.Path,
) -> dict[str, float]:
  """Read task.toml and return timeout values.

  Returns dict with keys: agent_timeout, build_timeout,
  verifier_timeout.
  """
  toml_path = task_dir / "task.toml"
  defaults = {
      "agent_timeout": _DEFAULT_AGENT_TIMEOUT,
      "build_timeout": _DEFAULT_BUILD_TIMEOUT,
      "verifier_timeout": _DEFAULT_VERIFIER_TIMEOUT,
  }
  if not toml_path.exists():
    return defaults

  try:
    with open(toml_path, "rb") as f:
      cfg = tomllib.load(f)
  except Exception:
    return defaults

  agent_sec = cfg.get("agent", {})
  env_sec = cfg.get("environment", {})
  ver_sec = cfg.get("verifier", {})

  return {
      "agent_timeout": float(
          agent_sec.get("timeout_sec", defaults["agent_timeout"])
      ),
      "build_timeout": float(
          env_sec.get("build_timeout_sec", defaults["build_timeout"])
      ),
      "verifier_timeout": float(
          ver_sec.get("timeout_sec", defaults["verifier_timeout"])
      ),
  }


# ── scoring ──────────────────────────────────────────────────────────


def score_task_in_container(
    executor: TaskContainerExecutor,
    task_dir: pathlib.Path,
) -> int:
  """Run the real pytest tests inside the container.

  1. Copy task_dir/tests/ into /tests/ in the container.
  2. Run ``bash /tests/test.sh``.
  3. Read /logs/verifier/reward.txt — ``1`` = PASS.

  Returns 1 (pass) or 0 (fail).
  """
  tests_dir = task_dir / "tests"
  if not tests_dir.exists():
    logger.warning("No tests/ directory for %s", task_dir.name)
    return 0

  # Ensure /tests/ directory exists in container
  executor.exec_in_container(["mkdir", "-p", "/tests"])

  # Copy tests into container
  executor.copy_to_container(tests_dir, "/tests")

  # Make test.sh executable
  executor.exec_in_container(["chmod", "+x", "/tests/test.sh"])

  # Run test.sh
  rc, output = executor.exec_in_container(
      ["bash", "/tests/test.sh"],
      workdir="/root",
  )
  logger.debug(
      "test.sh for %s exited %d:\n%s",
      task_dir.name,
      rc,
      output[:2000],
  )

  # Read reward
  reward = executor.read_file_from_container("/logs/verifier/reward.txt")
  if reward is not None and reward.strip() == "1":
    return 1
  return 0


def score_task_heuristic(
    actual_invocations: list[Invocation],
) -> int:
  """Heuristic fallback (--skip-tests mode).

  Passes if the agent loaded a skill, used it, and
  produced a final response.
  """
  tool_names: list[str] = []
  for inv in actual_invocations:
    for tc in get_all_tool_calls(inv.intermediate_data):
      tool_names.append(tc.name)

  has_load = "load_skill" in tool_names
  has_use = (
      "execute_skill_script" in tool_names
      or "load_skill_resource" in tool_names
  )
  has_response = False
  for inv in actual_invocations:
    if inv.final_response and inv.final_response.parts:
      for part in inv.final_response.parts:
        if part.text:
          has_response = True
          break
    if has_response:
      break

  return 1 if (has_load and has_use and has_response) else 0


# ── container bash tool ───────────────────────────────────────────


class ContainerBashTool(BaseTool):
  """Simple bash tool that runs commands inside the task container."""

  def __init__(self, executor: TaskContainerExecutor):
    super().__init__(
        name="bash",
        description=(
            "Run a bash command inside the task environment."
            " Use this to read files, write files, install"
            " packages, or run arbitrary shell commands."
        ),
    )
    self._executor = executor

  def _get_declaration(self) -> genai_types.FunctionDeclaration | None:
    return genai_types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["command"],
        },
    )

  async def run_async(self, *, args, tool_context) -> Any:
    command = args.get("command", "")
    if not command:
      return {"error": "command is required"}
    rc, output = self._executor.exec_in_container(
        ["bash", "-c", command],
    )
    # Truncate very long output to avoid blowing context
    if len(output) > 10000:
      output = output[:10000] + "\n... (truncated)"
    return {
        "exit_code": rc,
        "output": output,
    }


# ── agent builder ────────────────────────────────────────────────────


def build_agent(
    skills: list[Skill],
    executor: TaskContainerExecutor,
) -> Agent:
  """Create a fresh agent with the given skills."""
  toolset = SkillToolset(
      skills=skills,
      code_executor=executor,
  )
  bash_tool = ContainerBashTool(executor)
  model = Gemini(
      model=_MODEL_NAME,
      retry_options=genai_types.HttpRetryOptions(
          attempts=5,
          initialDelay=2.0,
          maxDelay=60.0,
          expBase=2.0,
          httpStatusCodes=[429, 503],
      ),
  )
  return Agent(
      model=model,
      name="skillsbench_agent",
      description=(
          "An agent that completes tasks by discovering and using"
          " available skills from the SkillsBench benchmark."
      ),
      instruction=(
          "You are an agent that completes tasks by discovering"
          " and using available skills. Follow this workflow:\n"
          "1. Use list_skills to find relevant skills.\n"
          "2. Use load_skill to read the skill's instructions"
          " carefully.\n"
          "3. Use load_skill_resource to examine references or"
          " sample data if available.\n"
          "4. Use execute_skill_script to run the skill's"
          " scripts with appropriate arguments.\n"
          "5. Use bash to read/write files, install packages,"
          " or run any shell command in the environment.\n"
          "6. Write the required output files and present a"
          " clear answer.\n"
          "\nAlways check skill instructions before executing"
          " scripts."
      ),
      generate_content_config=genai_types.GenerateContentConfig(
          temperature=1.0,
          thinking_config=genai_types.ThinkingConfig(
              thinking_level="HIGH",
          ),
      ),
      tools=[toolset, bash_tool],
  )


# ── agent runner ─────────────────────────────────────────────────────


async def run_task(
    agent: Agent,
    user_query: str,
) -> list[Invocation]:
  """Run a single task and return invocations."""
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  memory_service = InMemoryMemoryService()

  app_name = "skillsbench_full_eval"
  user_id = "eval_user"
  session_id = str(uuid.uuid4())

  await session_service.create_session(
      app_name=app_name,
      user_id=user_id,
      state={},
      session_id=session_id,
  )

  user_content = genai_types.Content(
      role="user",
      parts=[genai_types.Part.from_text(text=user_query)],
  )

  async with Runner(
      app_name=app_name,
      agent=agent,
      artifact_service=artifact_service,
      session_service=session_service,
      memory_service=memory_service,
  ) as runner:
    events: list[Event] = []
    async with Aclosing(
        runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        )
    ) as agen:
      invocation_id = None
      async for event in agen:
        if not invocation_id:
          invocation_id = event.invocation_id
          events.append(
              Event(
                  content=user_content,
                  author="user",
                  invocation_id=invocation_id,
              )
          )
        events.append(event)

  return EvaluationGenerator.convert_events_to_eval_invocations(events)


# ── task result ──────────────────────────────────────────────────────


class TaskResult:
  """Result of a single task evaluation."""

  def __init__(
      self,
      task_name: str,
      score: int,
      num_skills: int,
      elapsed: float,
      error: str | None = None,
  ):
    self.task_name = task_name
    self.score = score
    self.num_skills = num_skills
    self.elapsed = elapsed
    self.error = error


# ── task discovery ───────────────────────────────────────────────────


def discover_tasks(
    tasks_dir: pathlib.Path,
    filter_pattern: str | None = None,
) -> list[pathlib.Path]:
  """Find task directories containing instruction.md."""
  tasks = []
  for d in sorted(tasks_dir.iterdir()):
    if not d.is_dir():
      continue
    if not (d / "instruction.md").exists():
      continue
    if filter_pattern and filter_pattern not in d.name:
      continue
    tasks.append(d)
  return tasks


# ── evaluate one task ────────────────────────────────────────────────


async def evaluate_task(
    task_dir: pathlib.Path,
    client: docker.DockerClient,
    *,
    timeout_override: float | None = None,
    rebuild: bool = False,
    skip_tests: bool = False,
) -> TaskResult:
  """Evaluate a single SkillsBench task end-to-end.

  1. Parse task.toml for timeouts
  2. Build Docker image (or use cached)
  3. Load skills with lenient loader
  4. Start TaskContainerExecutor
  5. Build Agent with SkillToolset
  6. Run agent with instruction.md
  7. Run tests inside same container (or heuristic)
  8. Stop container
  """
  task_name = task_dir.name
  start = time.time()
  executor = TaskContainerExecutor()
  num_skills = 0
  agent_timeout = _DEFAULT_AGENT_TIMEOUT

  try:
    # 1. Parse config
    config = parse_task_config(task_dir)
    agent_timeout = timeout_override or config["agent_timeout"]

    # 2. Build image
    try:
      image_tag = build_task_image(
          task_dir,
          client,
          rebuild=rebuild,
          build_timeout=config["build_timeout"],
      )
    except Exception as exc:
      return TaskResult(
          task_name=task_name,
          score=0,
          num_skills=0,
          elapsed=time.time() - start,
          error=f"image build: {str(exc)[:100]}",
      )

    # 3. Load instruction
    try:
      user_query = (
          (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
      )
    except Exception as exc:
      return TaskResult(
          task_name=task_name,
          score=0,
          num_skills=0,
          elapsed=time.time() - start,
          error=f"instruction.md: {exc}",
      )

    # 4. Load skills
    skills, skill_errors = load_task_skills(task_dir)
    num_skills = len(skills)
    if not skills:
      msg = "; ".join(skill_errors) if skill_errors else "No skills"
      return TaskResult(
          task_name=task_name,
          score=0,
          num_skills=0,
          elapsed=time.time() - start,
          error=f"skills: {msg}",
      )

    # 5. Start container
    executor.start(image_tag, client)

    # 6. Build agent and run
    agent = build_agent(skills, executor)
    invocations = await asyncio.wait_for(
        run_task(agent, user_query),
        timeout=agent_timeout,
    )

    # 7. Score
    if skip_tests:
      score = score_task_heuristic(invocations)
    else:
      score = score_task_in_container(executor, task_dir)

    return TaskResult(
        task_name=task_name,
        score=score,
        num_skills=num_skills,
        elapsed=time.time() - start,
    )

  except Exception as exc:
    # Score container tests even on error — the agent
    # may have produced partial output files.
    score = 0
    if not skip_tests and executor._container is not None:
      try:
        score = score_task_in_container(executor, task_dir)
      except Exception:
        pass
    is_timeout = isinstance(exc, asyncio.TimeoutError)
    err_msg = f"timeout ({agent_timeout}s)" if is_timeout else str(exc)[:120]
    return TaskResult(
        task_name=task_name,
        score=score,
        num_skills=num_skills,
        elapsed=time.time() - start,
        error=err_msg,
    )
  finally:
    executor.stop()


# ── printing ─────────────────────────────────────────────────────────


def print_header():
  print()
  print("=" * 60)
  print("  SkillsBench Full Evaluation — Docker + Pytest")
  print("=" * 60)
  print()


def print_task_result(
    idx: int,
    total: int,
    result: TaskResult,
):
  mark = "PASS" if result.score == 1 else "FAIL"
  name = result.task_name[:35].ljust(35)
  if result.error:
    detail = f"({result.error})"
  else:
    detail = f"({result.num_skills} skills, {result.elapsed:.1f}s)"
  print(f"[{idx:>2}/{total}]  {name} {mark}  {detail}")
  sys.stdout.flush()


def print_summary(
    results: list[TaskResult],
    total_tasks: int,
    elapsed: float,
):
  passed = sum(1 for r in results if r.score == 1)
  loadable = sum(1 for r in results if r.num_skills > 0)
  times = [r.elapsed for r in results if r.error != "timeout"]
  avg_time = sum(times) / max(len(times), 1)
  pct = (passed / max(total_tasks, 1)) * 100

  print()
  print("=" * 60)
  print("  Leaderboard Summary")
  print("=" * 60)
  print(f"  Model:              {_MODEL_NAME}")
  print(f"  Framework:          ADK SkillToolset (Docker)")
  print(f"  Score:              {passed}/{total_tasks} ({pct:.1f}%)")
  print(f"  Loadable tasks:     {loadable}/{total_tasks}")
  print(f"  Avg time/task:      {avg_time:.1f}s")
  print(f"  Total elapsed:      {elapsed:.0f}s")
  print("=" * 60)


# ── full evaluation loop ─────────────────────────────────────────────


async def run_full_evaluation(
    tasks_dir: pathlib.Path,
    client: docker.DockerClient,
    *,
    timeout_override: float | None = None,
    filter_pattern: str | None = None,
    rebuild: bool = False,
    build_only: bool = False,
    skip_tests: bool = False,
) -> list[TaskResult]:
  """Run evaluation on all matching tasks."""
  task_dirs = discover_tasks(tasks_dir, filter_pattern)
  total = len(task_dirs)
  print(f"Found {total} tasks in {tasks_dir}\n")

  if build_only:
    for idx, td in enumerate(task_dirs, 1):
      name = td.name[:35].ljust(35)
      try:
        config = parse_task_config(td)
        tag = build_task_image(
            td,
            client,
            rebuild=rebuild,
            build_timeout=config["build_timeout"],
        )
        print(f"[{idx:>2}/{total}]  {name} OK  ({tag})")
      except Exception as exc:
        print(f"[{idx:>2}/{total}]  {name} FAIL  ({str(exc)[:80]})")
      sys.stdout.flush()
    return []

  results: list[TaskResult] = []
  for idx, td in enumerate(task_dirs, 1):
    result = await evaluate_task(
        td,
        client,
        timeout_override=timeout_override,
        rebuild=rebuild,
        skip_tests=skip_tests,
    )
    results.append(result)
    print_task_result(idx, total, result)

  return results


# ── CLI ──────────────────────────────────────────────────────────────


def main():
  parser = argparse.ArgumentParser(
      description="SkillsBench Docker-based evaluation runner",
  )
  parser.add_argument(
      "--tasks-dir",
      type=pathlib.Path,
      default=_DEFAULT_TASKS_DIR,
      help=(
          "Path to SkillsBench tasks directory"
          " (default: /tmp/skillsbench/tasks)"
      ),
  )
  parser.add_argument(
      "--timeout",
      type=float,
      default=None,
      help="Override per-task agent timeout in seconds",
  )
  parser.add_argument(
      "--filter",
      type=str,
      default=None,
      help="Only run tasks whose name contains PATTERN",
  )
  parser.add_argument(
      "--rebuild",
      action="store_true",
      help="Force Docker image rebuild",
  )
  parser.add_argument(
      "--build-only",
      action="store_true",
      help="Build Docker images only, don't run agents",
  )
  parser.add_argument(
      "--skip-tests",
      action="store_true",
      help="Use tool-call heuristic instead of pytest scoring",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  print_header()

  # Prereq check
  client = ensure_docker_running()

  start = time.time()
  results = asyncio.run(
      run_full_evaluation(
          tasks_dir=args.tasks_dir,
          client=client,
          timeout_override=args.timeout,
          filter_pattern=args.filter,
          rebuild=args.rebuild,
          build_only=args.build_only,
          skip_tests=args.skip_tests,
      )
  )
  elapsed = time.time() - start

  if results:
    print_summary(results, len(results), elapsed)


if __name__ == "__main__":
  main()

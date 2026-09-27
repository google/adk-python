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

"""Dev-only endpoints that deploy the current agent from the developer UI.

Supports the three hosted targets `adk deploy` offers: Agent Runtime,
Cloud Run and GKE.

Registered by `DevServer` and never by `ApiServer`. Deploying builds an image
from the agent source and pushes it using the developer's cloud credentials, so
these routes must not exist in a deployed container. That guarantee is
structural rather than a runtime check: the generated Dockerfile deletes
`dev_server.py` from the image (see `deployers/_dockerfile_template.py`), so a
deployed agent has no way to reach this module.

Each deploy runs as a child `adk deploy <target>` process instead of by
importing `cli_deploy` and calling it. That is not a stylistic choice:
`cli_deploy.to_agent_engine` calls `os.chdir()` twice (it documents that Agent
Runtime deployment uses relative paths). Working directory is process-global,
so running it inside the server would corrupt the cwd of every other request
for the several minutes a deploy takes. A child process gives it a cwd of its
own.

The `agent_engine` spelling survives in route paths, request fields and flags
because those name real CLI surface: the `adk deploy agent_engine` subcommand
and its `--agent_engine_id` flag. Prose calls the product Agent Runtime.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Literal
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import field_validator

from .utils import common

logger = logging.getLogger("google_adk." + __name__)

TAG_DEPLOY = "Deploy"

_IS_WINDOWS = os.name == "nt"
_GCLOUD_CMD = "gcloud.cmd" if _IS_WINDOWS else "gcloud"
_OUTPUT_CHUNK_BYTES = 64 * 1024
_OUTPUT_POLL_SECONDS = 0.25

# The response body is the deploy's own console output, followed by one final
# line: this marker, a space, and a JSON object. Clients split on the marker
# rather than scraping human-readable log lines, which change freely.
_RESULT_MARKER = "__ADK_DEPLOY_RESULT__"

# Agent Runtime resource name as printed by cli_deploy, e.g.
# "Created a new instance: projects/p/locations/us-central1/reasoningEngines/1".
_RESOURCE_NAME_RE = re.compile(
    r"projects/[^/\s]+/locations/[^/\s]+/reasoningEngines/[^/\s'\"]+"
)
# `gcloud run deploy` prints "Service URL: https://svc-abc123-uc.a.run.app".
_SERVICE_URL_RE = re.compile(r"https://[^\s'\"]+\.run\.app\b")
# ...and "...to Cloud Run service [svc] in project [my-project] region [r]",
# which is the only place the project appears when the user left it blank and
# the CLI resolved it from gcloud config.
_LOG_PROJECT_RE = re.compile(r"in project \[([^\]\s]+)\]")
# gcloud emboldens those bracketed values, so colour codes must come out first.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Every `adk deploy` subcommand prints this before giving up. GKE has no
# resource identifier to check for, so the message itself is the only evidence
# in the log that the deploy failed.
_DEPLOY_FAILED_RE = re.compile(r"^Deploy failed:", re.MULTILINE)

# Every request value below is interpolated into the child's argv. The argv is
# always a list and never a shell string, so shell metacharacters are inert;
# the residual risk is argument injection, where a value beginning with "-" is
# read as another flag. Each validator therefore rejects a leading "-" and any
# control character, on top of the shape check for that field.
_PROJECT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")
_REGION_RE = re.compile(r"^[a-z][a-z0-9-]{1,30}[a-z0-9]$")
_AGENT_ENGINE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_FULL_RESOURCE_NAME_RE = re.compile(
    r"^projects/[^/]+/locations/[^/]+/reasoningEngines/[^/]+$"
)
# Cloud Run service names and GKE cluster names share this shape.
_RESOURCE_ID_RE = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_TEXT_FIELD_CHARS = 256
_MAX_SERVICE_NAME_CHARS = 63
_MAX_CLUSTER_NAME_CHARS = 40

# `gcloud config get-value project` is normally instant, but the defaults
# lookup happens while the user waits for a dialog to open, so cap it.
_GCLOUD_LOOKUP_TIMEOUT_SECONDS = 5.0

# The Service that `cli_deploy.to_gke` generates listens on this port and
# forwards to the container. It is a literal in that manifest, so it is a
# literal here too; if the manifest changes, the port-forward hint below goes
# stale with it.
_GKE_SERVICE_PORT = 80

# Variables the deploy form can be prefilled from.
_PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
_REGION_ENV_VAR = "GOOGLE_CLOUD_LOCATION"

# The server's environment as it was before any agent was loaded.
#
# Deliberately a snapshot rather than a live read of os.environ.
# `envs.load_dotenv_for_agent` copies each agent's .env into the server's own
# environment with override=True as agents load, so a later read of os.environ
# returns whichever agent happened to load last, which would prefill this form
# with a different agent's project. This module is imported while the server
# starts, before any agent loads, so the snapshot holds only what the user
# actually exported.
_STARTUP_ENVIRON: dict[str, str] = dict(os.environ)

# One deploy per app at a time, whichever target it is aimed at: they stage the
# same source and, for Agent Runtime, a second run would call
# `client.agent_engines.create()` again and leave the developer paying for two
# instances, only one of which the UI knows about.
#
# Keyed by app name, valued by the task awaiting that app's child process, so
# the entry lives exactly as long as the deploy does, not as long as the
# request, which a client can end early by disconnecting. The value is None
# only for the moment between claiming the slot and the child being spawned.
# Holding the task here also keeps a strong reference to it, which the event
# loop does not.
_deploys_in_flight: dict[str, Optional["asyncio.Task[int]"]] = {}

DeployTarget = Literal["agent_engine", "cloud_run", "gke"]


def _reject_unsafe(field: str, value: str) -> str:
  """Rejects values that would be read as a flag or corrupt the log stream."""
  if _CONTROL_CHARS_RE.search(value):
    raise ValueError(f"{field} must not contain control characters")
  if value.startswith("-"):
    raise ValueError(
        f"{field} must not start with '-', which would be parsed as a flag"
    )
  return value


def _check_free_text(value: Optional[str]) -> Optional[str]:
  """Validates a human-readable field that still becomes an argv entry."""
  if value is None:
    return None
  value = value.strip()
  if not value:
    return None
  if len(value) > _MAX_TEXT_FIELD_CHARS:
    raise ValueError(f"value exceeds {_MAX_TEXT_FIELD_CHARS} characters")
  return _reject_unsafe("value", value)


class _BaseDeployRequest(common.BaseModel, abc.ABC):
  """Fields every deploy target needs."""

  region: str
  """Required. Google Cloud region, e.g. "us-central1".

  Required for all three targets even though the CLI flag is optional
  everywhere, because every target fails badly without it and in a different
  way. Agent Runtime cannot build a `vertexai.Client` and exits 0 having
  deployed nothing; `gcloud run deploy` prompts for a region interactively,
  which cannot be answered by a child process with no terminal; GKE cannot
  fetch cluster credentials.
  """

  project: Optional[str] = None
  """Google Cloud project. When omitted the CLI reads it from gcloud config."""

  @field_validator("region")
  @classmethod
  def _check_region(cls, value: str) -> str:
    _reject_unsafe("region", value)
    if not _REGION_RE.fullmatch(value):
      raise ValueError(f"invalid region: {value!r}")
    return value

  @field_validator("project")
  @classmethod
  def _check_project(cls, value: Optional[str]) -> Optional[str]:
    if value is None:
      return None
    _reject_unsafe("project", value)
    if not _PROJECT_RE.fullmatch(value):
      raise ValueError(f"invalid project id: {value!r}")
    return value

  target: ClassVar[DeployTarget]
  """Which `adk deploy` subcommand this request drives.

  Declared per subclass rather than passed alongside the request, so the two
  cannot disagree.
  """

  @abc.abstractmethod
  def target_flags(self) -> list[str]:
    """Returns the flags specific to this target, as argv entries."""

  @abc.abstractmethod
  def result_fields(self, *, log: str, exit_code: int) -> dict[str, Any]:
    """Returns the target-specific parts of a finished deploy's result.

    Each target identifies its deployment differently and has a different
    notion of what "succeeded" looks like, so each one reads its own answer
    out of the log rather than a central function switching on target.
    """

  def common_flags(self) -> list[str]:
    flags = ["--region", self.region]
    if self.project:
      flags += ["--project", self.project]
    return flags


class AgentEngineDeployRequest(_BaseDeployRequest):
  """Body of `POST /dev/apps/{app_name}/deploy/agent_engine`."""

  target: ClassVar[DeployTarget] = "agent_engine"

  display_name: Optional[str] = None
  """Display name of the Agent Runtime. Defaults to the agent folder name."""

  description: Optional[str] = None

  agent_engine_id: Optional[str] = None
  """Existing runtime to update; a bare id or a full resource name.

  When omitted the CLI creates a brand new Agent Runtime on every deploy, so a
  client that wants to update rather than accumulate instances must send back
  the resource name from the previous deploy's result line.
  """

  @field_validator("agent_engine_id")
  @classmethod
  def _check_agent_engine_id(cls, value: Optional[str]) -> Optional[str]:
    if value is None:
      return None
    _reject_unsafe("agent_engine_id", value)
    if not (
        _AGENT_ENGINE_ID_RE.fullmatch(value)
        or _FULL_RESOURCE_NAME_RE.fullmatch(value)
    ):
      raise ValueError(
          "agent_engine_id must be a bare id or a full resource name of the"
          " form projects/{project}/locations/{location}/reasoningEngines/{id}"
      )
    return value

  @field_validator("display_name", "description")
  @classmethod
  def _check_text(cls, value: Optional[str]) -> Optional[str]:
    return _check_free_text(value)

  def target_flags(self) -> list[str]:
    flags: list[str] = []
    if self.display_name:
      flags += ["--display_name", self.display_name]
    if self.description:
      flags += ["--description", self.description]
    if self.agent_engine_id:
      flags += ["--agent_engine_id", self.agent_engine_id]
    return flags

  def result_fields(self, *, log: str, exit_code: int) -> dict[str, Any]:
    matches = _RESOURCE_NAME_RE.findall(log)
    resource_name = matches[-1] if matches else None
    fields: dict[str, Any] = {"resourceName": resource_name}
    if exit_code == 0 and not resource_name:
      # Exit status alone does not mean the deploy happened: several paths in
      # cli_deploy print a message and return normally without creating
      # anything. Absence of a resource name is the reliable tell.
      fields["status"] = "failed"
      fields["message"] = (
          "adk deploy exited cleanly but reported no Agent Runtime resource"
          " name, so nothing was deployed. Check the log above."
      )
      return fields
    if resource_name:
      fields["consoleUrl"] = _agent_runtime_console_url(resource_name)
    return fields


class _ContainerDeployRequest(_BaseDeployRequest):
  """Shared by the two targets that run the agent as a container."""

  service_name: str
  port: int = 8000
  """Container port. Mirrors the CLI's own `--port` default."""

  with_ui: bool = False
  """Ship the dev UI alongside the API server.

  The image has `dev_server.py` removed, so the shipped UI can chat but its
  Builder, Tests and Trace tabs call `/dev/*` routes that will not exist.
  """

  @field_validator("service_name")
  @classmethod
  def _check_service_name(cls, value: str) -> str:
    value = value.strip()
    _reject_unsafe("service_name", value)
    if len(value) > _MAX_SERVICE_NAME_CHARS or not _RESOURCE_ID_RE.fullmatch(
        value
    ):
      raise ValueError(
          "service_name must be lower-case letters, digits and hyphens,"
          " start with a letter, end alphanumerically, and be at most"
          f" {_MAX_SERVICE_NAME_CHARS} characters"
      )
    return value

  @field_validator("port")
  @classmethod
  def _check_port(cls, value: int) -> int:
    if not 1 <= value <= 65535:
      raise ValueError("port must be between 1 and 65535")
    return value

  def container_flags(self) -> list[str]:
    flags = ["--service_name", self.service_name, "--port", str(self.port)]
    if self.with_ui:
      flags.append("--with_ui")
    return flags


class CloudRunDeployRequest(_ContainerDeployRequest):
  """Body of `POST /dev/apps/{app_name}/deploy/cloud_run`."""

  target: ClassVar[DeployTarget] = "cloud_run"

  allow_unauthenticated: bool = False
  """Whether the service accepts calls without credentials.

  Always sent to gcloud, never left to default. For a service that does not
  exist yet, `gcloud run deploy` asks "Allow unauthenticated invocations?" on
  the terminal; the child has no stdin, so it would read EOF and quietly pick
  "no", deciding the service's exposure by accident rather than by choice.
  """

  def target_flags(self) -> list[str]:
    # Reaches gcloud via CloudRunDeployer, which appends provider_args to the
    # `gcloud run deploy` command verbatim.
    exposure = (
        "--allow-unauthenticated"
        if self.allow_unauthenticated
        else "--no-allow-unauthenticated"
    )
    return self.container_flags() + [f"--provider-args={exposure}"]

  def result_fields(self, *, log: str, exit_code: int) -> dict[str, Any]:
    urls = _SERVICE_URL_RE.findall(log)
    service_url = urls[-1] if urls else None
    # `project` is empty whenever the user let the CLI resolve it from gcloud
    # config, in which case the log is the only record of which project was
    # actually used.
    scraped = _LOG_PROJECT_RE.search(log)
    project = self.project or (scraped.group(1) if scraped else None)
    fields: dict[str, Any] = {
        "resourceName": self.service_name,
        "serviceUrl": service_url,
        "consoleUrl": _cloud_run_console_url(
            project=project,
            region=self.region,
            service_name=self.service_name,
        ),
    }
    if exit_code == 0 and not service_url:
      # gcloud exits non-zero when a deploy fails, so a clean exit is
      # trustworthy here even if the URL could not be scraped from the log.
      fields["message"] = (
          "Deployed, but no service URL appeared in the output. Find it with:"
          f" gcloud run services describe {self.service_name}"
          f" --region {self.region}"
      )
    return fields


class GkeDeployRequest(_ContainerDeployRequest):
  """Body of `POST /dev/apps/{app_name}/deploy/gke`."""

  target: ClassVar[DeployTarget] = "gke"

  cluster_name: str
  service_type: Literal["ClusterIP", "LoadBalancer"] = "ClusterIP"

  @field_validator("cluster_name")
  @classmethod
  def _check_cluster_name(cls, value: str) -> str:
    value = value.strip()
    _reject_unsafe("cluster_name", value)
    if len(value) > _MAX_CLUSTER_NAME_CHARS or not _RESOURCE_ID_RE.fullmatch(
        value
    ):
      raise ValueError(
          "cluster_name must be lower-case letters, digits and hyphens,"
          " start with a letter, end alphanumerically, and be at most"
          f" {_MAX_CLUSTER_NAME_CHARS} characters"
      )
    return value

  def target_flags(self) -> list[str]:
    return self.container_flags() + [
        "--cluster_name",
        self.cluster_name,
        "--service_type",
        self.service_type,
    ]

  def result_fields(self, *, log: str, exit_code: int) -> dict[str, Any]:
    fields: dict[str, Any] = {"resourceName": self.service_name}
    if exit_code != 0:
      return fields
    if _DEPLOY_FAILED_RE.search(log):
      # Belt and braces. `cli_deploy_gke` now exits non-zero on failure, but
      # GKE creates nothing this side can look up, so a clean exit is the only
      # other evidence of success -- and it was wrong until recently. Trusting
      # it alone would report a failed deploy as succeeded.
      fields["status"] = "failed"
      fields["message"] = (
          "adk deploy reported a failure but exited cleanly. Check the log"
          " above."
      )
      return fields
    if self.service_type == "LoadBalancer":
      # The external IP is assigned asynchronously, so there is nothing to
      # report yet however long we waited.
      fields["message"] = (
          "Deployed. The load balancer IP is assigned asynchronously; check"
          f" it with: kubectl get svc {self.service_name}"
      )
    else:
      # 8080 is just a free local port; the right-hand side is the Service's.
      fields["message"] = (
          "Deployed. A ClusterIP service is only reachable inside the"
          " cluster; reach it with: kubectl port-forward svc/"
          f"{self.service_name} 8080:{_GKE_SERVICE_PORT}"
      )
    return fields


class DeployDefaults(common.BaseModel):
  """Prefill for the deploy form, resolved from what the CLI itself reads.

  One payload covers all three targets so switching target in the dialog does
  not need another round trip. Serialized with camelCase aliases, like every
  other response the UI consumes.
  """

  project: Optional[str] = None
  project_source: Optional[str] = None
  """Where `project` came from.

  One of "dotenv" (the agent's .env), "environment" (a variable exported to
  the server), "gcloud" (gcloud config), or None when nothing supplied it.
  """

  region: Optional[str] = None
  region_source: Optional[str] = None
  """Where `region` came from: "dotenv", "environment", or None.

  There is no gcloud fallback: gcloud config has no region default that
  applies here.
  """

  display_name: Optional[str] = None
  description: Optional[str] = None
  service_name: Optional[str] = None
  """The agent name in Cloud Run / GKE form: lower-case, hyphens for spaces."""

  env_file: Optional[str] = None
  """Path of the `.env` that supplied values, shown so the source is visible."""


def _find_dotenv(agent_dir: str) -> Optional[str]:
  """Finds the `.env` that applies to an agent.

  Prefers the agent's own folder, which is the only place `adk deploy` looks
  (`cli_deploy.to_agent_engine`), then walks up the way
  `envs.load_dotenv_for_agent` does at runtime, so a `.env` shared by a whole
  agents directory is still found.

  Note the two disagree about more than lookup: values are forwarded to the
  deployed Agent Runtime only from the agent folder's own `.env`. A parent
  `.env` can therefore prefill this form without its variables reaching the
  deployed agent, which is why the resolved path is returned for display.
  """
  folder = os.path.abspath(agent_dir)
  while True:
    candidate = os.path.join(folder, ".env")
    if os.path.isfile(candidate):
      return candidate
    parent = os.path.dirname(folder)
    if parent == folder:
      return None
    folder = parent


def _sanitize_service_name(name: str) -> str:
  """Turns an agent folder name into a legal Cloud Run / GKE service name.

  Agent folders are Python identifiers, so they routinely contain underscores
  and capitals that neither Cloud Run nor Kubernetes accepts.
  """
  slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
  slug = re.sub(r"-{2,}", "-", slug)
  if not slug or not slug[0].isalpha():
    slug = f"adk-{slug}" if slug else "adk-agent"
  return slug[:_MAX_SERVICE_NAME_CHARS].rstrip("-")


async def _gcloud_project() -> Optional[str]:
  """Returns the active gcloud project, or None if it cannot be determined.

  Mirrors `cli_deploy._resolve_project` but never raises and never prompts:
  an unset project just means one less prefilled field.
  """
  try:
    process = await asyncio.create_subprocess_exec(
        _GCLOUD_CMD,
        "config",
        "get-value",
        "project",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
  except (OSError, ValueError):
    logger.info("gcloud is not available; skipping project prefill")
    return None

  try:
    stdout, _ = await asyncio.wait_for(
        process.communicate(), timeout=_GCLOUD_LOOKUP_TIMEOUT_SECONDS
    )
  except asyncio.TimeoutError:
    logger.warning("gcloud config lookup timed out; skipping project prefill")
    process.kill()
    return None

  if process.returncode != 0:
    return None
  project = stdout.decode("utf-8", errors="replace").strip()
  # gcloud prints this literal when no project is configured.
  if not project or project == "(unset)":
    return None
  return project


def _lookup(
    key: str, env_values: dict[str, Any]
) -> tuple[Optional[str], Optional[str]]:
  """Resolves one variable, returning its value and where it came from.

  The agent's own `.env` wins over an exported environment variable: it sits
  next to the agent, it travels with it, and it is the file `adk deploy` reads
  when it decides what to ship to the deployed agent.
  """
  value = (env_values.get(key) or "").strip()
  if value:
    return value, "dotenv"
  value = (_STARTUP_ENVIRON.get(key) or "").strip()
  if value:
    return value, "environment"
  return None, None


async def _resolve_defaults(*, app_name: str, agent_dir: str) -> DeployDefaults:
  """Resolves what the deploy form should be prefilled with."""
  folder_name = os.path.basename(os.path.normpath(agent_dir)) or app_name
  defaults = DeployDefaults(
      display_name=folder_name,
      service_name=_sanitize_service_name(folder_name),
  )

  env_file = _find_dotenv(agent_dir)
  env_values: dict[str, Any] = {}
  if env_file:
    defaults.env_file = env_file
    try:
      from dotenv import dotenv_values

      # dotenv_values parses without touching os.environ. Reading the process
      # environment instead would be wrong: the server loads each agent's
      # .env into its own environment with override=True as agents are
      # loaded, so os.environ reflects whichever agent was loaded last, not
      # the one being deployed.
      env_values = {
          k: v for k, v in dotenv_values(env_file).items() if v is not None
      }
    except Exception:  # pylint: disable=broad-except
      logger.exception("Could not read %s for deploy defaults", env_file)

  defaults.region, defaults.region_source = _lookup(_REGION_ENV_VAR, env_values)

  defaults.project, defaults.project_source = _lookup(
      _PROJECT_ENV_VAR, env_values
  )
  if not defaults.project:
    # Project alone has a third source. Region has no equivalent, so it stays
    # empty and the user picks one.
    project = await _gcloud_project()
    if project:
      defaults.project = project
      defaults.project_source = "gcloud"

  config_path = os.path.join(agent_dir, ".agent_engine_config.json")
  if os.path.isfile(config_path):
    try:
      with open(config_path, "r", encoding="utf-8") as f:
        agent_config = json.load(f)
      if isinstance(agent_config, dict):
        defaults.display_name = (
            agent_config.get("display_name") or defaults.display_name
        )
        defaults.description = agent_config.get("description")
    except (OSError, ValueError):
      logger.exception("Could not read %s for deploy defaults", config_path)

  return defaults


def _build_deploy_argv(
    *,
    agent_dir: str,
    temp_folder: str,
    request: _BaseDeployRequest,
) -> list[str]:
  """Builds the child argv for `adk deploy <target>`.

  Invoked through `sys.executable -m google.adk.cli` rather than a bare `adk`
  so the child always runs in the same interpreter and virtualenv as the
  server, whether or not `adk` is on PATH.
  """
  return [
      sys.executable,
      "-m",
      "google.adk.cli",
      "deploy",
      request.target,
      *request.common_flags(),
      *request.target_flags(),
      # cli_deploy defaults the Agent Runtime temp folder to a timestamped
      # directory beside the agent, i.e. inside agents_dir, where
      # `--reload_agents` watches recursively and `/list-apps` would enumerate
      # it. An absolute path wins the os.path.join and keeps the staging copy
      # out of the way for every target. The CLI creates and removes it; do
      # not pre-create it here.
      "--temp_folder",
      temp_folder,
      agent_dir,
  ]


def _agent_runtime_console_url(resource_name: str) -> Optional[str]:
  """Builds the Cloud console URL for a deployed Agent Runtime."""
  parts = resource_name.split("/")
  if len(parts) < 6 or parts[0] != "projects" or parts[2] != "locations":
    return None
  return (
      "https://console.cloud.google.com/vertex-ai/agents/agent-engines"
      f"/locations/{parts[3]}/agent-engines/{parts[5]}/playground"
      f"?project={parts[1]}"
  )


def _cloud_run_console_url(
    *, project: Optional[str], region: str, service_name: str
) -> Optional[str]:
  """Builds the Cloud console URL for a deployed Cloud Run service."""
  if not project:
    return None
  return (
      f"https://console.cloud.google.com/run/detail/{region}/{service_name}"
      f"/metrics?project={project}"
  )


def _read_log(log_path: str) -> str:
  """Returns the deploy log with ANSI colour codes stripped.

  gcloud wraps values in bold escape sequences. The deploy line reads
  `service [ESC[1msvcESC[m] in project [ESC[1mprojESC[m]`,
  so anything scraped out of the raw text would carry them along.
  """
  try:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
      return _ANSI_RE.sub("", f.read())
  except OSError:
    logger.exception("Unable to read deploy log at %s", log_path)
    return ""


def _base_result(
    *, target: DeployTarget, exit_code: Optional[int], log_path: str
) -> dict[str, Any]:
  """Returns the keys every deploy result carries, whatever the outcome.

  One definition so the success and failure paths cannot drift into reporting
  different shapes to the client.
  """
  return {
      "target": target,
      "exitCode": exit_code,
      "logPath": log_path,
      "resourceName": None,
      # Where to send someone who clicks through: always the Cloud console.
      "consoleUrl": None,
      # The live endpoint, for targets that have one worth calling directly.
      "serviceUrl": None,
  }


def _summarize(
    *, exit_code: int, log_path: str, request: _BaseDeployRequest
) -> dict[str, Any]:
  """Derives the structured result from the child's exit code and output.

  Identifiers are recovered by re-reading the log rather than by matching the
  streamed chunks, so a value straddling a chunk boundary is still found.
  """
  result = _base_result(
      target=request.target, exit_code=exit_code, log_path=log_path
  )
  result.update(
      request.result_fields(log=_read_log(log_path), exit_code=exit_code)
  )

  if exit_code != 0:
    result["status"] = "failed"
    result["message"] = f"adk deploy exited with status {exit_code}."
  else:
    # A target may already have failed on its own evidence; keep that.
    result.setdefault("status", "succeeded")
  return result


def _result_line(payload: dict[str, Any]) -> bytes:
  return f"\n{_RESULT_MARKER} {json.dumps(payload)}\n".encode("utf-8")


async def _spawn_deploy(
    *,
    app_name: str,
    agent_dir: str,
    request: _BaseDeployRequest,
) -> tuple[asyncio.subprocess.Process, str, "asyncio.Task[int]"]:
  """Starts the deploy child and returns it, its log path, and its waiter.

  Spawning happens while the request is being handled rather than on the first
  read of the response body. An async generator does nothing until it is
  pulled, so spawning inside one would make the deploy start only once the
  client began reading, and a client that connected and never read would
  silently deploy nothing.

  The child writes to a log file rather than to a pipe this server drains,
  because the child deliberately outlives the request (see `_tail_deploy`) and
  would block forever once an undrained pipe buffer filled. A file has no such
  limit, and it leaves the full log behind for debugging.
  """
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  base = os.path.join(tempfile.gettempdir(), "adk_deploy")
  staging_dir = os.path.join(base, f"{app_name}_{request.target}_{stamp}")
  log_path = os.path.join(base, f"{app_name}_{request.target}_{stamp}.log")
  os.makedirs(base, exist_ok=True)

  argv = _build_deploy_argv(
      agent_dir=agent_dir,
      temp_folder=staging_dir,
      request=request,
  )
  env = os.environ.copy()
  # click.echo into a file descriptor is block-buffered; without this the UI
  # shows nothing for minutes and then the whole log at once.
  env["PYTHONUNBUFFERED"] = "1"

  with open(log_path, "wb") as log_writer:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=log_writer,
        stderr=subprocess.STDOUT,
        # gcloud and kubectl prompt on a terminal; a child with no stdin gets
        # an EOF and fails fast instead of hanging the deploy forever.
        stdin=subprocess.DEVNULL,
        env=env,
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if _IS_WINDOWS
            else 0
        ),
        # Detach from the server's process group so Ctrl-C on the `adk web`
        # terminal does not abort a deploy that is midway through.
        start_new_session=not _IS_WINDOWS,
    )
  logger.info(
      "Deploying %s to %s as pid %d, log: %s",
      app_name,
      request.target,
      process.pid,
      log_path,
  )

  # Hold the app's slot until the child exits, however the request ends.
  waiter = asyncio.ensure_future(process.wait())
  _deploys_in_flight[app_name] = waiter
  waiter.add_done_callback(lambda _: _deploys_in_flight.pop(app_name, None))
  return process, log_path, waiter


async def _tail_deploy(
    *,
    app_name: str,
    agent_dir: str,
    process: asyncio.subprocess.Process,
    log_path: str,
    waiter: "asyncio.Task[int]",
    request: _BaseDeployRequest,
) -> AsyncIterator[bytes]:
  """Yields the running deploy's output, then one result line.

  The deploy is deliberately *not* killed when the client disconnects. For
  Agent Runtime, `cli_deploy.to_agent_engine` creates the instance before it
  builds the image and deletes it again only if the subsequent update raises,
  so killing the child in between would strand an empty instance. Cloud Run
  and GKE are less fragile but no happier being interrupted midway through a
  Cloud Build. A closed browser tab loses the log, not the deploy.
  """
  reader = None
  try:
    yield f"$ adk deploy {request.target} {agent_dir}\n".encode("utf-8")

    reader = open(log_path, "rb")
    while True:
      chunk = reader.read(_OUTPUT_CHUNK_BYTES)
      if chunk:
        yield chunk
        continue
      if waiter.done():
        while chunk := reader.read(_OUTPUT_CHUNK_BYTES):
          yield chunk
        break
      await asyncio.sleep(_OUTPUT_POLL_SECONDS)

    yield _result_line(
        _summarize(exit_code=await waiter, log_path=log_path, request=request)
    )
  except Exception as e:  # pylint: disable=broad-except
    logger.exception("Deploy of %s failed while streaming", app_name)
    yield _result_line({
        **_base_result(
            target=request.target, exit_code=None, log_path=log_path
        ),
        "status": "failed",
        "message": f"Could not follow the deploy: {e}",
    })
  finally:
    # No await here: on client disconnect this runs during the generator's
    # aclose(), and the child is intentionally left running (see docstring).
    # The app's slot in _deploys_in_flight is released by the waiter's done
    # callback when the child exits, not here, so a disconnect cannot free the
    # slot for a duplicate deploy while the first is still going.
    if reader is not None:
      reader.close()
    if process.returncode is None:
      logger.info(
          "Client stopped reading; deploy of %s continues as pid %d, log: %s",
          app_name,
          process.pid,
          log_path,
      )


def register_dev_deploy_endpoints(
    app: FastAPI,
    *,
    get_agent_dir: Callable[[str], str],
) -> None:
  """Registers the dev-only deploy endpoints on `app`.

  Args:
    app: The FastAPI app to register on.
    get_agent_dir: Resolves an app name to its agent folder, raising for names
      that escape the agents directory. `DevServer._get_agent_dir` passed in
      rather than the server itself, to keep this module free of a circular
      import.
  """

  def _resolve_agent_dir(app_name: str) -> str:
    agent_dir = get_agent_dir(app_name)
    if not os.path.isdir(agent_dir):
      raise HTTPException(
          status_code=404, detail=f"Agent not found: {app_name}"
      )
    return agent_dir

  async def _start_deploy(
      app_name: str,
      request: _BaseDeployRequest,
  ) -> StreamingResponse:
    """Claims the app's deploy slot, starts the child, and streams it."""
    agent_dir = _resolve_agent_dir(app_name)
    if app_name in _deploys_in_flight:
      raise HTTPException(
          status_code=409,
          detail=f"A deploy of {app_name} is already running.",
      )
    # Claim the slot before the first await, so two requests arriving together
    # cannot both pass the check above. _spawn_deploy replaces the placeholder
    # with the real waiter, which owns the slot from then on.
    _deploys_in_flight[app_name] = None

    try:
      process, log_path, waiter = await _spawn_deploy(
          app_name=app_name, agent_dir=agent_dir, request=request
      )
    except Exception as e:  # pylint: disable=broad-except
      _deploys_in_flight.pop(app_name, None)
      logger.exception("Could not start a deploy of %s", app_name)
      raise HTTPException(
          status_code=500, detail=f"Could not start adk deploy: {e}"
      ) from e

    return StreamingResponse(
        _tail_deploy(
            app_name=app_name,
            agent_dir=agent_dir,
            process=process,
            log_path=log_path,
            waiter=waiter,
            request=request,
        ),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-store",
            # Stops intermediaries from buffering the stream into one blob.
            "X-Accel-Buffering": "no",
        },
    )

  @app.get("/dev/apps/{app_name}/deploy/defaults", tags=[TAG_DEPLOY])
  async def get_deploy_defaults(app_name: str) -> DeployDefaults:
    """Returns what the deploy form should be prefilled with, for any target.

    The browser cannot read the agent's `.env`, its
    `.agent_engine_config.json`, or the local gcloud config, so it asks the
    server. Every value here is only a suggestion: the client sends back
    whatever the user confirms, and the deploy passes those as explicit flags.

    Making the flags explicit also settles a precedence wrinkle. Left to
    itself, `cli_deploy` resolves the project from gcloud config before it
    reads `.env`, so a `GOOGLE_CLOUD_PROJECT` there never takes effect. This
    prefills `.env` first, which is the precedence a reader of that file
    expects, and sending it explicitly makes the deploy honour it.
    """
    agent_dir = _resolve_agent_dir(app_name)
    return await _resolve_defaults(app_name=app_name, agent_dir=agent_dir)

  @app.post(
      "/dev/apps/{app_name}/deploy/agent_engine",
      tags=[TAG_DEPLOY],
      response_class=StreamingResponse,
  )
  async def deploy_to_agent_engine(
      app_name: str,
      request: AgentEngineDeployRequest,
  ) -> StreamingResponse:
    """Deploys an agent to Agent Runtime, streaming the deploy's output.

    The body is plain text: the deploy's own console output, then a final line
    of the form `__ADK_DEPLOY_RESULT__ {json}` carrying the status and, on
    success, the Agent Runtime resource name and console URL. Clients should
    treat that line, not the HTTP status, as the outcome: the response headers
    are sent before the deploy starts, so the status is always 200 once
    validation has passed.
    """
    return await _start_deploy(app_name, request)

  @app.post(
      "/dev/apps/{app_name}/deploy/cloud_run",
      tags=[TAG_DEPLOY],
      response_class=StreamingResponse,
  )
  async def deploy_to_cloud_run(
      app_name: str,
      request: CloudRunDeployRequest,
  ) -> StreamingResponse:
    """Deploys an agent to Cloud Run, streaming the deploy's output.

    Same body shape as the Agent Runtime route; the result line carries the
    service URL scraped from `gcloud run deploy` output.
    """
    return await _start_deploy(app_name, request)

  @app.post(
      "/dev/apps/{app_name}/deploy/gke",
      tags=[TAG_DEPLOY],
      response_class=StreamingResponse,
  )
  async def deploy_to_gke(
      app_name: str,
      request: GkeDeployRequest,
  ) -> StreamingResponse:
    """Deploys an agent to GKE, streaming the deploy's output.

    Same body shape as the other routes. There is no URL to report: a
    ClusterIP service is cluster-internal, and a LoadBalancer's external IP is
    assigned after the deploy returns, so the result line carries the command
    to reach or inspect the service instead.
    """
    return await _start_deploy(app_name, request)

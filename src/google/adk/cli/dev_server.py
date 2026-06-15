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

"""Development server with all ADK endpoints.

This module provides the DevServer class which extends ApiServer with development-only endpoints.
All production endpoints are inherited from ApiServer.
All dev-only endpoints (eval, debug, graph, test management) are added by DevServer.

Use this for local development with `adk web`.
For production deployments, use api_server.py instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from fastapi.responses import StreamingResponse
import graphviz
from pydantic import Field
from pydantic import ValidationError
from typing_extensions import deprecated
import yaml

from . import agent_graph
from ..errors.not_found_error import NotFoundError
from ..evaluation.base_eval_service import InferenceConfig
from ..evaluation.base_eval_service import InferenceRequest
from ..evaluation.eval_case import EvalCase
from ..evaluation.eval_case import SessionInput
from ..evaluation.eval_metrics import EvalMetric
from ..evaluation.eval_metrics import EvalMetricResult
from ..evaluation.eval_metrics import EvalMetricResultPerInvocation
from ..evaluation.eval_metrics import EvalStatus
from ..evaluation.eval_metrics import MetricInfo
from ..evaluation.eval_result import EvalSetResult
from ..evaluation.eval_set import EvalSet
from .api_server import ApiServer
from .utils import common
from .utils import evals
from .utils.graph_serialization import serialize_app_info
from .utils.graph_visualization import plot_workflow_graph
from .utils.state import create_empty_state

logger = logging.getLogger("google_adk." + __name__)

_EVAL_SET_FILE_EXTENSION = ".evalset.json"

TAG_DEBUG = "Debug"
TAG_EVALUATION = "Evaluation"


class CreateTestRequest(common.BaseModel):
  session_data: dict


class AddSessionToEvalSetRequest(common.BaseModel):
  eval_id: str
  session_id: str
  user_id: str


class RunEvalRequest(common.BaseModel):
  eval_ids: list[str] = Field(
      deprecated=True,
      default_factory=list,
      description="This field is deprecated, use eval_case_ids instead.",
  )
  eval_case_ids: list[str] = Field(
      default_factory=list,
      description=(
          "List of eval case ids to evaluate. if empty, then all eval cases in"
          " the eval set are run."
      ),
  )
  eval_metrics: list[EvalMetric]


class RunEvalResult(common.BaseModel):
  eval_set_file: str
  eval_set_id: str
  eval_id: str
  final_eval_status: EvalStatus
  eval_metric_results: list[tuple[EvalMetric, EvalMetricResult]] = Field(
      deprecated=True,
      default=[],
      description=(
          "This field is deprecated, use overall_eval_metric_results instead."
      ),
  )
  overall_eval_metric_results: list[EvalMetricResult]
  eval_metric_result_per_invocation: list[EvalMetricResultPerInvocation]
  user_id: str
  session_id: str


class RunEvalResponse(common.BaseModel):
  run_eval_results: list[RunEvalResult]


class GetEventGraphResult(common.BaseModel):
  dot_src: str


class CreateEvalSetRequest(common.BaseModel):
  eval_set: EvalSet


class ListEvalSetsResponse(common.BaseModel):
  eval_set_ids: list[str]


class EvalResult(EvalSetResult):
  """This class has no field intentionally.

  The goal here is to just give a new name to the class to align with the API
  endpoint.
  """


class ListEvalResultsResponse(common.BaseModel):
  eval_result_ids: list[str]


class ListMetricsInfoResponse(common.BaseModel):
  metrics_info: list[MetricInfo]


class DevServer(ApiServer):
  """Development server that extends ApiServer with dev-only endpoints.

  Inherits all production endpoints from ApiServer and adds development-specific
  endpoints for evaluation, debugging, and developer UI features.
  """

  def _register_dev_endpoints(
      self,
      app: FastAPI,
      trace_dict: dict,
      memory_exporter: Any,
      web_assets_dir: Optional[str] = None,
  ):
    """Register all development-only endpoints.

    This includes debug, evaluation, and graph visualization endpoints.
    These endpoints should NOT be exposed in production deployments.
    """

    # Import needed for eval endpoints
    from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE

    # ========== BUILDER / YAML EDITOR ENDPOINTS ==========
    agents_base_path = (Path.cwd() / self.agents_dir).resolve()

    def _get_app_root(app_name: str) -> Path:
      if app_name in ("", ".", ".."):
        raise ValueError(f"Invalid app name: {app_name!r}")
      if Path(app_name).name != app_name or "\\" in app_name:
        raise ValueError(f"Invalid app name: {app_name!r}")
      app_root = (agents_base_path / app_name).resolve()
      if not app_root.is_relative_to(agents_base_path):
        raise ValueError(f"Invalid app name: {app_name!r}")
      return app_root

    def _normalize_relative_path(path: str) -> str:
      return path.replace("\\", "/").lstrip("/")

    def _has_parent_reference(path: str) -> bool:
      return any(part == ".." for part in path.split("/"))

    _ALLOWED_EXTENSIONS = frozenset({".yaml", ".yml"})

    # --- YAML content security ---
    _BLOCKED_YAML_KEYS = frozenset({"args"})

    def _check_yaml_for_blocked_keys(content: bytes, filename: str) -> None:
      """Raise if the YAML document contains any blocked keys."""
      try:
        docs = list(yaml.safe_load_all(content))
      except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {filename!r}: {exc}") from exc

      def _walk(node: Any) -> None:
        if isinstance(node, dict):
          for key, value in node.items():
            if key in _BLOCKED_YAML_KEYS:
              raise ValueError(
                  f"Blocked key {key!r} found in {filename!r}. "
                  f"The '{key}' field is not allowed in builder uploads "
                  "because it can execute arbitrary code."
              )
            _walk(value)
        elif isinstance(node, list):
          for item in node:
            _walk(item)

      for doc in docs:
        _walk(doc)

    def _parse_upload_filename(app_name: str, filename: Optional[str]) -> str:
      if not filename:
        raise ValueError("Upload filename is missing.")
      filename = _normalize_relative_path(filename)
      prefix = f"{app_name}/"
      if filename.startswith(prefix):
        rel_path = filename[len(prefix) :]
      else:
        rel_path = filename
      if not rel_path:
        raise ValueError(f"Invalid upload filename: {filename!r}")
      if rel_path.startswith("/"):
        raise ValueError(f"Absolute upload path rejected: {filename!r}")
      if _has_parent_reference(rel_path):
        raise ValueError(f"Path traversal rejected: {filename!r}")
      ext = os.path.splitext(rel_path)[1].lower()
      if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type not allowed: {rel_path!r}"
            f" (allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))})"
        )
      return rel_path

    def _parse_file_path(file_path: str) -> str:
      file_path = _normalize_relative_path(file_path)
      if not file_path:
        raise ValueError("file_path is missing.")
      if file_path.startswith("/"):
        raise ValueError(f"Absolute file_path rejected: {file_path!r}")
      if _has_parent_reference(file_path):
        raise ValueError(f"Path traversal rejected: {file_path!r}")
      ext = os.path.splitext(file_path)[1].lower()
      if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type not allowed: {file_path!r}"
            f" (allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))})"
        )
      return file_path

    def _resolve_under_dir(root_dir: Path, rel_path: str) -> Path:
      file_path = root_dir / rel_path
      resolved_root_dir = root_dir.resolve()
      resolved_file_path = file_path.resolve()
      if not resolved_file_path.is_relative_to(resolved_root_dir):
        raise ValueError(f"Path escapes root_dir: {rel_path!r}")
      return file_path

    def _get_tmp_agent_root(app_root: Path, app_name: str) -> Path:
      tmp_agent_root = app_root / "tmp" / app_name
      resolved_tmp_agent_root = tmp_agent_root.resolve()
      if not resolved_tmp_agent_root.is_relative_to(app_root):
        raise ValueError(f"Invalid tmp path for app: {app_name!r}")
      return tmp_agent_root

    def copy_dir_contents(source_dir: Path, dest_dir: Path) -> None:
      dest_dir.mkdir(parents=True, exist_ok=True)
      for source_path in source_dir.iterdir():
        if source_path.name == "tmp":
          continue

        dest_path = dest_dir / source_path.name
        if source_path.is_dir():
          if dest_path.exists() and dest_path.is_file():
            dest_path.unlink()
          shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
        elif source_path.is_file():
          if dest_path.exists() and dest_path.is_dir():
            shutil.rmtree(dest_path)
          shutil.copy2(source_path, dest_path)

    def cleanup_tmp(app_name: str) -> bool:
      try:
        app_root = _get_app_root(app_name)
      except ValueError as exc:
        logger.exception("Error in cleanup_tmp: %s", exc)
        return False

      try:
        tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
      except ValueError as exc:
        logger.exception("Error in cleanup_tmp: %s", exc)
        return False

      try:
        shutil.rmtree(tmp_agent_root)
      except FileNotFoundError:
        pass
      except OSError as exc:
        logger.exception("Error deleting tmp agent root: %s", exc)
        return False

      tmp_dir = app_root / "tmp"
      resolved_tmp_dir = tmp_dir.resolve()
      if not resolved_tmp_dir.is_relative_to(app_root):
        logger.error(
            "Refusing to delete tmp outside app_root: %s", resolved_tmp_dir
        )
        return False

      try:
        tmp_dir.rmdir()
      except OSError:
        pass

      return True

    def ensure_tmp_exists(app_name: str) -> bool:
      try:
        app_root = _get_app_root(app_name)
      except ValueError as exc:
        logger.exception("Error in ensure_tmp_exists: %s", exc)
        return False

      if not app_root.is_dir():
        return False

      try:
        tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
      except ValueError as exc:
        logger.exception("Error in ensure_tmp_exists: %s", exc)
        return False

      if tmp_agent_root.exists():
        return True

      try:
        tmp_agent_root.mkdir(parents=True, exist_ok=True)
        copy_dir_contents(app_root, tmp_agent_root)
      except OSError as exc:
        logger.exception("Error in ensure_tmp_exists: %s", exc)
        return False

      return True

    @app.post(
        "/dev/apps/{app_name}/builder/save", response_model_exclude_none=True
    )
    async def builder_build(
        app_name: str, files: list[UploadFile], tmp: Optional[bool] = False
    ) -> bool:
      try:
        uploads: list[tuple[str, bytes]] = []
        for file in files:
          rel_path = _parse_upload_filename(app_name, file.filename)
          content = await file.read()
          uploads.append((rel_path, content))

        for rel_path, content in uploads:
          _check_yaml_for_blocked_keys(content, f"{app_name}/{rel_path}")

        if tmp:
          app_root = _get_app_root(app_name)
          tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
          tmp_agent_root.mkdir(parents=True, exist_ok=True)

          for rel_path, content in uploads:
            destination_path = _resolve_under_dir(tmp_agent_root, rel_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(content)

          return True

        app_root = _get_app_root(app_name)
        app_root.mkdir(parents=True, exist_ok=True)

        tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
        if tmp_agent_root.is_dir():
          copy_dir_contents(tmp_agent_root, app_root)

        for rel_path, content in uploads:
          destination_path = _resolve_under_dir(app_root, rel_path)
          destination_path.parent.mkdir(parents=True, exist_ok=True)
          destination_path.write_bytes(content)

        return cleanup_tmp(app_name)
      except ValueError as exc:
        logger.exception("Error in builder_build: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
      except OSError as exc:
        logger.exception("Error in builder_build: %s", exc)
        return False

    @app.post(
        "/dev/apps/{app_name}/builder/cancel", response_model_exclude_none=True
    )
    async def builder_cancel(app_name: str) -> bool:
      return cleanup_tmp(app_name)

    @app.get(
        "/dev/apps/{app_name}/builder",
        response_model_exclude_none=True,
        response_class=PlainTextResponse,
    )
    async def get_agent_builder(
        app_name: str,
        file_path: Optional[str] = None,
        tmp: Optional[bool] = False,
    ):
      try:
        app_root = _get_app_root(app_name)
      except ValueError as exc:
        logger.exception("Error in get_agent_builder: %s", exc)
        return ""

      agent_dir = app_root
      if tmp:
        if not ensure_tmp_exists(app_name):
          return ""
        agent_dir = app_root / "tmp" / app_name

      if not file_path:
        rel_path = "root_agent.yaml"
      else:
        try:
          rel_path = _parse_file_path(file_path)
        except ValueError as exc:
          logger.exception("Error in get_agent_builder: %s", exc)
          return ""

      try:
        agent_file_path = _resolve_under_dir(agent_dir, rel_path)
      except ValueError as exc:
        logger.exception("Error in get_agent_builder: %s", exc)
        return ""

      if not agent_file_path.is_file():
        return ""

      return FileResponse(
          path=agent_file_path,
          media_type="application/x-yaml",
          filename=file_path or f"{app_name}.yaml",
          headers={"Cache-Control": "no-store"},
      )

    # ========== DEBUG & GRAPH ENDPOINTS ==========

    @app.get("/dev/apps/{app_name}/debug/trace/{event_id}", tags=[TAG_DEBUG])
    async def get_trace_dict(app_name: str, event_id: str) -> Any:
      event_dict = trace_dict.get(event_id, None)
      if event_dict is None:
        raise HTTPException(status_code=404, detail="Trace not found")
      return event_dict

    @app.get(
        "/dev/apps/{app_name}/debug/trace/session/{session_id}",
        tags=[TAG_DEBUG],
    )
    async def get_session_trace(app_name: str, session_id: str) -> Any:
      spans = memory_exporter.get_finished_spans(session_id)
      if not spans:
        return []
      return [
          {
              "name": s.name,
              "span_id": s.context.span_id,
              "trace_id": s.context.trace_id,
              "start_time": s.start_time,
              "end_time": s.end_time,
              "attributes": dict(s.attributes),
              "parent_span_id": s.parent.span_id if s.parent else None,
          }
          for s in spans
      ]

    if web_assets_dir:
      # TODO: remove this endpoint once build_graph_image is completed
      @app.get("/dev/apps/{app_name}/build_graph")
      async def get_app_info(app_name: str) -> Any:
        runner = await self.get_runner_async(app_name)

        if not runner.app:
          raise HTTPException(
              status_code=404, detail=f"App not found: {app_name}"
          )

        # Read README.md if it exists
        readme_content = None
        if self.agents_dir:
          import os

          readme_path = os.path.join(self.agents_dir, app_name, "README.md")
          if os.path.exists(readme_path):
            try:
              with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            except Exception as e:
              print(f"Error reading README.md: {e}")

        return serialize_app_info(runner.app, readme_content)

    @app.get("/dev/apps/{app_name}/build_graph_image")
    async def get_app_info_image(
        app_name: str, dark_mode: bool = False, node: Optional[str] = None
    ) -> dict[str, GetEventGraphResult]:
      runner = await self.get_runner_async(app_name)

      if not runner.app:
        raise HTTPException(
            status_code=404, detail=f"App not found: {app_name}"
        )

      app_info = serialize_app_info(runner.app)

      # Navigate to specific level if node is provided
      if node:
        target_agent = self._navigate_to_node(app_info, node)
        if not target_agent:
          raise HTTPException(status_code=404, detail=f"Node not found: {node}")
        # Create a temporary app_info structure for the target level
        app_info = {"root_agent": target_agent}

      workflows = self._get_all_sub_workflows(app_info, node if node else "")

      # This allows plotting non-workflow agents as a tree.
      target_path = node if node else ""
      if target_path not in workflows:
        target_agent = app_info.get("root_agent")
        if target_agent:
          workflows[target_path] = target_agent

      results = {}
      for path, info in workflows.items():
        dot_string = plot_workflow_graph(
            {"root_agent": info}, format="dot", dark_mode=dark_mode
        )
        if dot_string:
          results[path] = GetEventGraphResult(dot_src=dot_string)

      return results

    # ========== AGENT TESTING ENDPOINTS ==========

    @app.get("/dev/apps/{app_name}/tests")
    async def list_tests(app_name: str) -> list[str]:
      """Lists all test JSON files for the given app."""
      agent_dir = os.path.join(self.agents_dir, app_name)
      tests_dir = os.path.join(agent_dir, "tests")
      if not os.path.exists(tests_dir):
        return []

      import glob

      pattern = os.path.join(tests_dir, "*.json")
      test_files = glob.glob(pattern)
      return sorted([os.path.basename(f) for f in test_files])

    @app.post("/dev/apps/{app_name}/tests/rebuild")
    async def rebuild_app_tests(
        app_name: str, test_name: Optional[str] = None
    ) -> dict[str, str]:
      """Rebuilds tests for the app."""
      agent_dir = os.path.join(self.agents_dir, app_name)

      if test_name:
        if not test_name.endswith(".json"):
          test_name += ".json"
        path = os.path.join(agent_dir, "tests", test_name)
      else:
        path = agent_dir

      from .agent_test_runner import rebuild_tests

      await asyncio.to_thread(rebuild_tests, path)
      return {"status": "success"}

    @app.post("/dev/apps/{app_name}/tests/run")
    async def run_app_tests(
        app_name: str, test_name: Optional[str] = None
    ) -> StreamingResponse:
      """Runs tests and streams pytest output."""
      agent_dir = os.path.join(self.agents_dir, app_name)

      import subprocess
      import sys

      queue = asyncio.Queue()

      async def run_pytest_subprocess():
        cmd_args = [
            sys.executable,
            "-m",
            "pytest",
            os.path.join(os.path.dirname(__file__), "agent_test_runner.py"),
            "-s",
            "-vv",
        ]
        if test_name:
          name_to_use = (
              test_name[:-5] if test_name.endswith(".json") else test_name
          )
          cmd_args.extend(["-k", name_to_use])

        # Ensure environment variable is set
        env = os.environ.copy()
        env["ADK_TEST_FOLDER"] = agent_dir

        try:
          process = await asyncio.create_subprocess_exec(
              *cmd_args,
              stdout=subprocess.PIPE,
              stderr=subprocess.STDOUT,
              env=env,
          )

          while True:
            line = await process.stdout.readline()
            if not line:
              break
            await queue.put(line.decode("utf-8"))

          await process.wait()
        finally:
          # Signal completion to generator
          await queue.put(None)

      # Start pytest in a background task
      asyncio.create_task(run_pytest_subprocess())

      async def generate():
        while True:
          item = await queue.get()
          if item is None:
            break
          yield item.encode("utf-8")

      return StreamingResponse(generate(), media_type="text/plain")

    @app.put("/dev/apps/{app_name}/tests/{test_name}")
    async def create_test(
        app_name: str, test_name: str, req: CreateTestRequest
    ) -> dict[str, str]:
      """Creates or updates a test file from session data."""
      # Sanitize test_name to prevent directory traversal
      test_name = os.path.basename(test_name)
      agent_dir = os.path.join(self.agents_dir, app_name)
      tests_dir = os.path.join(agent_dir, "tests")
      os.makedirs(tests_dir, exist_ok=True)

      if not test_name.endswith(".json"):
        test_name += ".json"

      test_file_path = os.path.join(tests_dir, test_name)

      with open(test_file_path, "w") as f:
        json.dump(req.session_data, f, indent=2, sort_keys=True)

      return {"status": "success", "file": test_name}

    @app.delete("/dev/apps/{app_name}/tests/{test_name}")
    async def delete_test(app_name: str, test_name: str) -> dict[str, str]:
      """Deletes a specific test file."""
      agent_dir = os.path.join(self.agents_dir, app_name)
      tests_dir = os.path.join(agent_dir, "tests")

      if not test_name.endswith(".json"):
        test_name += ".json"

      test_file_path = os.path.join(tests_dir, test_name)

      if not os.path.exists(test_file_path):
        raise HTTPException(status_code=404, detail="Test file not found")

      os.remove(test_file_path)
      return {"status": "success"}

    @app.get("/dev/apps/{app_name}/tests/{test_name}")
    async def get_test_content(app_name: str, test_name: str) -> dict[str, Any]:
      """Fetches the content of a specific test file."""
      agent_dir = os.path.join(self.agents_dir, app_name)
      tests_dir = os.path.join(agent_dir, "tests")

      if not test_name.endswith(".json"):
        test_name += ".json"

      test_file_path = os.path.join(tests_dir, test_name)

      if not os.path.exists(test_file_path):
        raise HTTPException(status_code=404, detail="Test file not found")

      with open(test_file_path, "r") as f:
        return json.load(f)

    # ========== EVALUATION ENDPOINTS ==========

    @app.post(
        "/dev/apps/{app_name}/eval-sets",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def create_eval_set(
        app_name: str, create_eval_set_request: CreateEvalSetRequest
    ) -> EvalSet:
      try:
        return self.eval_sets_manager.create_eval_set(
            app_name=app_name,
            eval_set_id=create_eval_set_request.eval_set.eval_set_id,
        )
      except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve),
        ) from ve

    # TODO - remove after migration
    @deprecated(
        "Please use create_eval_set instead. This will be removed in future"
        " releases."
    )
    @app.post(
        "/dev/apps/{app_name}/eval_sets/{eval_set_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def create_eval_set_legacy(
        app_name: str,
        eval_set_id: str,
    ):
      """Creates an eval set, given the id."""
      await create_eval_set(
          app_name=app_name,
          create_eval_set_request=CreateEvalSetRequest(
              eval_set=UserEvalSet(eval_set_id=eval_set_id, eval_cases=[]),
          ),
      )

    # TODO - remove after migration
    @deprecated(
        "Please use list_eval_sets instead. This will be removed in future"
        " releases."
    )
    @app.get(
        "/dev/apps/{app_name}/eval_sets",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_sets_legacy(app_name: str) -> list[str]:
      list_eval_sets_response = await list_eval_sets(app_name)
      return list_eval_sets_response.eval_set_ids

    # TODO - remove after migration
    @deprecated(
        "Please use run_eval instead. This will be removed in future releases."
    )
    @app.post(
        "/dev/apps/{app_name}/eval_sets/{eval_set_id}/run_eval",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def run_eval_legacy(
        app_name: str, eval_set_id: str, req: RunEvalRequest
    ) -> list[RunEvalResult]:
      run_eval_response = await run_eval(
          app_name=app_name, eval_set_id=eval_set_id, req=req
      )
      return run_eval_response.run_eval_results

    # TODO - remove after migration
    @deprecated(
        "Please use get_eval_result instead. This will be removed in future"
        " releases."
    )
    @app.get(
        "/dev/apps/{app_name}/eval_results/{eval_result_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def get_eval_result_legacy(
        app_name: str,
        eval_result_id: str,
    ) -> EvalSetResult:
      try:
        return self.eval_set_results_manager.get_eval_set_result(
            app_name, eval_result_id
        )
      except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve
      except ValidationError as ve:
        raise HTTPException(status_code=500, detail=str(ve)) from ve

    # TODO - remove after migration
    @deprecated(
        "Please use list_eval_results instead. This will be removed in future"
        " releases."
    )
    @app.get(
        "/dev/apps/{app_name}/eval_results",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_results_legacy(app_name: str) -> list[str]:
      list_eval_results_response = await list_eval_results(app_name)
      return list_eval_results_response.eval_result_ids

    @app.get(
        "/dev/apps/{app_name}/eval-sets",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_sets(app_name: str) -> ListEvalSetsResponse:
      """Lists all eval sets for the given app."""
      eval_sets = []
      try:
        eval_sets = self.eval_sets_manager.list_eval_sets(app_name)
      except NotFoundError as e:
        logger.warning(e)

      return ListEvalSetsResponse(eval_set_ids=eval_sets)

    @app.post(
        "/dev/apps/{app_name}/eval-sets/{eval_set_id}/add-session",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    @app.post(
        "/dev/apps/{app_name}/eval_sets/{eval_set_id}/add_session",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def add_session_to_eval_set(
        app_name: str, eval_set_id: str, req: AddSessionToEvalSetRequest
    ):
      # Get the session
      session = await self.session_service.get_session(
          app_name=app_name, user_id=req.user_id, session_id=req.session_id
      )
      assert session, "Session not found."

      # Convert the session data to eval invocations
      invocations = evals.convert_session_to_eval_invocations(session)

      # Populate the session with initial session state.
      agent_or_app = self.agent_loader.load_agent(app_name)
      root_agent = self._get_root_agent(agent_or_app)
      initial_session_state = create_empty_state(root_agent)

      new_eval_case = EvalCase(
          eval_id=req.eval_id,
          conversation=invocations,
          session_input=SessionInput(
              app_name=app_name,
              user_id=req.user_id,
              state=initial_session_state,
          ),
          creation_timestamp=time.time(),
      )

      try:
        self.eval_sets_manager.add_eval_case(
            app_name, eval_set_id, new_eval_case
        )
      except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve

    @app.get(
        "/dev/apps/{app_name}/eval_sets/{eval_set_id}/evals",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_evals_in_eval_set(
        app_name: str,
        eval_set_id: str,
    ) -> list[str]:
      """Lists all evals in an eval set."""
      eval_set_data = self.eval_sets_manager.get_eval_set(app_name, eval_set_id)

      if not eval_set_data:
        raise HTTPException(
            status_code=400, detail=f"Eval set `{eval_set_id}` not found."
        )

      return sorted([x.eval_id for x in eval_set_data.eval_cases])

    @app.get(
        "/dev/apps/{app_name}/eval-sets/{eval_set_id}/eval-cases/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    @app.get(
        "/dev/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def get_eval(
        app_name: str, eval_set_id: str, eval_case_id: str
    ) -> EvalCase:
      """Gets an eval case in an eval set."""
      eval_case_to_find = self.eval_sets_manager.get_eval_case(
          app_name, eval_set_id, eval_case_id
      )

      if eval_case_to_find:
        return eval_case_to_find

      raise HTTPException(
          status_code=404,
          detail=(
              f"Eval set `{eval_set_id}` or Eval `{eval_case_id}` not found."
          ),
      )

    @app.put(
        "/dev/apps/{app_name}/eval-sets/{eval_set_id}/eval-cases/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    @app.put(
        "/dev/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def update_eval(
        app_name: str,
        eval_set_id: str,
        eval_case_id: str,
        updated_eval_case: EvalCase,
    ):
      if (
          updated_eval_case.eval_id
          and updated_eval_case.eval_id != eval_case_id
      ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Eval id in EvalCase should match the eval id in the API route."
            ),
        )

      # Overwrite the value. We are either overwriting the same value or an empty
      # field.
      updated_eval_case.eval_id = eval_case_id
      try:
        self.eval_sets_manager.update_eval_case(
            app_name, eval_set_id, updated_eval_case
        )
      except NotFoundError as nfe:
        raise HTTPException(status_code=404, detail=str(nfe)) from nfe

    @app.delete(
        "/dev/apps/{app_name}/eval-sets/{eval_set_id}/eval-cases/{eval_case_id}",
        tags=[TAG_EVALUATION],
    )
    @app.delete(
        "/dev/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        tags=[TAG_EVALUATION],
    )
    async def delete_eval(
        app_name: str, eval_set_id: str, eval_case_id: str
    ) -> None:
      try:
        self.eval_sets_manager.delete_eval_case(
            app_name, eval_set_id, eval_case_id
        )
      except NotFoundError as nfe:
        raise HTTPException(status_code=404, detail=str(nfe)) from nfe

    @app.post(
        "/dev/apps/{app_name}/eval-sets/{eval_set_id}/run",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def run_eval(
        app_name: str, eval_set_id: str, req: RunEvalRequest
    ) -> RunEvalResponse:
      """Runs an eval given the details in the eval request."""
      # Create a mapping from eval set file to all the evals that needed to be
      # run.
      try:
        from ..evaluation.local_eval_service import LocalEvalService
        from .cli_eval import _collect_eval_results
        from .cli_eval import _collect_inferences

        eval_set = self.eval_sets_manager.get_eval_set(app_name, eval_set_id)

        if not eval_set:
          raise HTTPException(
              status_code=400, detail=f"Eval set `{eval_set_id}` not found."
          )

        agent_or_app = self.agent_loader.load_agent(app_name)
        root_agent = self._get_root_agent(agent_or_app)

        eval_case_results = []

        eval_service = LocalEvalService(
            root_agent=root_agent,
            eval_sets_manager=self.eval_sets_manager,
            eval_set_results_manager=self.eval_set_results_manager,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
        )
        inference_request = InferenceRequest(
            app_name=app_name,
            eval_set_id=eval_set.eval_set_id,
            eval_case_ids=req.eval_case_ids or req.eval_ids,
            inference_config=InferenceConfig(),
        )
        inference_results = await _collect_inferences(
            inference_requests=[inference_request], eval_service=eval_service
        )

        eval_case_results = await _collect_eval_results(
            inference_results=inference_results,
            eval_service=eval_service,
            eval_metrics=req.eval_metrics,
        )
      except ModuleNotFoundError as e:
        logger.exception("%s", e)
        raise HTTPException(
            status_code=400, detail=MISSING_EVAL_DEPENDENCIES_MESSAGE
        ) from e

      run_eval_results = []
      for eval_case_result in eval_case_results:
        run_eval_results.append(
            RunEvalResult(
                eval_set_file=eval_case_result.eval_set_file,
                eval_set_id=eval_set_id,
                eval_id=eval_case_result.eval_id,
                final_eval_status=eval_case_result.final_eval_status,
                overall_eval_metric_results=eval_case_result.overall_eval_metric_results,
                eval_metric_result_per_invocation=eval_case_result.eval_metric_result_per_invocation,
                user_id=eval_case_result.user_id,
                session_id=eval_case_result.session_id,
            )
        )

      return RunEvalResponse(run_eval_results=run_eval_results)

    @app.get(
        "/dev/apps/{app_name}/eval-results/{eval_result_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def get_eval_result(
        app_name: str,
        eval_result_id: str,
    ) -> EvalResult:
      """Gets the eval result for the given eval id."""
      try:
        eval_set_result = self.eval_set_results_manager.get_eval_set_result(
            app_name, eval_result_id
        )
        return EvalResult(**eval_set_result.model_dump())
      except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve
      except ValidationError as ve:
        raise HTTPException(status_code=500, detail=str(ve)) from ve

    @app.get(
        "/dev/apps/{app_name}/eval-results",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_results(app_name: str) -> ListEvalResultsResponse:
      """Lists all eval results for the given app."""
      eval_result_ids = self.eval_set_results_manager.list_eval_set_results(
          app_name
      )
      return ListEvalResultsResponse(eval_result_ids=eval_result_ids)

    @app.get(
        "/dev/apps/{app_name}/metrics-info",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_metrics_info(app_name: str) -> ListMetricsInfoResponse:
      """Lists all eval metrics for the given app."""
      try:
        from ..evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY

        # Right now we ignore the app_name as eval metrics are not tied to the
        # app_name, but they could be moving forward.
        metrics_info = (
            DEFAULT_METRIC_EVALUATOR_REGISTRY.get_registered_metrics()
        )
        return ListMetricsInfoResponse(metrics_info=metrics_info)
      except ModuleNotFoundError as e:
        logger.exception("%s\n%s", MISSING_EVAL_DEPENDENCIES_MESSAGE, e)
        raise HTTPException(
            status_code=400, detail=MISSING_EVAL_DEPENDENCIES_MESSAGE
        ) from e

    # ========== GRAPH VISUALIZATION ENDPOINTS ==========

    @app.get(
        "/dev/apps/{app_name}/graph",
        response_model_exclude_none=True,
        tags=[TAG_DEBUG],
    )
    async def get_app_graph_dot(
        app_name: str, dark_mode: bool = False
    ) -> GetEventGraphResult | dict:
      """Returns the base agent graph in DOT format without any highlights.

      This endpoint allows the frontend to fetch the graph structure once
      and compute highlights client-side for better performance.

      Args:
        app_name: The name of the agent/app
        dark_mode: Whether to use dark theme background color
      """
      agent_or_app = self.agent_loader.load_agent(app_name)
      root_agent = self._get_root_agent(agent_or_app)

      # Get graph with NO highlights (empty list) and specified theme
      dot_graph = await agent_graph.get_agent_graph(
          root_agent, [], dark_mode=dark_mode
      )

      if dot_graph and isinstance(dot_graph, graphviz.Digraph):
        return GetEventGraphResult(dot_src=dot_graph.source)
      else:
        return {}

    # TODO: This endpoint can be removed once we update adk web to stop consuming it
    @app.get(
        "/dev/apps/{app_name}/users/{user_id}/sessions/{session_id}/events/{event_id}/graph",
        response_model_exclude_none=True,
        tags=[TAG_DEBUG],
    )
    async def get_event_graph(
        app_name: str, user_id: str, session_id: str, event_id: str
    ):
      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      session_events = session.events if session else []
      event = next((x for x in session_events if x.id == event_id), None)
      if not event:
        return {}

      function_calls = event.get_function_calls()
      function_responses = event.get_function_responses()
      agent_or_app = self.agent_loader.load_agent(app_name)
      root_agent = self._get_root_agent(agent_or_app)
      dot_graph = None
      if function_calls:
        function_call_highlights = []
        for function_call in function_calls:
          from_name = event.author
          to_name = function_call.name
          function_call_highlights.append((from_name, to_name))
          dot_graph = await agent_graph.get_agent_graph(
              root_agent, function_call_highlights
          )
      elif function_responses:
        function_responses_highlights = []
        for function_response in function_responses:
          from_name = function_response.name
          to_name = event.author
          function_responses_highlights.append((from_name, to_name))
          dot_graph = await agent_graph.get_agent_graph(
              root_agent, function_responses_highlights
          )
      else:
        from_name = event.author
        to_name = ""
        dot_graph = await agent_graph.get_agent_graph(
            root_agent, [(from_name, to_name)]
        )
      if dot_graph and isinstance(dot_graph, graphviz.Digraph):
        return GetEventGraphResult(dot_src=dot_graph.source)
      else:
        return {}

  def _navigate_to_node(self, app_info: dict, node_path: str) -> dict | None:
    """Navigate to a specific node in the agent hierarchy.

    Args:
      app_info: The full app info structure
      node_path: Path like "agent1/agent2/agent3"

    Returns:
      The agent data at that path, or None if not found
    """
    if not node_path:
      return app_info.get("root_agent")

    # Strip leading/trailing slashes and split, filter out empty strings
    path_parts = [p for p in node_path.strip("/").split("/") if p]
    current = app_info.get("root_agent")

    if not current:
      return None

    # Navigate through each level (skip first if it's the root name)
    start_idx = 0
    if path_parts[0] == current.get("name"):
      start_idx = 1

    for part in path_parts[start_idx:]:
      found = None
      # Check potential containers in order of preference
      containers = []
      if current.get("graph") and current["graph"].get("nodes"):
        containers.append(current["graph"]["nodes"])
      if current.get("nodes"):
        containers.append(current["nodes"])
      if current.get("sub_agents"):
        containers.append(current["sub_agents"])

      for container in containers:
        for item in container:
          if item.get("name") == part:
            found = item
            break
        if found:
          break

      if not found:
        return None
      current = found

    return current

  def _get_all_sub_workflows(
      self, app_info: dict, current_path: str = ""
  ) -> dict[str, dict]:
    """Recursively discover all sub-workflows within the given app info.

    Args:
      app_info: Current app_info snippet or agent dict
      current_path: The accumulated string path (e.g., 'agent_a/workflow_b')

    Returns:
      A dictionary mapping the node path to the corresponding agent info dict.
    """
    workflows = {}

    agent_info = app_info.get("root_agent", app_info)
    if agent_info.get("graph"):
      workflows[current_path] = agent_info

    children = list(agent_info.get("sub_agents", []))
    children.extend(agent_info.get("nodes", []))
    graph = agent_info.get("graph")
    if graph:
      children.extend(graph.get("nodes", []))

    for child in children:
      child_name = child.get("name")
      if not child_name:
        continue
      child_path = (
          f"{current_path}/{child_name}" if current_path else child_name
      )
      workflows.update(
          self._get_all_sub_workflows({"root_agent": child}, child_path)
      )

    return workflows

  def get_fast_api_app(self, **kwargs):
    """Override to add dev endpoints after production endpoints.

    Calls parent's get_fast_api_app() to get the base app with production
    endpoints, then registers dev-only endpoints.
    """
    app = super().get_fast_api_app(**kwargs)

    web_assets_dir = kwargs.get("web_assets_dir", None)
    self._register_dev_endpoints(
        app, self._trace_dict, self._memory_exporter, web_assets_dir
    )

    return app

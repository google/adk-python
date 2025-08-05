# Copyright 2025 Google LLC
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

import asyncio
import collections
from contextlib import asynccontextmanager
from datetime import datetime
import functools
import logging
import os
import tempfile
from typing import Optional

import click
from click.core import ParameterSource
from fastapi import FastAPI
import uvicorn

from . import cli_create
from . import cli_deploy
from .. import version
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from .cli import run_cli
from .fast_api import get_fast_api_app
from .utils import envs
from .utils import evals
from .utils import logs

LOG_LEVELS = click.Choice(
    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    case_sensitive=False,
)


class HelpfulCommand(click.Command):
  """Command that shows full help on error instead of just the error message.

  A custom Click Command class that overrides the default error handling
  behavior to display the full help text when a required argument is missing,
  followed by the error message. This provides users with better context
  about command usage without needing to run a separate --help command.

  Args:
    *args: Variable length argument list to pass to the parent class.
    **kwargs: Arbitrary keyword arguments to pass to the parent class.

  Returns:
    None. Inherits behavior from the parent Click Command class.

  Returns:
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @staticmethod
  def _format_missing_arg_error(click_exception):
    """Format the missing argument error with uppercase parameter name.

    Args:
      click_exception: The MissingParameter exception from Click.

    Returns:
      str: Formatted error message with uppercase parameter name.
    """
    name = click_exception.param.name
    return f"Missing required argument: {name.upper()}"

  def parse_args(self, ctx, args):
    """Override the parse_args method to show help text on error.

    Args:
      ctx: Click context object for the current command.
      args: List of command-line arguments to parse.

    Returns:
      The parsed arguments as returned by the parent class's parse_args method.

    Raises:
      click.MissingParameter: When a required parameter is missing, but this
        is caught and handled by displaying the help text before exiting.
    """
    try:
      return super().parse_args(ctx, args)
    except click.MissingParameter as exc:
      error_message = self._format_missing_arg_error(exc)

      click.echo(ctx.get_help())
      click.secho(f"\nError: {error_message}", fg="red", err=True)
      ctx.exit(2)


logger = logging.getLogger("google_adk." + __name__)


@click.group(context_settings={"max_content_width": 240})
@click.version_option(version.__version__)
def main():
  """Agent Development Kit CLI tools."""
  pass


@main.group()
def deploy():
  """Deploys agent to hosted environments."""
  pass


@main.command("create", cls=HelpfulCommand)
@click.option(
    "--model",
    type=str,
    help="Optional. The model used for the root agent.",
)
@click.option(
    "--api_key",
    type=str,
    help=(
        "Optional. The API Key needed to access the model, e.g. Google AI API"
        " Key."
    ),
)
@click.option(
    "--project",
    type=str,
    help="Optional. The Google Cloud Project for using VertexAI as backend.",
)
@click.option(
    "--region",
    type=str,
    help="Optional. The Google Cloud Region for using VertexAI as backend.",
)
@click.option(
    "--type",
    type=click.Choice([t.value for t in cli_create.Type]),
    help=(
        "EXPERIMENTAL Optional. Type of agent to create: 'config' or 'code'."
        " 'config' is not ready for use so it defaults to 'code'. It may change"
        " later once 'config' is ready for use."
    ),
    default=cli_create.Type.CODE.value,
    show_default=True,
    hidden=True,  # Won't show in --help output. Not ready for use.
)
@click.argument("app_name", type=str, required=True)
def cli_create_cmd(
    app_name: str,
    model: Optional[str],
    api_key: Optional[str],
    project: Optional[str],
    region: Optional[str],
    type: Optional[cli_create.Type],
):
  """Creates a new app in the current folder with prepopulated agent template.

  APP_NAME: required, the folder of the agent source code.

  Example:

    adk create path/to/my_app
  """
  cli_create.run_cmd(
      app_name,
      model=model,
      google_api_key=api_key,
      google_cloud_project=project,
      google_cloud_region=region,
      type=type,
  )


def validate_exclusive(ctx, param, value):
  # Store the validated parameters in the context
  if not hasattr(ctx, "exclusive_opts"):
    ctx.exclusive_opts = {}

  # If this option has a value and we've already seen another exclusive option
  if value is not None and any(ctx.exclusive_opts.values()):
    exclusive_opt = next(key for key, val in ctx.exclusive_opts.items() if val)
    raise click.UsageError(
        f"Options '{param.name}' and '{exclusive_opt}' cannot be set together."
    )

  # Record this option's value
  ctx.exclusive_opts[param.name] = value is not None
  return value


@main.command("run", cls=HelpfulCommand)
@click.option(
    "--save_session",
    type=bool,
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to save the session to a json file on exit.",
)
@click.option(
    "--session_id",
    type=str,
    help=(
        "Optional. The session ID to save the session to on exit when"
        " --save_session is set to true. User will be prompted to enter a"
        " session ID if not set."
    ),
)
@click.option(
    "--replay",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help=(
        "The json file that contains the initial state of the session and user"
        " queries. A new session will be created using this state. And user"
        " queries are run against the newly created session. Users cannot"
        " continue to interact with the agent."
    ),
    callback=validate_exclusive,
)
@click.option(
    "--resume",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help=(
        "The json file that contains a previously saved session (by"
        "--save_session option). The previous session will be re-displayed. And"
        " user can continue to interact with the agent."
    ),
    callback=validate_exclusive,
)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
def cli_run(
    agent: str,
    save_session: bool,
    session_id: Optional[str],
    replay: Optional[str],
    resume: Optional[str],
):
  """Runs an interactive CLI for a certain agent.

  AGENT: The path to the agent source code folder.

  Example:

    adk run path/to/my_agent
  """
  logs.log_to_tmp_folder()

  agent_parent_folder = os.path.dirname(agent)
  agent_folder_name = os.path.basename(agent)

  asyncio.run(
      run_cli(
          agent_parent_dir=agent_parent_folder,
          agent_folder_name=agent_folder_name,
          input_file=replay,
          saved_session_file=resume,
          save_session=save_session,
          session_id=session_id,
      )
  )


@main.command("eval", cls=HelpfulCommand)
@click.argument(
    "agent_module_file_path",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.argument("eval_set_file_path_or_id", nargs=-1)
@click.option("--config_file_path", help="Optional. The path to config file.")
@click.option(
    "--print_detailed_results",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to print detailed results on console or not.",
)
@click.option(
    "--eval_storage_uri",
    type=str,
    help=(
        "Optional. The evals storage URI to store agent evals,"
        " supported URIs: gs://<bucket name>."
    ),
    default=None,
)
def cli_eval(
    agent_module_file_path: str,
    eval_set_file_path_or_id: list[str],
    config_file_path: str,
    print_detailed_results: bool,
    eval_storage_uri: Optional[str] = None,
):
  """Evaluates an agent given the eval sets.

  AGENT_MODULE_FILE_PATH: The path to the __init__.py file that contains a
  module by the name "agent". "agent" module contains a root_agent.

  EVAL_SET_FILE_PATH_OR_ID: You can specify one or more eval set file paths or
  eval set id.

  Mixing of eval set file paths with eval set ids is not allowed.

  *Eval Set File Path*
  For each file, all evals will be run by default.

  If you want to run only specific evals from a eval set, first create a comma
  separated list of eval names and then add that as a suffix to the eval set
  file name, demarcated by a `:`.

  For example, we have `sample_eval_set_file.json` file that has following the
  eval cases:
  sample_eval_set_file.json:
    |....... eval_1
    |....... eval_2
    |....... eval_3
    |....... eval_4
    |....... eval_5

  sample_eval_set_file.json:eval_1,eval_2,eval_3

  This will only run eval_1, eval_2 and eval_3 from sample_eval_set_file.json.

  *Eval Set Id*
  For each eval set, all evals will be run by default.

  If you want to run only specific evals from a eval set, first create a comma
  separated list of eval names and then add that as a suffix to the eval set
  file name, demarcated by a `:`.

  For example, we have `sample_eval_set_id` that has following the eval cases:
  sample_eval_set_id:
    |....... eval_1
    |....... eval_2
    |....... eval_3
    |....... eval_4
    |....... eval_5

  If we did:
      sample_eval_set_id:eval_1,eval_2,eval_3

  This will only run eval_1, eval_2 and eval_3 from sample_eval_set_id.

  CONFIG_FILE_PATH: The path to config file.

  PRINT_DETAILED_RESULTS: Prints detailed results on the console.
  """
  envs.load_dotenv_for_agent(agent_module_file_path, ".")

  try:
    from ..evaluation.base_eval_service import InferenceConfig
    from ..evaluation.base_eval_service import InferenceRequest
    from ..evaluation.eval_metrics import EvalMetric
    from ..evaluation.eval_metrics import JudgeModelOptions
    from ..evaluation.eval_result import EvalCaseResult
    from ..evaluation.evaluator import EvalStatus
    from ..evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
    from ..evaluation.local_eval_service import LocalEvalService
    from ..evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
    from ..evaluation.local_eval_sets_manager import load_eval_set_from_file
    from ..evaluation.local_eval_sets_manager import LocalEvalSetsManager
    from .cli_eval import _collect_eval_results
    from .cli_eval import _collect_inferences
    from .cli_eval import get_evaluation_criteria_or_default
    from .cli_eval import get_root_agent
    from .cli_eval import parse_and_get_evals_to_run
  except ModuleNotFoundError as mnf:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE) from mnf

  evaluation_criteria = get_evaluation_criteria_or_default(config_file_path)
  eval_metrics = []
  for metric_name, threshold in evaluation_criteria.items():
    eval_metrics.append(
        EvalMetric(
            metric_name=metric_name,
            threshold=threshold,
            judge_model_options=JudgeModelOptions(),
        )
    )

  print(f"Using evaluation criteria: {evaluation_criteria}")

  root_agent = get_root_agent(agent_module_file_path)
  app_name = os.path.basename(agent_module_file_path)
  agents_dir = os.path.dirname(agent_module_file_path)
  eval_sets_manager = None
  eval_set_results_manager = None

  if eval_storage_uri:
    gcs_eval_managers = evals.create_gcs_eval_managers_from_uri(
        eval_storage_uri
    )
    eval_sets_manager = gcs_eval_managers.eval_sets_manager
    eval_set_results_manager = gcs_eval_managers.eval_set_results_manager
  else:
    eval_set_results_manager = LocalEvalSetResultsManager(agents_dir=agents_dir)

  inference_requests = []
  eval_set_file_or_id_to_evals = parse_and_get_evals_to_run(
      eval_set_file_path_or_id
  )

  # Check if the first entry is a file that exists, if it does then we assume
  # rest of the entries are also files. We enforce this assumption in the if
  # block.
  if eval_set_file_or_id_to_evals and os.path.exists(
      list(eval_set_file_or_id_to_evals.keys())[0]
  ):
    eval_sets_manager = InMemoryEvalSetsManager()

    # Read the eval_set files and get the cases.
    for (
        eval_set_file_path,
        eval_case_ids,
    ) in eval_set_file_or_id_to_evals.items():
      try:
        eval_set = load_eval_set_from_file(
            eval_set_file_path, eval_set_file_path
        )
      except FileNotFoundError as fne:
        raise click.ClickException(
            f"`{eval_set_file_path}` should be a valid eval set file."
        ) from fne

      eval_sets_manager.create_eval_set(
          app_name=app_name, eval_set_id=eval_set.eval_set_id
      )
      for eval_case in eval_set.eval_cases:
        eval_sets_manager.add_eval_case(
            app_name=app_name,
            eval_set_id=eval_set.eval_set_id,
            eval_case=eval_case,
        )
      inference_requests.append(
          InferenceRequest(
              app_name=app_name,
              eval_set_id=eval_set.eval_set_id,
              eval_case_ids=eval_case_ids,
              inference_config=InferenceConfig(),
          )
      )
  else:
    # We assume that what we have are eval set ids instead.
    eval_sets_manager = (
        eval_sets_manager
        if eval_storage_uri
        else LocalEvalSetsManager(agents_dir=agents_dir)
    )

    for eval_set_id_key, eval_case_ids in eval_set_file_or_id_to_evals.items():
      inference_requests.append(
          InferenceRequest(
              app_name=app_name,
              eval_set_id=eval_set_id_key,
              eval_case_ids=eval_case_ids,
              inference_config=InferenceConfig(),
          )
      )

  try:
    eval_service = LocalEvalService(
        root_agent=root_agent,
        eval_sets_manager=eval_sets_manager,
        eval_set_results_manager=eval_set_results_manager,
    )

    inference_results = asyncio.run(
        _collect_inferences(
            inference_requests=inference_requests, eval_service=eval_service
        )
    )
    eval_results = asyncio.run(
        _collect_eval_results(
            inference_results=inference_results,
            eval_service=eval_service,
            eval_metrics=eval_metrics,
        )
    )
  except ModuleNotFoundError as mnf:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE) from mnf

  print("*********************************************************************")
  eval_run_summary = {}

  for eval_result in eval_results:
    eval_result: EvalCaseResult

    if eval_result.eval_set_id not in eval_run_summary:
      eval_run_summary[eval_result.eval_set_id] = [0, 0]

    if eval_result.final_eval_status == EvalStatus.PASSED:
      eval_run_summary[eval_result.eval_set_id][0] += 1
    else:
      eval_run_summary[eval_result.eval_set_id][1] += 1
  print("Eval Run Summary")
  for eval_set_id, pass_fail_count in eval_run_summary.items():
    print(
        f"{eval_set_id}:\n  Tests passed: {pass_fail_count[0]}\n  Tests"
        f" failed: {pass_fail_count[1]}"
    )

  if print_detailed_results:
    for eval_result in eval_results:
      eval_result: EvalCaseResult
      print(
          "*********************************************************************"
      )
      print(eval_result.model_dump_json(indent=2))


def adk_services_options():
  """Decorator to add ADK services options to click commands."""

  def decorator(func):
    @click.option(
        "--session_service_uri",
        help=(
            """Optional. The URI of the session service.
          - Use 'agentengine://<agent_engine>' to connect to Agent Engine
            sessions. <agent_engine> can either be the full qualified resource
            name 'projects/abc/locations/us-central1/reasoningEngines/123' or
            the resource id '123'.
          - Use 'sqlite://<path_to_sqlite_file>' to connect to a SQLite DB.
          - See https://docs.sqlalchemy.org/en/20/core/engines.html#backend-specific-urls for more details on supported database URIs."""
        ),
    )
    @click.option(
        "--artifact_service_uri",
        type=str,
        help=(
            "Optional. The URI of the artifact service,"
            " supported URIs: gs://<bucket name> for GCS artifact service."
        ),
        default=None,
    )
    @click.option(
        "--memory_service_uri",
        type=str,
        help=("""Optional. The URI of the memory service.
            - Use 'rag://<rag_corpus_id>' to connect to Vertex AI Rag Memory Service.
            - Use 'agentengine://<agent_engine>' to connect to Agent Engine
              sessions. <agent_engine> can either be the full qualified resource
              name 'projects/abc/locations/us-central1/reasoningEngines/123' or
              the resource id '123'."""),
        default=None,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)

    return wrapper

  return decorator


def deprecated_adk_services_options():
  """Depracated ADK services options."""

  def warn(alternative_param, ctx, param, value):
    if value:
      click.echo(
          click.style(
              f"WARNING: Deprecated option {param.name} is used. Please use"
              f" {alternative_param} instead.",
              fg="yellow",
          ),
          err=True,
      )
    return value

  def decorator(func):
    @click.option(
        "--session_db_url",
        help="Deprecated. Use --session_service_uri instead.",
        callback=functools.partial(warn, "--session_service_uri"),
    )
    @click.option(
        "--artifact_storage_uri",
        type=str,
        help="Deprecated. Use --artifact_service_uri instead.",
        callback=functools.partial(warn, "--artifact_service_uri"),
        default=None,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)

    return wrapper

  return decorator


def fast_api_common_options():
  """Decorator to add common fast api options to click commands."""

  def decorator(func):
    @click.option(
        "--host",
        type=str,
        help="Optional. The binding host of the server",
        default="127.0.0.1",
        show_default=True,
    )
    @click.option(
        "--port",
        type=int,
        help="Optional. The port of the server",
        default=8000,
    )
    @click.option(
        "--allow_origins",
        help="Optional. Any additional origins to allow for CORS.",
        multiple=True,
    )
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        show_default=True,
        default=False,
        help="Enable verbose (DEBUG) logging. Shortcut for --log_level DEBUG.",
    )
    @click.option(
        "--log_level",
        type=LOG_LEVELS,
        default="INFO",
        help="Optional. Set the logging level",
    )
    @click.option(
        "--trace_to_cloud",
        is_flag=True,
        show_default=True,
        default=False,
        help="Optional. Whether to enable cloud trace for telemetry.",
    )
    @click.option(
        "--reload/--no-reload",
        default=True,
        help=(
            "Optional. Whether to enable auto reload for server. Not supported"
            " for Cloud Run."
        ),
    )
    @click.option(
        "--a2a",
        is_flag=True,
        show_default=True,
        default=False,
        help="Optional. Whether to enable A2A endpoint.",
    )
    @click.option(
        "--reload_agents",
        is_flag=True,
        default=False,
        show_default=True,
        help="Optional. Whether to enable live reload for agents changes.",
    )
    @click.option(
        "--eval_storage_uri",
        type=str,
        help=(
            "Optional. The evals storage URI to store agent evals,"
            " supported URIs: gs://<bucket name>."
        ),
        default=None,
    )
    @functools.wraps(func)
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
      # If verbose flag is set and log level is not set, set log level to DEBUG.
      log_level_source = ctx.get_parameter_source("log_level")
      if (
          kwargs.pop("verbose", False)
          and log_level_source == ParameterSource.DEFAULT
      ):
        kwargs["log_level"] = "DEBUG"

      return func(*args, **kwargs)

    return wrapper

  return decorator


@main.command("web")
@fast_api_common_options()
@adk_services_options()
@deprecated_adk_services_options()
@click.argument(
    "agents_dir",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    default=os.getcwd,
)
def cli_web(
    agents_dir: str,
    eval_storage_uri: Optional[str] = None,
    log_level: str = "INFO",
    allow_origins: Optional[list[str]] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    trace_to_cloud: bool = False,
    reload: bool = True,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    session_db_url: Optional[str] = None,  # Deprecated
    artifact_storage_uri: Optional[str] = None,  # Deprecated
    a2a: bool = False,
    reload_agents: bool = False,
):
  """Starts a FastAPI server with Web UI for agents.

  AGENTS_DIR: The directory of agents, where each sub-directory is a single
  agent, containing at least `__init__.py` and `agent.py` files.

  Example:

    adk web --session_service_uri=[uri] --port=[port] path/to/agents_dir
  """
  logs.setup_adk_logger(getattr(logging, log_level.upper()))

  @asynccontextmanager
  async def _lifespan(app: FastAPI):
    click.secho(
        f"""
+-----------------------------------------------------------------------------+
| ADK Web Server started                                                      |
|                                                                             |
| For local testing, access at http://localhost:{port}.{" "*(29 - len(str(port)))}|
+-----------------------------------------------------------------------------+
""",
        fg="green",
    )
    yield  # Startup is done, now app is running
    click.secho(
        """
+-----------------------------------------------------------------------------+
| ADK Web Server shutting down...                                             |
+-----------------------------------------------------------------------------+
""",
        fg="green",
    )

  session_service_uri = session_service_uri or session_db_url
  artifact_service_uri = artifact_service_uri or artifact_storage_uri
  app = get_fast_api_app(
      agents_dir=agents_dir,
      session_service_uri=session_service_uri,
      artifact_service_uri=artifact_service_uri,
      memory_service_uri=memory_service_uri,
      eval_storage_uri=eval_storage_uri,
      allow_origins=allow_origins,
      web=True,
      trace_to_cloud=trace_to_cloud,
      lifespan=_lifespan,
      a2a=a2a,
      host=host,
      port=port,
      reload_agents=reload_agents,
  )
  config = uvicorn.Config(
      app,
      host=host,
      port=port,
      reload=reload,
  )

  server = uvicorn.Server(config)
  server.run()


@main.command("api_server")
# The directory of agents, where each sub-directory is a single agent.
# By default, it is the current working directory
@click.argument(
    "agents_dir",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    default=os.getcwd(),
)
@fast_api_common_options()
@adk_services_options()
@deprecated_adk_services_options()
def cli_api_server(
    agents_dir: str,
    eval_storage_uri: Optional[str] = None,
    log_level: str = "INFO",
    allow_origins: Optional[list[str]] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    trace_to_cloud: bool = False,
    reload: bool = True,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    session_db_url: Optional[str] = None,  # Deprecated
    artifact_storage_uri: Optional[str] = None,  # Deprecated
    a2a: bool = False,
    reload_agents: bool = False,
):
  """Starts a FastAPI server for agents.

  AGENTS_DIR: The directory of agents, where each sub-directory is a single
  agent, containing at least `__init__.py` and `agent.py` files.

  Example:

    adk api_server --session_service_uri=[uri] --port=[port] path/to/agents_dir
  """
  logs.setup_adk_logger(getattr(logging, log_level.upper()))

  session_service_uri = session_service_uri or session_db_url
  artifact_service_uri = artifact_service_uri or artifact_storage_uri
  config = uvicorn.Config(
      get_fast_api_app(
          agents_dir=agents_dir,
          session_service_uri=session_service_uri,
          artifact_service_uri=artifact_service_uri,
          memory_service_uri=memory_service_uri,
          eval_storage_uri=eval_storage_uri,
          allow_origins=allow_origins,
          web=False,
          trace_to_cloud=trace_to_cloud,
          a2a=a2a,
          host=host,
          port=port,
          reload_agents=reload_agents,
      ),
      host=host,
      port=port,
      reload=reload,
  )
  server = uvicorn.Server(config)
  server.run()


@deploy.command("cloud_run")
@click.option(
    "--project",
    type=str,
    help=(
        "Required. Google Cloud project to deploy the agent. When absent,"
        " default project from gcloud config is used."
    ),
)
@click.option(
    "--region",
    type=str,
    help=(
        "Required. Google Cloud region to deploy the agent. When absent,"
        " gcloud run deploy will prompt later."
    ),
)
@click.option(
    "--service_name",
    type=str,
    default="adk-default-service-name",
    help=(
        "Optional. The service name to use in Cloud Run (default:"
        " 'adk-default-service-name')."
    ),
)
@click.option(
    "--app_name",
    type=str,
    default="",
    help=(
        "Optional. App name of the ADK API server (default: the folder name"
        " of the AGENT source code)."
    ),
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Optional. The port of the ADK API server (default: 8000).",
)
@click.option(
    "--trace_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable Cloud Trace for cloud run.",
)
@click.option(
    "--with_ui",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Deploy ADK Web UI if set. (default: deploy ADK API server"
        " only)"
    ),
)
@click.option(
    "--temp_folder",
    type=str,
    default=os.path.join(
        tempfile.gettempdir(),
        "cloud_run_deploy_src",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
    help=(
        "Optional. Temp folder for the generated Cloud Run source files"
        " (default: a timestamped folder in the system temp directory)."
    ),
)
@click.option(
    "--log_level",
    type=LOG_LEVELS,
    default="INFO",
    help="Optional. Set the logging level",
)
@click.option(
    "--verbosity",
    type=LOG_LEVELS,
    help="Deprecated. Use --log_level instead.",
)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.option(
    "--adk_version",
    type=str,
    default=version.__version__,
    show_default=True,
    help=(
        "Optional. The ADK version used in Cloud Run deployment. (default: the"
        " version in the dev environment)"
    ),
)
@click.option(
    "--a2a",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable A2A endpoint.",
)
@click.option(
    "--allow_origins",
    help="Optional. Any additional origins to allow for CORS.",
    multiple=True,
)
# TODO: Add eval_storage_uri option back when evals are supported in Cloud Run.
@adk_services_options()
@deprecated_adk_services_options()
def cli_deploy_cloud_run(
    agent: str,
    project: Optional[str],
    region: Optional[str],
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    with_ui: bool,
    adk_version: str,
    log_level: str,
    verbosity: Optional[str],
    allow_origins: Optional[list[str]] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    session_db_url: Optional[str] = None,  # Deprecated
    artifact_storage_uri: Optional[str] = None,  # Deprecated
    a2a: bool = False,
):
  """Deploys an agent to Cloud Run.

  AGENT: The path to the agent source code folder.

  Example:

    adk deploy cloud_run --project=[project] --region=[region] path/to/my_agent
  """
  if verbosity:
    click.secho(
        "WARNING: The --verbosity option is deprecated. Use --log_level"
        " instead.",
        fg="yellow",
        err=True,
    )

  session_service_uri = session_service_uri or session_db_url
  artifact_service_uri = artifact_service_uri or artifact_storage_uri
  try:
    cli_deploy.to_cloud_run(
        agent_folder=agent,
        project=project,
        region=region,
        service_name=service_name,
        app_name=app_name,
        temp_folder=temp_folder,
        port=port,
        trace_to_cloud=trace_to_cloud,
        allow_origins=allow_origins,
        with_ui=with_ui,
        log_level=log_level,
        verbosity=verbosity,
        adk_version=adk_version,
        session_service_uri=session_service_uri,
        artifact_service_uri=artifact_service_uri,
        memory_service_uri=memory_service_uri,
        a2a=a2a,
    )
  except Exception as e:
    click.secho(f"Deploy failed: {e}", fg="red", err=True)


@deploy.command("agent_engine")
@click.option(
    "--project",
    type=str,
    help=(
        "Required. Google Cloud project to deploy the agent. It will override"
        " GOOGLE_CLOUD_PROJECT in the .env file (if it exists)."
    ),
)
@click.option(
    "--region",
    type=str,
    help=(
        "Required. Google Cloud region to deploy the agent. It will override"
        " GOOGLE_CLOUD_LOCATION in the .env file (if it exists)."
    ),
)
@click.option(
    "--staging_bucket",
    type=str,
    help="Required. GCS bucket for staging the deployment artifacts.",
)
@click.option(
    "--agent_engine_id",
    type=str,
    default=None,
    help=(
        "Optional. ID of the Agent Engine instance to update if it exists"
        " (default: None, which means a new instance will be created)."
        " The corresponding resource name in Agent Engine will be:"
        " `projects/{project}/locations/{region}/reasoningEngines/{agent_engine_id}`."
    ),
)
@click.option(
    "--trace_to_cloud",
    type=bool,
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable Cloud Trace for Agent Engine.",
)
@click.option(
    "--display_name",
    type=str,
    show_default=True,
    default="",
    help="Optional. Display name of the agent in Agent Engine.",
)
@click.option(
    "--description",
    type=str,
    show_default=True,
    default="",
    help="Optional. Description of the agent in Agent Engine.",
)
@click.option(
    "--adk_app",
    type=str,
    default="agent_engine_app",
    help=(
        "Optional. Python file for defining the ADK application"
        " (default: a file named agent_engine_app.py)"
    ),
)
@click.option(
    "--temp_folder",
    type=str,
    default=os.path.join(
        tempfile.gettempdir(),
        "agent_engine_deploy_src",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
    help=(
        "Optional. Temp folder for the generated Agent Engine source files."
        " If the folder already exists, its contents will be removed."
        " (default: a timestamped folder in the system temp directory)."
    ),
)
@click.option(
    "--env_file",
    type=str,
    default="",
    help=(
        "Optional. The filepath to the `.env` file for environment variables."
        " (default: the `.env` file in the `agent` directory, if any.)"
    ),
)
@click.option(
    "--requirements_file",
    type=str,
    default="",
    help=(
        "Optional. The filepath to the `requirements.txt` file to use."
        " (default: the `requirements.txt` file in the `agent` directory, if"
        " any.)"
    ),
)
@click.option(
    "--absolutize_imports",
    type=bool,
    default=True,
    help=(
        "Optional. Whether to absolutize imports. If True, all relative imports"
        " will be converted to absolute import statements (default: True)."
        " NOTE: This flag is temporary and will be removed in the future."
    ),
)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
def cli_deploy_agent_engine(
    agent: str,
    project: str,
    region: str,
    staging_bucket: str,
    agent_engine_id: Optional[str],
    trace_to_cloud: bool,
    display_name: str,
    description: str,
    adk_app: str,
    temp_folder: str,
    env_file: str,
    requirements_file: str,
    absolutize_imports: bool,
):
  """Deploys an agent to Agent Engine.

  Example:

    adk deploy agent_engine --project=[project] --region=[region]
      --staging_bucket=[staging_bucket] --display_name=[app_name] path/to/my_agent
  """
  try:
    cli_deploy.to_agent_engine(
        agent_folder=agent,
        project=project,
        region=region,
        staging_bucket=staging_bucket,
        agent_engine_id=agent_engine_id,
        trace_to_cloud=trace_to_cloud,
        display_name=display_name,
        description=description,
        adk_app=adk_app,
        temp_folder=temp_folder,
        env_file=env_file,
        requirements_file=requirements_file,
        absolutize_imports=absolutize_imports,
    )
  except Exception as e:
    click.secho(f"Deploy failed: {e}", fg="red", err=True)


@deploy.command("gke")
@click.option(
    "--project",
    type=str,
    help=(
        "Required. Google Cloud project to deploy the agent. When absent,"
        " default project from gcloud config is used."
    ),
)
@click.option(
    "--region",
    type=str,
    help=(
        "Required. Google Cloud region to deploy the agent. When absent,"
        " gcloud run deploy will prompt later."
    ),
)
@click.option(
    "--cluster_name",
    type=str,
    help="Required. The name of the GKE cluster.",
)
@click.option(
    "--service_name",
    type=str,
    default="adk-default-service-name",
    help=(
        "Optional. The service name to use in GKE (default:"
        " 'adk-default-service-name')."
    ),
)
@click.option(
    "--app_name",
    type=str,
    default="",
    help=(
        "Optional. App name of the ADK API server (default: the folder name"
        " of the AGENT source code)."
    ),
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Optional. The port of the ADK API server (default: 8000).",
)
@click.option(
    "--trace_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable Cloud Trace for GKE.",
)
@click.option(
    "--with_ui",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Deploy ADK Web UI if set. (default: deploy ADK API server"
        " only)"
    ),
)
@click.option(
    "--log_level",
    type=LOG_LEVELS,
    default="INFO",
    help="Optional. Set the logging level",
)
@click.option(
    "--temp_folder",
    type=str,
    default=os.path.join(
        tempfile.gettempdir(),
        "gke_deploy_src",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
    help=(
        "Optional. Temp folder for the generated GKE source files"
        " (default: a timestamped folder in the system temp directory)."
    ),
)
@click.option(
    "--adk_version",
    type=str,
    default=version.__version__,
    show_default=True,
    help=(
        "Optional. The ADK version used in GKE deployment. (default: the"
        " version in the dev environment)"
    ),
)
@adk_services_options()
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
def cli_deploy_gke(
    agent: str,
    project: Optional[str],
    region: Optional[str],
    cluster_name: str,
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    with_ui: bool,
    adk_version: str,
    log_level: Optional[str] = None,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
):
  """Deploys an agent to GKE.

  AGENT: The path to the agent source code folder.

  Example:

    adk deploy gke --project=[project] --region=[region] --cluster_name=[cluster_name] path/to/my_agent
  """
  try:
    cli_deploy.to_gke(
        agent_folder=agent,
        project=project,
        region=region,
        cluster_name=cluster_name,
        service_name=service_name,
        app_name=app_name,
        temp_folder=temp_folder,
        port=port,
        trace_to_cloud=trace_to_cloud,
        with_ui=with_ui,
        log_level=log_level,
        adk_version=adk_version,
        session_service_uri=session_service_uri,
        artifact_service_uri=artifact_service_uri,
        memory_service_uri=memory_service_uri,
    )
  except Exception as e:
    click.secho(f"Deploy failed: {e}", fg="red", err=True)

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

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os
import tempfile
from typing import Optional

import click
from fastapi import FastAPI
import uvicorn

from . import cli_create
from . import cli_deploy
from .. import version
from .cli import run_cli
from .cli_eval import MISSING_EVAL_DEPENDENCIES_MESSAGE
from .fast_api import get_fast_api_app
from .utils import envs
from .utils import logs


class HelpfulCommand(click.Command):
    """Command that shows full help on error instead of just the error message."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        try:
            return super().parse_args(ctx, args)
        except click.MissingParameter as exc:
            click.echo(ctx.get_help())
            click.secho(f"\nError: {str(exc)}", fg="red", err=True)
            ctx.exit(2)


logger = logging.getLogger(__name__)


@click.group(context_settings={"max_content_width": 240})
def main():
    """Agent Development Kit CLI tools."""
    pass


@main.group()
def deploy():
    """Deploys agent to hosted environments."""
    pass


@main.command("create", cls=HelpfulCommand)
@click.option("--model", type=str, help="Optional. The model used for the root agent.")
@click.option("--api_key", type=str, help="Optional. The API Key needed to access the model, e.g. Google AI API Key.")
@click.option("--project", type=str, help="Optional. The Google Cloud Project for using VertexAI as backend.")
@click.option("--region", type=str, help="Optional. The Google Cloud Region for using VertexAI as backend.")
@click.argument("app_name", type=str, required=True)
def cli_create_cmd(app_name: str, model: Optional[str], api_key: Optional[str], project: Optional[str], region: Optional[str]):
    """Creates a new app in the current folder with prepopulated agent template."""
    cli_create.run_cmd(
        app_name,
        model=model,
        google_api_key=api_key,
        google_cloud_project=project,
        google_cloud_region=region,
    )


def validate_exclusive(ctx, param, value):
    if not hasattr(ctx, "exclusive_opts"):
        ctx.exclusive_opts = {}

    if value is not None and any(ctx.exclusive_opts.values()):
        exclusive_opt = next(key for key, val in ctx.exclusive_opts.items() if val)
        raise click.UsageError(f"Options '{param.name}' and '{exclusive_opt}' cannot be set together.")

    ctx.exclusive_opts[param.name] = value is not None
    return value


@main.command("run", cls=HelpfulCommand)
@click.option("--save_session", type=bool, is_flag=True, show_default=True, default=False, help="Optional. Whether to save the session to a json file on exit.")
@click.option("--session_id", type=str, help="Optional. The session ID to save the session to on exit when --save_session is set to true.")
@click.option("--replay", type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True), help="The json file that contains the initial state of the session.", callback=validate_exclusive)
@click.option("--resume", type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True), help="The json file that contains a previously saved session.", callback=validate_exclusive)
@click.argument("agent", type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True))
def cli_run(agent: str, save_session: bool, session_id: Optional[str], replay: Optional[str], resume: Optional[str]):
    """Runs an interactive CLI for a certain agent."""
    logs.log_to_tmp_folder()
    agent_parent_folder = os.path.dirname(agent)
    agent_folder_name = os.path.basename(agent)

    asyncio.run(run_cli(
        agent_parent_dir=agent_parent_folder,
        agent_folder_name=agent_folder_name,
        input_file=replay,
        saved_session_file=resume,
        save_session=save_session,
        session_id=session_id,
    ))


@main.command("eval", cls=HelpfulCommand)
@click.argument("agent_module_file_path", type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True))
@click.argument("eval_set_file_path", nargs=-1)
@click.option("--config_file_path", help="Optional. The path to config file.")
@click.option("--print_detailed_results", is_flag=True, show_default=True, default=False, help="Optional. Whether to print detailed results on console or not.")
def cli_eval(agent_module_file_path: str, eval_set_file_path: tuple[str], config_file_path: str, print_detailed_results: bool):
    """Evaluates an agent given the eval sets."""
    envs.load_dotenv_for_agent(agent_module_file_path, ".")

    try:
        from ..evaluation.local_eval_sets_manager import load_eval_set_from_file
        from .cli_eval import EvalCaseResult, EvalMetric, EvalStatus
        from .cli_eval import get_evaluation_criteria_or_default, get_root_agent, parse_and_get_evals_to_run, run_evals, try_get_reset_func
    except ModuleNotFoundError:
        raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE)

    evaluation_criteria = get_evaluation_criteria_or_default(config_file_path)
    eval_metrics = [EvalMetric(metric_name=k, threshold=v) for k, v in evaluation_criteria.items()]
    print(f"Using evaluation criteria: {evaluation_criteria}")

    root_agent = get_root_agent(agent_module_file_path)
    reset_func = try_get_reset_func(agent_module_file_path)

    eval_set_file_path_to_evals = parse_and_get_evals_to_run(eval_set_file_path)
    eval_set_id_to_eval_cases = {}

    for eval_set_file_path, eval_case_ids in eval_set_file_path_to_evals.items():
        eval_set = load_eval_set_from_file(eval_set_file_path, eval_set_file_path)
        eval_cases = eval_set.eval_cases
        if eval_case_ids:
            eval_cases = [e for e in eval_cases if e.eval_id in eval_case_ids]
        eval_set_id_to_eval_cases[eval_set_file_path] = eval_cases

    async def _collect_eval_results() -> list[EvalCaseResult]:
        return [
            result
            async for result in run_evals(
                eval_set_id_to_eval_cases, root_agent, reset_func, eval_metrics
            )
        ]

    eval_results = asyncio.run(_collect_eval_results())

    print("*********************************************************************")
    eval_run_summary = {}
    for eval_result in eval_results:
        if eval_result.eval_set_id not in eval_run_summary:
            eval_run_summary[eval_result.eval_set_id] = [0, 0]

        if eval_result.final_eval_status == EvalStatus.PASSED:
            eval_run_summary[eval_result.eval_set_id][0] += 1
        else:
            eval_run_summary[eval_result.eval_set_id][1] += 1

    print("Eval Run Summary")
    for eval_set_id, pass_fail_count in eval_run_summary.items():
        print(f"{eval_set_id}:\n  Tests passed: {pass_fail_count[0]}\n  Tests failed: {pass_fail_count[1]}")

    if print_detailed_results:
        for eval_result in eval_results:
            print("*********************************************************************")
            print(eval_result.model_dump_json(indent=2))


@main.command("web")
@click.option("--session_db_url", help="Optional. The database URL to store the session.")
@click.option("--host", type=str, default="127.0.0.1", show_default=True, help="Optional. The binding host of the server")
@click.option("--port", type=int, default=8000, help="Optional. The port of the server")
@click.option("--allow_origins", multiple=True, help="Optional. Any additional origins to allow for CORS.")
@click.option("--log_level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False), default="INFO", help="Optional. Set the logging level")
@click.option("--trace_to_cloud", is_flag=True, show_default=True, default=False, help="Optional. Whether to enable cloud trace for telemetry.")
@click.option("--reload/--no-reload", default=True, help="Optional. Whether to enable auto reload for server.")
@click.argument("agents_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True), default=os.getcwd)
def cli_web(
    agents_dir: str,
    session_db_url: str = "",
    log_level: str = "INFO",
    allow_origins: Optional[list[str]] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    trace_to_cloud: bool = False,
    reload: bool = True,
):
    """Starts a FastAPI server with Web UI for agents."""
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
        yield

    app = get_fast_api_app(
        agents_dir=agents_dir,
        session_db_url=session_db_url,
        allow_origins=allow_origins,
        trace_to_cloud=trace_to_cloud,
        lifespan=_lifespan,
    )

    uvicorn.run(app, host=host, port=port, reload=reload)

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

"""Runs Tau-bench."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
from datetime import datetime
import json
import logging
import multiprocessing
import os
import random
import traceback
from typing import Any
from typing import TypedDict

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.adapter import GEPAAdapter
from litellm import provider_list
import rater_lib
from retry import retry
from tau_bench.envs import get_env
from tau_bench.envs.retail import tasks_dev
from tau_bench.envs.retail import tasks_test
from tau_bench.envs.retail import tasks_train
from tau_bench.envs.user import UserStrategy
from tau_bench.run import display_metrics
from tau_bench.types import EnvRunResult
from tau_bench.types import RunConfig
import tau_bench_agent as tau_bench_agent_lib

import utils


def run_tau_bench_rollouts(
    config: RunConfig,
    print_results: bool = False,
    system_instruction: str | None = None,
    rater: rater_lib.Rater | None = None,
) -> list[EnvRunResult]:
  """Runs a set of tau-bench tasks with a given agent configuration.

  This is a customized version of the standard tau-bench run function, adapted
  for this experiment's needs. It handles environment setup, agent creation,
  task execution in parallel, and result aggregation.

  Args:
    config: A RunConfig object specifying the environment, models, and other
      parameters for the run.
    print_results: If True, prints the result of each task as it completes.
    system_instruction: An optional system instruction to use for the agent,
      overriding the default.
    rater: An optional rater to evaluate the agent's performance.

  Returns:
    A list of EnvRunResult objects, one for each completed task.
  """
  if config.env not in ['retail', 'airline']:
    raise ValueError('Only retail and airline envs are supported')
  if config.model_provider not in provider_list:
    raise ValueError('Invalid model provider')
  if config.user_model_provider not in provider_list:
    raise ValueError('Invalid user model provider')
  if config.agent_strategy not in ['tool-calling', 'act', 'react', 'few-shot']:
    raise ValueError('Invalid agent strategy')
  if config.task_split not in ['train', 'test', 'dev']:
    raise ValueError('Invalid task split')
  if config.user_strategy not in [item.value for item in UserStrategy]:
    raise ValueError('Invalid user strategy')

  random.seed(config.seed)
  time_str = datetime.now().strftime('%m%d%H%M%S')
  model_name = config.model.split('/')[-1]
  ckpt_filename = (
      f'{config.agent_strategy}-{model_name}-{config.temperature}_range_'
      f'{config.start_index}-{config.end_index}_user-{config.user_model}-'
      f'{config.user_strategy}_{time_str}.json'
  )
  ckpt_path = os.path.join(config.log_dir, ckpt_filename)
  if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)

  print(f'Loading user with strategy: {config.user_strategy}')
  env = get_env(
      config.env,
      user_strategy=config.user_strategy,
      user_model=config.user_model,
      user_provider=config.user_model_provider,
      task_split=config.task_split,
  )
  if system_instruction:
    env.wiki = system_instruction
  agent = tau_bench_agent_lib.adk_agent_factory(
      tools_info=env.tools_info,
      wiki=env.wiki,
      config=config,
  )
  if config.end_index == -1:
    end_index = len(env.tasks)
  else:
    end_index = min(config.end_index, len(env.tasks))
  results: list[EnvRunResult] = []
  lock = multiprocessing.Lock()
  if config.task_ids:
    print(f'Running tasks {config.task_ids} (checkpoint path: {ckpt_path})')
  else:
    print(
        f'Running tasks {config.start_index} to {end_index} '
        f'(checkpoint path: {ckpt_path})'
    )
  for i in range(config.num_trials):
    if config.task_ids:
      idxs = config.task_ids
    else:
      idxs = list(range(config.start_index, end_index))
    if config.shuffle:
      random.shuffle(idxs)

    @retry(tries=3, delay=10, backoff=2)
    def _run_with_retry(idx: int) -> EnvRunResult:
      isolated_env = get_env(
          config.env,
          user_strategy=config.user_strategy,
          user_model=config.user_model,
          task_split=config.task_split,
          user_provider=config.user_model_provider,
          task_index=idx,
      )
      if print_results:
        print(f'Running task {idx}')
      res = agent.solve(
          env=isolated_env,
          task_index=idx,
      )

      rating = (
          rater(res.messages[1:] if len(res.messages) > 1 else res.messages)
          if rater
          else None
      )
      info = dict(res.info)
      info['metrics'] = dict(rating=rating, reward=res.reward)

      if rater:
        score = rating['score']
        feedback = {k: v for k, v in rating.items() if k != 'score'}
      else:
        score = res.reward
        feedback = (
            'The agent successfully resolved all customer issues'
            if score > 0
            else 'The agent failed to resolve all customer issues correctly'
        )

      info['feedback'] = feedback
      return EnvRunResult(
          task_id=idx,
          reward=score,
          info=info,
          traj=res.messages,
          trial=i,
      )

    def _run(idx: int) -> EnvRunResult:
      try:
        result = _run_with_retry(idx)
      except Exception as e:
        logging.warning('Inference error: %s', str(e))
        result = EnvRunResult(
            task_id=idx,
            reward=0.0,
            info={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'metrics': dict(reward=0.0),
            },
            traj=[],
            trial=i,
        )

      if print_results:
        print(
            '✅' if result.reward == 1 else '❌',
            f'task_id={idx}',
        )
        print('-----')
      with lock:
        data = []
        if os.path.exists(ckpt_path):
          with open(ckpt_path, 'r') as f:
            data = json.load(f)
        with open(ckpt_path, 'w') as f:
          json.dump(data + [result.model_dump()], f, indent=2)
      return result

    with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
      res = list(executor.map(_run, idxs))
      results.extend(res)

  display_metrics(results)

  if rater:
    print('Environment reward:')
    display_metrics([
        EnvRunResult(
            task_id=r.task_id,
            reward=r.info['metrics']['reward'],
            info={},
            traj=[],
            trial=r.trial,
        )
        for r in results
    ])

  with open(ckpt_path, 'w') as f:
    json.dump([result.model_dump() for result in results], f, indent=2)
    print(f'\n📄 Results saved to {ckpt_path}\n')
  return results


class TauBenchDataInst(TypedDict):
  env: str
  task_id: int
  task_split: str


class TauBenchTrajectory(TypedDict):

  result_traj: list[dict[str, Any]]


class TauBenchRolloutOutput(TypedDict):
  env: str
  task_id: int
  reward: float
  task_info: dict[str, Any]


class TauBenchAdapter(
    GEPAAdapter[
        TauBenchDataInst,
        TauBenchTrajectory,
        TauBenchRolloutOutput,
    ]
):
  """A GEPA adapter for evaluating agent performance on tau-bench benchmark."""

  def __init__(
      self,
      env_name: str,
      agent_model: str = 'gemini-2.5-flash',
      agent_model_provider: str = 'vertex_ai',
      user_model: str = 'gemini-2.5-pro',
      user_model_provider: str = 'vertex_ai',
      agent_strategy: str = 'tool-calling',
      user_strategy: str = 'llm',
      system_instruction_name: str = 'system_instruction',
      max_concurrency: int = 4,
      rater: rater_lib.Rater | None = None,
      log_dir: str | None = None,
  ):
    """Initializes the TauBenchAdapter.

    Args:
      env_name: environment
      agent_model: The model to use for the agent.
      agent_model_provider: The provider for the agent model.
      user_model: The model to use for simulating the user.
      user_model_provider: The provider for the user model.
      agent_strategy: The agent strategy to use (e.g., 'tool-calling').
      user_strategy: The user simulation strategy (e.g., 'llm').
      system_instruction_name: The key in the candidate dictionary that holds
        the system instruction.
      max_concurrency: The maximum number of tasks to run in parallel.
      rater: An optional rater to evaluate the agent's performance.
      log_dir: The directory to save traces and other logs.
    """
    self._env_name = env_name
    self._agent_model = agent_model
    self._agent_model_provider = agent_model_provider
    self._user_model = user_model
    self._user_model_provider = user_model_provider
    self._agent_strategy = agent_strategy
    self._user_strategy = user_strategy
    self._max_concurrency = max_concurrency
    self._system_instruction_name = system_instruction_name
    self._rater = rater
    self._log_dir = log_dir

  def evaluate(
      self,
      batch: list[TauBenchDataInst],
      candidate: dict[str, str],
      capture_traces: bool = False,
  ) -> EvaluationBatch[TauBenchTrajectory, TauBenchRolloutOutput]:
    """Evaluates a candidate prompt on a batch of tau-bench tasks.

    This method is called by GEPA during the optimization loop. It takes a
    candidate prompt, runs it against the specified tasks from tau-bench, and
    returns the results.

    Args:
      batch: A list of task instances to evaluate on. Each instance specifies
        the environment and task ID.
      candidate: A dictionary containing the components to be evaluated,
        including the system instruction.
      capture_traces: (Not used in this adapter) Whether to capture detailed
        traces.

    Returns:
      An EvaluationBatch object containing scores, outputs, and trajectories for
      each task in the batch.
    """
    del capture_traces  # Not used.
    env = batch[0]['env']
    task_ids = [inst['task_id'] for inst in batch]
    tau_bench_run_config = RunConfig(
        env=env,
        model=self._agent_model,
        model_provider=self._agent_model_provider,
        user_model=self._user_model,
        user_model_provider=self._user_model_provider,
        agent_strategy=self._agent_strategy,
        user_strategy=self._user_strategy,
        max_concurrency=self._max_concurrency,
        task_ids=task_ids,
        log_dir=self._log_dir,
        task_split=batch[0]['task_split'],
    )
    tau_bench_results = run_tau_bench_rollouts(
        tau_bench_run_config,
        system_instruction=candidate.get(self._system_instruction_name),
        rater=self._rater,
    )

    outputs = []
    trajectories = []
    scores = []
    for res in tau_bench_results:
      outputs.append(
          TauBenchRolloutOutput(
              env=env,
              task_id=res.task_id,
              reward=res.reward,
              task_info=res.info,
          )
      )
      result_traj = res.traj
      trajectories.append(TauBenchTrajectory(result_traj=result_traj))
      scores.append(res.reward)

    return EvaluationBatch(
        scores=scores, outputs=outputs, trajectories=trajectories
    )

  def make_reflective_dataset(
      self,
      candidate: dict[str, str],
      eval_batch: EvaluationBatch[TauBenchTrajectory, TauBenchRolloutOutput],
      components_to_update: list[str],
  ) -> dict[str, list[dict[str, Any]]]:
    """Creates a dataset for reflection based on evaluation results.

    This method transforms the trajectories and scores from an evaluation run
    into a structured format that a reflection model can use to generate
    suggestions for improving the prompt.

    Args:
      candidate: The candidate that was evaluated.
      eval_batch: The results of the evaluation.
      components_to_update: A list of component names that the reflection should
        focus on improving.

    Returns:
      A dictionary where keys are component names and values are lists of
      data instances for reflection.
    """
    system_instruction = candidate[self._system_instruction_name]

    env = get_env(
        self._env_name,
        user_strategy=self._user_strategy,
        user_model=self._user_model,
        user_provider=self._user_model_provider,
        task_split='train',
    )

    tool_definitions = json.dumps(
        env.tools_info,
        indent=2,
        default=str,
    )
    inputs = '\n\n'.join([
        f'# System Instruction\n{system_instruction}',
        f'# Tool Definitions\n{tool_definitions}',
    ])
    ret_d: dict[str, list[dict[str, Any]]] = {}
    for comp in components_to_update:
      items: list[dict[str, Any]] = []
      trace_instances = list(
          zip(
              eval_batch.trajectories,
              eval_batch.scores,
              eval_batch.outputs,
              strict=True,
          )
      )
      for trace_instance in trace_instances:
        traj, _, rollout = trace_instance
        messages = traj['result_traj']
        # Remove instructions.
        if len(messages) > 1:
          messages = messages[1:]
        d = {
            'Inputs': inputs,
            'Generated Outputs': json.dumps(messages, indent=2, default=str),
            'Feedback': json.dumps(
                rollout['task_info']['feedback'], indent=2, default=str
            ),
        }
        items.append(d)
      if items:
        ret_d[comp] = items
    assert ret_d, (
        'empty reflective dataset for components '
        f'{[comp for comp in components_to_update]}'
    )
    return ret_d


_DATASET_SPLITS = {
    'train': tasks_train.TASKS_TRAIN,
    'dev': tasks_dev.TASKS_DEV,
    'test': tasks_test.TASKS_TEST,
}


def _get_dataset(ds: Dataset) -> list[TauBenchDataInst]:
  task_ids = ds.indexes or list(range(len(_DATASET_SPLITS[ds.split])))
  if ds.max_size is not None:
    task_ids = task_ids[: ds.max_size]
  random.shuffle(task_ids)
  return task_ids


def _get_datasets(
    config: ExperimentConfig,
) -> dict[str, list[int]]:
  """Returns Tau-bench dataset splits."""
  random.seed(config.rnd_seed)
  train_task_ids = _get_dataset(config.feedback_dataset)
  eval_task_ids = _get_dataset(config.pareto_dataset)
  test_task_ids = _get_dataset(config.eval_dataset)
  logging.info(
      'Using datasets of size: train=%d, eval=%d, test=%d',
      len(train_task_ids),
      len(eval_task_ids),
      len(test_task_ids),
  )
  return dict(
      train=train_task_ids,
      dev=eval_task_ids,
      test=test_task_ids,
  )


SEED_SYSTEM_INSTRUCTION = (
    'you are a customer support agent helping customers resolve their '
    'issues by using the right tools'
)


@dataclasses.dataclass(frozen=True)
class Dataset:

  split: str
  indexes: list[int] | None = None
  max_size: int = None


@dataclasses.dataclass
class ExperimentConfig:
  """Configures a GEPA experiment on Tau-bench."""

  tau_bench_env: str
  agent_model: str
  agent_model_provider: str
  user_model: str
  user_model_provider: str
  max_concurrency: int
  num_eval_trials: int
  rnd_seed: int
  max_metric_calls: int
  reflection_model: str
  reflection_minibatch_size: int
  use_rater: bool
  feedback_dataset: Dataset
  pareto_dataset: Dataset
  eval_dataset: Dataset


def _rater(config: ExperimentConfig) -> rater_lib.Rater:
  env = get_env(
      config.tau_bench_env,
      user_strategy='llm',
      user_model=config.user_model,
      user_provider=config.user_model_provider,
      task_split='train',
  )
  return rater_lib.Rater(json.dumps(env.tools_info, indent=2))


def run_gepa(
    output_dir: str, seed_instructions: str, config: ExperimentConfig
) -> Any:
  """Runs the GEPA optimization loop to train a new system instruction.

  Args:
    output_dir: The directory to save experiment results and artifacts.
    seed_instructions: Agent instructions to initialize the agent with.
    config: The experiment configuration.

  Returns:
    The results of the GEPA optimization.
  """
  # This section sets up and runs the GEPA optimization experiment.
  # Here we define all the parameters for the tau-bench environment, the GEPA
  # optimization loop, and the models to be used.
  datasets = _get_datasets(config)
  training_set = [
      TauBenchDataInst(
          env=config.tau_bench_env,
          task_id=task_id,
          task_split=config.feedback_dataset.split,
      )
      for task_id in datasets['train']
  ]
  eval_set = [
      TauBenchDataInst(
          env=config.tau_bench_env,
          task_id=task_id,
          task_split=config.pareto_dataset.split,
      )
      for task_id in datasets['dev']
  ]
  system_instruction_name = 'system_instruction'

  tau_bench_adapter = TauBenchAdapter(
      env_name=config.tau_bench_env,
      agent_model=config.agent_model,
      agent_model_provider=config.agent_model_provider,
      user_model=config.user_model,
      user_model_provider=config.user_model_provider,
      agent_strategy='tool-calling',
      user_strategy='llm',
      system_instruction_name=system_instruction_name,
      max_concurrency=config.max_concurrency,
      rater=_rater(config) if config.use_rater else None,
      log_dir=os.path.join(output_dir, 'traces'),
  )

  gepa_results = gepa.optimize(
      seed_candidate={
          system_instruction_name: seed_instructions,
      },
      trainset=training_set,
      valset=eval_set,
      task_lm=None,  # this must be None when a custom adapter is used
      adapter=tau_bench_adapter,
      max_metric_calls=config.max_metric_calls,
      reflection_lm=utils.reflection_inference_fn(config.reflection_model),
      reflection_minibatch_size=config.reflection_minibatch_size,
      run_dir=output_dir,
  )
  json.dump(
      gepa_results.to_dict(),
      open(os.path.join(output_dir, 'results.json'), 'w'),
  )
  return gepa_results


def run_eval(output_dir: str, instructions: str, config: ExperimentConfig):
  """Runs evaluation on the test set using the given instructions.

  Args:
    output_dir: The directory to save evaluation results.
    instructions: The system instructions to evaluate.
    config: The experiment configuration.
  """
  eval_dataset = _get_dataset(config.eval_dataset)
  tau_bench_run_config = RunConfig(
      env=config.tau_bench_env,
      model=config.agent_model,
      model_provider=config.agent_model_provider,
      user_model=config.user_model,
      user_model_provider=config.user_model_provider,
      agent_strategy='tool-calling',
      user_strategy='llm',
      max_concurrency=config.max_concurrency,
      num_trials=config.num_eval_trials,
      task_ids=eval_dataset,
      log_dir=output_dir,
      task_split=config.eval_dataset.split,
  )
  with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
    f.write(instructions)

  json.dump(
      tau_bench_run_config.model_dump(),
      open(os.path.join(output_dir, 'run_config.json'), 'w'),
  )
  tau_bench_results = run_tau_bench_rollouts(
      tau_bench_run_config,
      system_instruction=instructions,
      rater=_rater(config) if config.use_rater else None,
  )
  total = len(tau_bench_results)
  numerator = sum(1 for res in tau_bench_results if res.reward == 1)
  print(
      f'average reward (total={total}): {numerator/total if total > 0 else 0}'
  )
  json.dump(
      dict(results=[r.model_dump() for r in tau_bench_results]),
      open(os.path.join(output_dir, 'results.json'), 'w'),
  )

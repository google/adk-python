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

"""Runs a GEPA experiment on Tau-Bench."""

from collections.abc import Sequence
import dataclasses
from datetime import datetime
import json
import logging
import os

from absl import app
from absl import flags
import experiment
from google.genai import types

_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, '')
_EVAL_SET_SIZE = flags.DEFINE_integer('eval_set_size', None, '')
_MAX_METRIC_CALLS = flags.DEFINE_integer('max_metric_calls', 500, '')
_NUM_TEST_RECORDS = flags.DEFINE_integer('num_test_records', None, '')
_NUM_EVAL_TRIALS = flags.DEFINE_integer('num_eval_trials', 4, '')
_MAX_CONCURRENCY = flags.DEFINE_integer('max_concurrency', 8, '')
_EVAL_MODE = flags.DEFINE_bool('eval_mode', False, '')
_USE_RATER = flags.DEFINE_bool('use_rater', False, '')
_TRAIN_BATCH_SIZE = flags.DEFINE_integer('train_batch_size', 3, '')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Get a list of all existing loggers
  # logging.root.manager.loggerDict contains all named loggers
  # logging.getLogger(name) retrieves the logger object
  loggers = [
      logging.getLogger(name) for name in logging.root.manager.loggerDict
  ]

  # Iterate through the loggers and set their level to WARNING
  for logger in loggers:
    logger.setLevel(logging.WARNING)

  types.logger.addFilter(experiment.FilterInferenceWarnings())
  if not _OUTPUT_DIR.value:
    raise ValueError('outptut dir must be specified')
  output_dir = os.path.join(
      _OUTPUT_DIR.value, datetime.now().strftime('%Y%m%d%H%M%S%f')
  )
  os.makedirs(output_dir)
  logging.info('Writing to output_dir=%s', output_dir)
  config = experiment.ExperimentConfig(
      tau_bench_env='retail',
      agent_model='gemini-2.5-flash',
      agent_model_provider='vertex_ai',
      user_model='gemini-2.5-flash',
      user_model_provider='vertex_ai',
      max_concurrency=_MAX_CONCURRENCY.value,
      num_eval_trials=_NUM_EVAL_TRIALS.value,
      rnd_seed=42,
      max_metric_calls=_MAX_METRIC_CALLS.value,
      reflection_model='gemini-2.5-pro',
      reflection_minibatch_size=_TRAIN_BATCH_SIZE.value,
      use_rater=_USE_RATER.value,
      feedback_dataset=experiment.Dataset(split='train'),
      pareto_dataset=experiment.Dataset(
          split='dev', max_size=_EVAL_SET_SIZE.value
      ),
      eval_dataset=experiment.Dataset(
          split='test', max_size=_NUM_TEST_RECORDS.value
      ),
  )
  json.dump(
      dataclasses.asdict(config),
      open(os.path.join(output_dir, 'config.json'), 'w'),
  )
  logging.info('Using config=%s', config)

  if _EVAL_MODE.value:
    return experiment.run_eval(
        output_dir=output_dir,
        instructions=experiment.SEED_SYSTEM_INSTRUCTION,
        config=config,
    )

  results = experiment.run_gepa(
      config=config,
      seed_instructions=experiment.SEED_SYSTEM_INSTRUCTION,
      output_dir=output_dir,
  )
  print(list(enumerate(results.val_aggregate_scores)))

  eval_dir = os.path.join(
      output_dir, 'evals', datetime.now().strftime('%Y%m%d%H%M%S%f')
  )
  os.makedirs(eval_dir)
  experiment.run_eval(
      output_dir=eval_dir,
      instructions=results.best_candidate['system_instruction'],
      config=config,
  )


if __name__ == '__main__':
  app.run(main)

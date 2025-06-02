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

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__file__)


def _walk_to_root_until_found(folder, filename) -> str:
  checkpath = os.path.join(folder, filename)
  if os.path.exists(checkpath) and os.path.isfile(checkpath):
    return checkpath

  parent_folder = os.path.dirname(folder)
  if parent_folder == folder:  # reached the root
    return ''

  return _walk_to_root_until_found(parent_folder, filename)


def load_dotenv_for_agent(
    agent_name: str, agent_parent_folder: str, filename: str = '.env'
):
  """Loads the .env file for the agent module."""

  # Gets the folder of agent_module as starting_folder
  starting_folder = os.path.abspath(
      os.path.join(agent_parent_folder, agent_name)
  )
  dotenv_file_path = _walk_to_root_until_found(starting_folder, filename)
  if dotenv_file_path:
    load_dotenv(dotenv_file_path, override=True, verbose=True)
    logger.info(
        'Loaded %s file for %s at %s',
        filename,
        agent_name,
        dotenv_file_path,
    )
  else:
    logger.info('No %s file found for %s', filename, agent_name)

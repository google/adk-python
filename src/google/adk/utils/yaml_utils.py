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

from pathlib import Path
from typing import Union

from pydantic import BaseModel
import yaml


def dump_pydantic_to_yaml(
    model: BaseModel,
    file_path: Union[str, Path],
    *,
    indent: int = 2,
    sort_keys: bool = True,
    exclude_none: bool = True,
) -> None:
  """Dump a Pydantic model to a YAML file with multiline strings using | style.

  Args:
    model: The Pydantic model instance to dump.
    file_path: Path to the output YAML file.
    indent: Number of spaces for indentation (default: 2).
    sort_keys: Whether to sort dictionary keys (default: True).
    exclude_none: Exclude fields with None values (default: True).
  """
  model_dict = model.model_dump(exclude_none=exclude_none, mode='json')

  file_path = Path(file_path)
  file_path.parent.mkdir(parents=True, exist_ok=True)

  # Create a custom dumper class
  class _MultilineDumper(yaml.SafeDumper):
    pass

  def multiline_str_representer(dumper, data):
    if '\n' in data:
      return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

  # Add representer only to our custom dumper
  _MultilineDumper.add_representer(str, multiline_str_representer)

  with file_path.open('w', encoding='utf-8') as f:
    yaml.dump(
        model_dict,
        f,
        Dumper=_MultilineDumper,
        indent=indent,
        sort_keys=sort_keys,
        default_flow_style=False,
    )

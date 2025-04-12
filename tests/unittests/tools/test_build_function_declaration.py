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

from typing import Dict
from typing import List

from google.adk.tools import _automatic_function_calling_util
from google.adk.tools.agent_tool import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
# TODO: crewai requires python 3.10 as minimum
# from crewai_tools import FileReadTool
from langchain_community.tools import ShellTool
from pydantic import BaseModel
import pytest


def test_unsupported_variant():
  def simple_function(input_str: str) -> str:
    return {'result': input_str}

  with pytest.raises(ValueError):
    _automatic_function_calling_util.build_function_declaration(
        func=simple_function, variant='Unsupported'
    )


def test_string_input():
  def simple_function(input_str: str) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'STRING'


def test_int_input():
  def simple_function(input_str: int) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'INTEGER'


def test_float_input():
  def simple_function(input_str: float) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'NUMBER'


def test_bool_input():
  def simple_function(input_str: bool) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'BOOLEAN'


def test_array_input():
  def simple_function(input_str: List[str]) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'ARRAY'


def test_dict_input():
  def simple_function(input_str: Dict[str, str]) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'OBJECT'


def test_basemodel_input():
  class CustomInput(BaseModel):
    input_str: str

  def simple_function(input: CustomInput) -> str:
    return {'result': input}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input'].type == 'OBJECT'
  assert (
      function_decl.parameters.properties['input'].properties['input_str'].type
      == 'STRING'
  )


def test_toolcontext_ignored():
  def simple_function(input_str: str, tool_context: ToolContext) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function, ignore_params=['tool_context']
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'STRING'
  assert 'tool_context' not in function_decl.parameters.properties


def test_basemodel():
  class SimpleFunction(BaseModel):
    input_str: str
    custom_input: int

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=SimpleFunction, ignore_params=['custom_input']
  )

  assert function_decl.name == 'SimpleFunction'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'STRING'
  assert 'custom_input' not in function_decl.parameters.properties


def test_nested_basemodel_input():
  class ChildInput(BaseModel):
    input_str: str

  class CustomInput(BaseModel):
    child: ChildInput

  def simple_function(input: CustomInput) -> str:
    return {'result': input}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input'].type == 'OBJECT'
  assert (
      function_decl.parameters.properties['input'].properties['child'].type
      == 'OBJECT'
  )
  assert (
      function_decl.parameters.properties['input']
      .properties['child']
      .properties['input_str']
      .type
      == 'STRING'
  )


def test_basemodel_with_nested_basemodel():
  class ChildInput(BaseModel):
    input_str: str

  class CustomInput(BaseModel):
    child: ChildInput

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=CustomInput, ignore_params=['custom_input']
  )

  assert function_decl.name == 'CustomInput'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['child'].type == 'OBJECT'
  assert (
      function_decl.parameters.properties['child'].properties['input_str'].type
      == 'STRING'
  )
  assert 'custom_input' not in function_decl.parameters.properties


def test_list():
  def simple_function(
      input_str: List[str], input_dir: List[Dict[str, str]]
  ) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'ARRAY'
  assert function_decl.parameters.properties['input_str'].items.type == 'STRING'
  assert function_decl.parameters.properties['input_dir'].type == 'ARRAY'
  assert function_decl.parameters.properties['input_dir'].items.type == 'OBJECT'


def test_basemodel_list():
  class ChildInput(BaseModel):
    input_str: str

  class CustomInput(BaseModel):
    child: ChildInput

  def simple_function(input_str: List[CustomInput]) -> str:
    return {'result': input_str}

  function_decl = _automatic_function_calling_util.build_function_declaration(
      func=simple_function
  )

  assert function_decl.name == 'simple_function'
  assert function_decl.parameters.type == 'OBJECT'
  assert function_decl.parameters.properties['input_str'].type == 'ARRAY'
  assert function_decl.parameters.properties['input_str'].items.type == 'OBJECT'
  assert (
      function_decl.parameters.properties['input_str']
      .items.properties['child']
      .type
      == 'OBJECT'
  )
  assert (
      function_decl.parameters.properties['input_str']
      .items.properties['child']
      .properties['input_str']
      .type
      == 'STRING'
  )


# TODO: comment out this test for now as crewai requires python 3.10 as minimum
# def test_crewai_tool():
#   docs_tool = CrewaiTool(
#       name='direcotry_read_tool',
#       description='use this to find files for you.',
#       tool=FileReadTool(),
#   )
#   function_decl = docs_tool.get_declaration()
#   assert function_decl.name == 'direcotry_read_tool'
#   assert function_decl.parameters.type == 'OBJECT'
#   assert function_decl.parameters.properties['file_path'].type == 'STRING'

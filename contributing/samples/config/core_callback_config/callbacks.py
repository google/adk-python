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

from google.genai import types


async def before_agent_callback(callback_context):
  print('@before_agent_callback')
  return None


async def after_agent_callback(callback_context):
  print('@after_agent_callback')
  return None


async def before_model_callback(callback_context, llm_request):
  print('@before_model_callback')
  return None


async def after_model_callback(callback_context, llm_response):
  print('@after_model_callback')
  return None


def after_agent_callback1(callback_context):
  print('@after_agent_callback1')


def after_agent_callback2(callback_context):
  print('@after_agent_callback2')
  # ModelContent (or Content with role set to 'model') must be returned.
  # Otherwise, the event will be excluded from the context in the next turn.
  return types.ModelContent(
      parts=[
          types.Part(
              text='(stopped) after_agent_callback2',
          ),
      ],
  )


def after_agent_callback3(callback_context):
  print('@after_agent_callback3')


def before_agent_callback1(callback_context):
  print('@before_agent_callback1')


def before_agent_callback2(callback_context):
  print('@before_agent_callback2')


def before_agent_callback3(callback_context):
  print('@before_agent_callback3')


def before_tool_callback1(tool, args, tool_context):
  print('@before_tool_callback1')


def before_tool_callback2(tool, args, tool_context):
  print('@before_tool_callback2')


def before_tool_callback3(tool, args, tool_context):
  print('@before_tool_callback3')


def after_tool_callback1(tool, args, tool_context, tool_response):
  print('@after_tool_callback1')


def after_tool_callback2(tool, args, tool_context, tool_response):
  print('@after_tool_callback2')
  return {'test': 'after_tool_callback2', 'response': tool_response}


def after_tool_callback3(tool, args, tool_context, tool_response):
  print('@after_tool_callback3')

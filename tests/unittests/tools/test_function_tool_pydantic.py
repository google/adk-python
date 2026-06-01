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

# Pydantic model conversion tests

from typing import Optional
from typing import Union
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
import pydantic
import pytest


class UserModel(pydantic.BaseModel):
  """Test Pydantic model for user data."""

  name: str
  age: int
  email: Optional[str] = None


class PreferencesModel(pydantic.BaseModel):
  """Test Pydantic model for preferences."""

  theme: str = "light"
  notifications: bool = True


class CompanyModel(pydantic.BaseModel):
  """Test Pydantic model for company data."""

  company_name: str
  industry: str
  employee_count: int


def sync_function_with_pydantic_model(user: UserModel) -> dict:
  """Sync function that takes a Pydantic model."""
  return {
      "name": user.name,
      "age": user.age,
      "email": user.email,
      "type": str(type(user).__name__),
  }


async def async_function_with_pydantic_model(user: UserModel) -> dict:
  """Async function that takes a Pydantic model."""
  return {
      "name": user.name,
      "age": user.age,
      "email": user.email,
      "type": str(type(user).__name__),
  }


def function_with_optional_pydantic_model(
    user: UserModel, preferences: Optional[PreferencesModel] = None
) -> dict:
  """Function with required and optional Pydantic models."""
  result = {
      "user_name": user.name,
      "user_type": str(type(user).__name__),
  }
  if preferences:
    result.update({
        "theme": preferences.theme,
        "notifications": preferences.notifications,
        "preferences_type": str(type(preferences).__name__),
    })
  return result


def function_with_mixed_args(
    name: str, user: UserModel, count: int = 5
) -> dict:
  """Function with mixed argument types including Pydantic model."""
  return {
      "name": name,
      "user_name": user.name,
      "user_type": str(type(user).__name__),
      "count": count,
  }


def test_preprocess_args_with_dict_to_pydantic_conversion():
  """Test _preprocess_args converts dict to Pydantic model."""
  tool = FunctionTool(sync_function_with_pydantic_model)

  input_args = {
      "user": {"name": "Alice", "age": 30, "email": "alice@example.com"}
  }

  processed_args = tool._preprocess_args(input_args)

  # Check that the dict was converted to a Pydantic model
  assert "user" in processed_args
  user = processed_args["user"]
  assert isinstance(user, UserModel)
  assert user.name == "Alice"
  assert user.age == 30
  assert user.email == "alice@example.com"


def test_preprocess_args_with_existing_pydantic_model():
  """Test _preprocess_args leaves existing Pydantic model unchanged."""
  tool = FunctionTool(sync_function_with_pydantic_model)

  # Create an existing Pydantic model
  existing_user = UserModel(name="Bob", age=25)
  input_args = {"user": existing_user}

  processed_args = tool._preprocess_args(input_args)

  # Check that the existing model was not changed (same object)
  assert "user" in processed_args
  user = processed_args["user"]
  assert user is existing_user
  assert isinstance(user, UserModel)
  assert user.name == "Bob"


def test_preprocess_args_with_optional_pydantic_model_none():
  """Test _preprocess_args handles None for optional Pydantic models."""
  tool = FunctionTool(function_with_optional_pydantic_model)

  input_args = {"user": {"name": "Charlie", "age": 35}, "preferences": None}

  processed_args = tool._preprocess_args(input_args)

  # Check user conversion
  assert isinstance(processed_args["user"], UserModel)
  assert processed_args["user"].name == "Charlie"

  # Check preferences remains None
  assert processed_args["preferences"] is None


def test_preprocess_args_with_optional_pydantic_model_dict():
  """Test _preprocess_args converts dict for optional Pydantic models."""
  tool = FunctionTool(function_with_optional_pydantic_model)

  input_args = {
      "user": {"name": "Diana", "age": 28},
      "preferences": {"theme": "dark", "notifications": False},
  }

  processed_args = tool._preprocess_args(input_args)

  # Check both conversions
  assert isinstance(processed_args["user"], UserModel)
  assert processed_args["user"].name == "Diana"

  assert isinstance(processed_args["preferences"], PreferencesModel)
  assert processed_args["preferences"].theme == "dark"
  assert processed_args["preferences"].notifications is False


def test_preprocess_args_with_mixed_types():
  """Test _preprocess_args handles mixed argument types correctly."""
  tool = FunctionTool(function_with_mixed_args)

  input_args = {
      "name": "test_name",
      "user": {"name": "Eve", "age": 40},
      "count": 10,
  }

  processed_args = tool._preprocess_args(input_args)

  # Check that only Pydantic model was converted
  assert processed_args["name"] == "test_name"  # string unchanged
  assert processed_args["count"] == 10  # int unchanged

  # Check Pydantic model conversion
  assert isinstance(processed_args["user"], UserModel)
  assert processed_args["user"].name == "Eve"
  assert processed_args["user"].age == 40


def test_preprocess_args_with_invalid_data_graceful_failure():
  """Test _preprocess_args handles invalid data gracefully."""
  tool = FunctionTool(sync_function_with_pydantic_model)

  # Invalid data that can't be converted to UserModel
  input_args = {"user": "invalid_string"}  # string instead of dict/model

  processed_args = tool._preprocess_args(input_args)

  # Should keep original value when conversion fails
  assert processed_args["user"] == "invalid_string"


def test_preprocess_args_with_non_pydantic_parameters():
  """Test _preprocess_args ignores non-Pydantic parameters."""

  def simple_function(name: str, age: int) -> dict:
    return {"name": name, "age": age}

  tool = FunctionTool(simple_function)

  input_args = {"name": "test", "age": 25}
  processed_args = tool._preprocess_args(input_args)

  # Should remain unchanged (no Pydantic models to convert)
  assert processed_args == input_args


@pytest.mark.asyncio
async def test_run_async_with_pydantic_model_conversion_sync_function():
  """Test run_async with Pydantic model conversion for sync function."""
  tool = FunctionTool(sync_function_with_pydantic_model)

  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock

  args = {"user": {"name": "Frank", "age": 45, "email": "frank@example.com"}}

  result = await tool.run_async(args=args, tool_context=tool_context_mock)

  # Verify the function received a proper Pydantic model
  assert result["name"] == "Frank"
  assert result["age"] == 45
  assert result["email"] == "frank@example.com"
  assert result["type"] == "UserModel"


@pytest.mark.asyncio
async def test_run_async_with_pydantic_model_conversion_async_function():
  """Test run_async with Pydantic model conversion for async function."""
  tool = FunctionTool(async_function_with_pydantic_model)

  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock

  args = {"user": {"name": "Grace", "age": 32}}

  result = await tool.run_async(args=args, tool_context=tool_context_mock)

  # Verify the function received a proper Pydantic model
  assert result["name"] == "Grace"
  assert result["age"] == 32
  assert result["email"] is None  # default value
  assert result["type"] == "UserModel"


@pytest.mark.asyncio
async def test_run_async_with_optional_pydantic_models():
  """Test run_async with optional Pydantic models."""
  tool = FunctionTool(function_with_optional_pydantic_model)

  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock

  # Test with both required and optional models
  args = {
      "user": {"name": "Henry", "age": 50},
      "preferences": {"theme": "dark", "notifications": True},
  }

  result = await tool.run_async(args=args, tool_context=tool_context_mock)

  assert result["user_name"] == "Henry"
  assert result["user_type"] == "UserModel"
  assert result["theme"] == "dark"
  assert result["notifications"] is True
  assert result["preferences_type"] == "PreferencesModel"


def test_preprocess_args_with_list_of_pydantic_models():
  """Test _preprocess_args converts list of dicts to list of Pydantic models."""

  def function_with_list(users: list[UserModel]) -> int:
    return sum(u.age for u in users)

  tool = FunctionTool(function_with_list)

  input_args = {
      "users": [
          {"name": "Alice", "age": 30},
          {"name": "Bob", "age": 25},
      ]
  }

  processed_args = tool._preprocess_args(input_args)

  assert isinstance(processed_args["users"], list)
  assert len(processed_args["users"]) == 2
  assert all(isinstance(u, UserModel) for u in processed_args["users"])
  assert processed_args["users"][0].name == "Alice"
  assert processed_args["users"][1].age == 25


def test_preprocess_args_with_list_of_pydantic_models_already_converted():
  """Test _preprocess_args leaves existing Pydantic model instances in list."""

  def function_with_list(users: list[UserModel]) -> int:
    return sum(u.age for u in users)

  tool = FunctionTool(function_with_list)

  existing = [UserModel(name="Alice", age=30)]
  input_args = {"users": existing}

  processed_args = tool._preprocess_args(input_args)

  assert processed_args["users"][0] is existing[0]


def test_preprocess_args_with_list_of_primitives_unchanged():
  """Test _preprocess_args leaves list of primitives unchanged."""

  def function_with_list(names: list[str], counts: list[int]) -> int:
    return len(names) + sum(counts)

  tool = FunctionTool(function_with_list)

  input_args = {"names": ["Alice", "Bob"], "counts": [1, 2, 3]}
  processed_args = tool._preprocess_args(input_args)

  assert processed_args["names"] == ["Alice", "Bob"]
  assert processed_args["counts"] == [1, 2, 3]


def test_preprocess_args_with_list_of_pydantic_models_empty():
  """Test _preprocess_args handles empty list for list[BaseModel]."""

  def function_with_list(users: list[UserModel]) -> int:
    return 0

  tool = FunctionTool(function_with_list)

  processed_args = tool._preprocess_args({"users": []})

  assert processed_args["users"] == []


@pytest.mark.asyncio
async def test_run_async_with_list_of_pydantic_models():
  """Test run_async end-to-end with list[BaseModel] conversion."""

  def place_order(orders: list[UserModel]) -> int:
    return sum(u.age for u in orders)

  tool = FunctionTool(place_order)

  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock

  args = {"orders": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 20}]}

  result = await tool.run_async(args=args, tool_context=tool_context_mock)

  assert result == 50


def _function_with_union_of_basemodels(
    entity: Union[UserModel, CompanyModel],
) -> str:
  return type(entity).__name__


def test_preprocess_args_with_union_of_basemodels_picks_user():
  """Dict matching UserModel is converted to UserModel."""
  tool = FunctionTool(_function_with_union_of_basemodels)

  processed_args = tool._preprocess_args(
      {"entity": {"name": "Diana", "age": 32, "email": "d@example.com"}}
  )

  assert isinstance(processed_args["entity"], UserModel)
  assert processed_args["entity"].name == "Diana"


def test_preprocess_args_with_union_of_basemodels_picks_company():
  """Dict matching CompanyModel is converted to CompanyModel."""
  tool = FunctionTool(_function_with_union_of_basemodels)

  processed_args = tool._preprocess_args({
      "entity": {
          "company_name": "Acme Corp",
          "industry": "tech",
          "employee_count": 50,
      }
  })

  assert isinstance(processed_args["entity"], CompanyModel)
  assert processed_args["entity"].company_name == "Acme Corp"


def test_preprocess_args_with_union_of_basemodels_existing_instance_unchanged():
  """Existing instance of any union member is left unchanged."""
  tool = FunctionTool(_function_with_union_of_basemodels)

  user = UserModel(name="Bob", age=25)
  assert tool._preprocess_args({"entity": user})["entity"] is user

  company = CompanyModel(
      company_name="Acme", industry="tech", employee_count=10
  )
  assert tool._preprocess_args({"entity": company})["entity"] is company


def test_preprocess_args_with_union_of_basemodels_unrelated_instance_passthrough():
  """A BaseModel instance not in the union is not silently accepted."""
  tool = FunctionTool(_function_with_union_of_basemodels)

  class UnrelatedModel(pydantic.BaseModel):
    name: str
    age: int

  unrelated = UnrelatedModel(name="Carol", age=20)
  processed_args = tool._preprocess_args({"entity": unrelated})

  # Conversion fails (UnrelatedModel is not in the union); value is left
  # alone so the function receives it and raises a clear error itself.
  assert processed_args["entity"] is unrelated


def test_preprocess_args_with_optional_union_of_basemodels_none():
  """Optional[Union[A, B]] passes None through unchanged."""

  def fn(entity: Optional[Union[UserModel, CompanyModel]] = None) -> str:
    return type(entity).__name__

  tool = FunctionTool(fn)

  processed_args = tool._preprocess_args({"entity": None})

  assert processed_args["entity"] is None


def test_preprocess_args_with_optional_union_of_basemodels_dict():
  """Optional[Union[A, B]] converts a dict to the matching model."""

  def fn(entity: Optional[Union[UserModel, CompanyModel]] = None) -> str:
    return type(entity).__name__

  tool = FunctionTool(fn)

  processed_args = tool._preprocess_args({"entity": {"name": "Eve", "age": 40}})

  assert isinstance(processed_args["entity"], UserModel)
  assert processed_args["entity"].name == "Eve"


def test_preprocess_args_with_union_of_basemodels_invalid_data():
  """Invalid data for Union[BaseModel, BaseModel] is kept unchanged."""
  tool = FunctionTool(_function_with_union_of_basemodels)

  # Dict matches neither model.
  processed_args = tool._preprocess_args(
      {"entity": {"unrelated_field": "value"}}
  )

  assert processed_args["entity"] == {"unrelated_field": "value"}


@pytest.mark.asyncio
async def test_run_async_with_union_of_basemodels():
  """run_async end-to-end converts dict to the matching union member."""

  def create_entity_profile(
      entity: Union[UserModel, CompanyModel],
  ) -> dict:
    if isinstance(entity, UserModel):
      return {"entity_type": "user", "name": entity.name}
    if isinstance(entity, CompanyModel):
      return {"entity_type": "company", "name": entity.company_name}
    return {"entity_type": "unknown"}

  tool = FunctionTool(create_entity_profile)

  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock

  user_result = await tool.run_async(
      args={"entity": {"name": "Diana", "age": 32}},
      tool_context=tool_context_mock,
  )
  assert user_result == {"entity_type": "user", "name": "Diana"}

  company_result = await tool.run_async(
      args={
          "entity": {
              "company_name": "Acme Corp",
              "industry": "tech",
              "employee_count": 50,
          }
      },
      tool_context=tool_context_mock,
  )
  assert company_result == {"entity_type": "company", "name": "Acme Corp"}

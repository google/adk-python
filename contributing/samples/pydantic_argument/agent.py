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

"""Simple agent demonstrating Pydantic model arguments in tools."""

from typing import Optional
from typing import Union

from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
import pydantic


class UserProfile(pydantic.BaseModel):
  """A user's profile information."""

  name: str
  age: int
  email: Optional[str] = None


class UserPreferences(pydantic.BaseModel):
  """A user's preferences."""

  theme: str = "light"
  language: str = "English"
  notifications_enabled: bool = True


class CompanyProfile(pydantic.BaseModel):
  """A company's profile information."""

  company_name: str
  industry: str
  employee_count: int
  website: Optional[str] = None


def create_full_user_account(
    profile: UserProfile, preferences: Optional[UserPreferences] = None
) -> dict:
  """Create a complete user account with profile and optional preferences.

  This function demonstrates Union/Optional Pydantic model handling.
  The preferences parameter is Optional[UserPreferences], which is
  internally Union[UserPreferences, None].

  Before the fix, we would need:
  if preferences is not None and not isinstance(preferences, UserPreferences):
      preferences = UserPreferences.model_validate(preferences)

  Now the FunctionTool automatically handles this conversion!

  Args:
      profile: The user's profile information (required)
      preferences: Optional user preferences (Union[UserPreferences, None])

  Returns:
      A dictionary containing the complete user account.
  """
  # Use default preferences if not provided
  if preferences is None:
    preferences = UserPreferences()

  # Both profile and preferences are guaranteed to be proper Pydantic instances!
  return {
      "status": "account_created",
      "message": f"Full account created for {profile.name}!",
      "profile": {
          "name": profile.name,
          "age": profile.age,
          "email": profile.email or "Not provided",
          "profile_type": type(profile).__name__,
      },
      "preferences": {
          "theme": preferences.theme,
          "language": preferences.language,
          "notifications_enabled": preferences.notifications_enabled,
          "preferences_type": type(preferences).__name__,
      },
      "conversion_demo": {
          "profile_converted": "JSON dict → UserProfile instance",
          "preferences_converted": (
              "JSON dict → UserPreferences instance"
              if preferences
              else "None → default UserPreferences"
          ),
      },
  }


def create_entity_profile(entity: Union[UserProfile, CompanyProfile]) -> dict:
  """Create a profile for either a user or a company.

  This function demonstrates Union type handling with multiple Pydantic models.
  The entity parameter accepts Union[UserProfile, CompanyProfile].

  Before the fix, we would need complex type checking:
  if isinstance(entity, dict):
      # Try to determine which model to use and convert manually
      if 'company_name' in entity:
          entity = CompanyProfile.model_validate(entity)
      elif 'name' in entity:
          entity = UserProfile.model_validate(entity)
      else:
          raise ValueError("Cannot determine entity type")

  Now the FunctionTool automatically handles Union type conversion!
  The LLM will send the appropriate JSON structure, and it gets converted
  to the correct Pydantic model based on the JSON schema matching.

  Args:
      entity: Either a UserProfile or CompanyProfile (Union type)

  Returns:
      A dictionary containing the entity profile information.
  """
  if isinstance(entity, UserProfile):
    return {
        "status": "user_profile_created",
        "entity_type": "user",
        "message": f"User profile created for {entity.name}!",
        "profile": {
            "name": entity.name,
            "age": entity.age,
            "email": entity.email or "Not provided",
            "model_type": type(entity).__name__,
        },
    }
  elif isinstance(entity, CompanyProfile):
    return {
        "status": "company_profile_created",
        "entity_type": "company",
        "message": f"Company profile created for {entity.company_name}!",
        "profile": {
            "company_name": entity.company_name,
            "industry": entity.industry,
            "employee_count": entity.employee_count,
            "website": entity.website or "Not provided",
            "model_type": type(entity).__name__,
        },
    }
  else:
    return {
        "status": "error",
        "message": f"Unexpected entity type: {type(entity)}",
    }


# Create the agent with all Pydantic tools
root_agent = Agent(
    model="gemini-2.5-pro",
    name="profile_agent",
    description=(
        "Helpful assistant that helps creating accounts and profiles for users"
        " and companies"
    ),
    instruction="""
You are a helpful assistant that can create accounts and profiles for users and companies.

When someone asks you to create a user account, use `create_full_user_account`.
When someone asks you to create a profile and it's unclear whether they mean a user or company, use `create_entity_profile`.
When someone specifically mentions a company, use `create_entity_profile`.

Use the tools with the structured data provided by the user.
""",
    tools=[
        FunctionTool(create_full_user_account),
        FunctionTool(create_entity_profile),
    ],
)

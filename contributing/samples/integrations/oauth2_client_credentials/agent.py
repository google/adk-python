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

"""Weather Assistant Agent.

This agent provides weather information for cities worldwide.
It demonstrates OAuth2 client credentials flow transparently
through AuthenticatedFunctionTool usage.
"""

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowClientCredentials
from fastapi.openapi.models import OAuthFlows
from google.adk.agents.llm_agent import Agent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
import requests


# OAuth2 configuration for weather API access
def create_auth_config() -> AuthConfig:
  """Create OAuth2 auth configuration for weather API."""

  # Define OAuth2 scheme with client credentials flow
  flows = OAuthFlows(
      clientCredentials=OAuthFlowClientCredentials(
          tokenUrl="http://localhost:8080/token",
          scopes={
              "read": "Read access to weather data",
              "write": "Write access for data updates",
              "admin": "Administrative access",
          },
      )
  )
  auth_scheme = OAuth2(flows=flows)

  # Create credential with client ID and secret
  raw_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test_client",
          client_secret="test_secret",
      ),
  )

  return AuthConfig(
      auth_scheme=auth_scheme,
      raw_auth_credential=raw_credential,
      credential_key="weather_api_client",
  )


def get_weather_data(city: str = "San Francisco", credential=None) -> str:
  """Get current weather data for a specified city.

  Args:
      city: City name to get weather for
      credential: API credential (automatically injected by AuthenticatedFunctionTool)

  Returns:
      Current weather information for the city.
  """

  try:
    # Use the credential to make authenticated requests to weather API
    headers = {}
    if credential and credential.oauth2 and credential.oauth2.access_token:
      headers["Authorization"] = f"Bearer {credential.oauth2.access_token}"

    # Call weather API endpoint
    params = {"city": city, "units": "metric"}
    response = requests.get(
        "http://localhost:8080/api/weather",
        headers=headers,
        params=params,
        timeout=10,
    )

    if response.status_code == 200:
      data = response.json()
      result = f"🌤️ Weather for {city}:\n"
      result += f"Temperature: {data.get('temperature', 'N/A')}°C\n"
      result += f"Condition: {data.get('condition', 'N/A')}\n"
      result += f"Humidity: {data.get('humidity', 'N/A')}%\n"
      result += f"Wind Speed: {data.get('wind_speed', 'N/A')} km/h\n"
      result += f"Last Updated: {data.get('timestamp', 'N/A')}\n"
      return result
    else:
      return (
          f"❌ Failed to get weather data: {response.status_code} -"
          f" {response.text}"
      )

  except Exception as e:
    return f"❌ Error getting weather data: {str(e)}"


# Create the weather assistant agent
root_agent = Agent(
    name="WeatherAssistant",
    description=(
        "Weather assistant that provides current weather information for cities"
        " worldwide."
    ),
    model="gemini-2.5-pro",
    instruction=(
        "You are a helpful Weather Assistant that provides current weather"
        " information for any city worldwide.\n\nWhen users ask for weather:\n•"
        " Ask for the city name if not provided\n• Provide temperature in"
        " Celsius\n• Include helpful details like humidity, wind speed, and"
        " conditions\n• Be friendly and conversational about the weather\n\nIf"
        " there are any issues getting weather data, apologize and suggest"
        " trying again or checking for a different city name."
    ),
    tools=[
        AuthenticatedFunctionTool(
            func=get_weather_data, auth_config=create_auth_config()
        ),
    ],
)

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

from unittest.mock import MagicMock

from google.adk.tools.google_api_tool.googleapi_to_openapi_converter import GoogleApiToOpenApiConverter
# Import the converter class
from googleapiclient.errors import HttpError
import pytest


@pytest.fixture
def calendar_api_spec():
  """Fixture that provides a mock Google Calendar API spec for testing."""
  return {
      "kind": "discovery#restDescription",
      "id": "calendar:v3",
      "name": "calendar",
      "version": "v3",
      "title": "Google Calendar API",
      "description": "Accesses the Google Calendar API",
      "documentationLink": "https://developers.google.com/calendar/",
      "protocol": "rest",
      "rootUrl": "https://www.googleapis.com/",
      "servicePath": "calendar/v3/",
      "auth": {
          "oauth2": {
              "scopes": {
                  "https://www.googleapis.com/auth/calendar": {
                      "description": "Full access to Google Calendar"
                  },
                  "https://www.googleapis.com/auth/calendar.readonly": {
                      "description": "Read-only access to Google Calendar"
                  },
              }
          }
      },
      "schemas": {
          "Calendar": {
              "type": "object",
              "description": "A calendar resource",
              "properties": {
                  "id": {
                      "type": "string",
                      "description": "Calendar identifier",
                  },
                  "summary": {
                      "type": "string",
                      "description": "Calendar summary",
                      "required": True,
                  },
                  "timeZone": {
                      "type": "string",
                      "description": "Calendar timezone",
                  },
              },
          },
          "Event": {
              "type": "object",
              "description": "An event resource",
              "properties": {
                  "id": {"type": "string", "description": "Event identifier"},
                  "summary": {"type": "string", "description": "Event summary"},
                  "start": {"$ref": "EventDateTime"},
                  "end": {"$ref": "EventDateTime"},
                  "attendees": {
                      "type": "array",
                      "description": "Event attendees",
                      "items": {"$ref": "EventAttendee"},
                  },
              },
          },
          "EventDateTime": {
              "type": "object",
              "description": "Date/time for an event",
              "properties": {
                  "dateTime": {
                      "type": "string",
                      "format": "date-time",
                      "description": "Date/time in RFC3339 format",
                  },
                  "timeZone": {
                      "type": "string",
                      "description": "Timezone for the date/time",
                  },
              },
          },
          "EventAttendee": {
              "type": "object",
              "description": "An attendee of an event",
              "properties": {
                  "email": {"type": "string", "description": "Attendee email"},
                  "responseStatus": {
                      "type": "string",
                      "description": "Response status",
                      "enum": [
                          "needsAction",
                          "declined",
                          "tentative",
                          "accepted",
                      ],
                  },
              },
          },
      },
      "resources": {
          "calendars": {
              "methods": {
                  "get": {
                      "id": "calendar.calendars.get",
                      "flatPath": "calendars/{calendarId}",
                      "httpMethod": "GET",
                      "description": "Returns metadata for a calendar.",
                      "parameters": {
                          "calendarId": {
                              "type": "string",
                              "description": "Calendar identifier",
                              "required": True,
                              "location": "path",
                          }
                      },
                      "response": {"$ref": "Calendar"},
                      "scopes": [
                          "https://www.googleapis.com/auth/calendar",
                          "https://www.googleapis.com/auth/calendar.readonly",
                      ],
                  },
                  "insert": {
                      "id": "calendar.calendars.insert",
                      "path": "calendars",
                      "httpMethod": "POST",
                      "description": "Creates a secondary calendar.",
                      "request": {"$ref": "Calendar"},
                      "response": {"$ref": "Calendar"},
                      "scopes": ["https://www.googleapis.com/auth/calendar"],
                  },
              },
              "resources": {
                  "events": {
                      "methods": {
                          "list": {
                              "id": "calendar.events.list",
                              "flatPath": "calendars/{calendarId}/events",
                              "httpMethod": "GET",
                              "description": (
                                  "Returns events on the specified calendar."
                              ),
                              "parameters": {
                                  "calendarId": {
                                      "type": "string",
                                      "description": "Calendar identifier",
                                      "required": True,
                                      "location": "path",
                                  },
                                  "maxResults": {
                                      "type": "integer",
                                      "description": (
                                          "Maximum number of events returned"
                                      ),
                                      "format": "int32",
                                      "minimum": "1",
                                      "maximum": "2500",
                                      "default": "250",
                                      "location": "query",
                                  },
                                  "orderBy": {
                                      "type": "string",
                                      "description": (
                                          "Order of the events returned"
                                      ),
                                      "enum": ["startTime", "updated"],
                                      "location": "query",
                                  },
                              },
                              "response": {"$ref": "Events"},
                              "scopes": [
                                  "https://www.googleapis.com/auth/calendar",
                                  "https://www.googleapis.com/auth/calendar.readonly",
                              ],
                          }
                      }
                  }
              },
          }
      },
  }


@pytest.fixture
def converter():
  """Fixture that provides a basic converter instance."""
  return GoogleApiToOpenApiConverter("calendar", "v3")


@pytest.fixture
def mock_api_resource(calendar_api_spec):
  """Fixture that provides a mock API resource with the test spec."""
  mock_resource = MagicMock()
  mock_resource._rootDesc = calendar_api_spec
  return mock_resource


@pytest.fixture
def prepared_converter(converter, calendar_api_spec):
  """Fixture that provides a converter with the API spec already set."""
  converter._google_api_spec = calendar_api_spec
  return converter


@pytest.fixture
def converter_with_patched_build(monkeypatch, mock_api_resource):
  """Fixture that provides a converter with the build function patched.

  This simulates a successful API spec fetch.
  """
  # Create a mock for the build function
  mock_build = MagicMock(return_value=mock_api_resource)

  # Patch the build function in the target module
  monkeypatch.setattr(
      "google.adk.tools.google_api_tool.googleapi_to_openapi_converter.build",
      mock_build,
  )

  # Create and return a converter instance
  return GoogleApiToOpenApiConverter("calendar", "v3")


class TestGoogleApiToOpenApiConverter:
  """Test suite for the GoogleApiToOpenApiConverter class."""

  def test_init(self, converter):
    """Test converter initialization."""
    assert converter._api_name == "calendar"
    assert converter._api_version == "v3"
    assert converter._google_api_resource is None
    assert converter._google_api_spec is None
    assert converter._openapi_spec["openapi"] == "3.0.0"
    assert "info" in converter._openapi_spec
    assert "paths" in converter._openapi_spec
    assert "components" in converter._openapi_spec

  def test_fetch_google_api_spec(
      self, converter_with_patched_build, calendar_api_spec
  ):
    """Test fetching Google API specification."""
    # Call the method
    converter_with_patched_build.fetch_google_api_spec()

    # Verify the results
    assert converter_with_patched_build._google_api_spec == calendar_api_spec

  def test_fetch_google_api_spec_error(self, monkeypatch, converter):
    """Test error handling when fetching Google API specification."""
    # Create a mock that raises an error
    mock_build = MagicMock(
        side_effect=HttpError(resp=MagicMock(status=404), content=b"Not Found")
    )
    monkeypatch.setattr(
        "google.adk.tools.google_api_tool.googleapi_to_openapi_converter.build",
        mock_build,
    )

    # Verify exception is raised
    with pytest.raises(HttpError):
      converter.fetch_google_api_spec()

  def test_convert_info(self, prepared_converter):
    """Test conversion of basic API information."""
    # Call the method
    prepared_converter._convert_info()

    # Verify the results
    info = prepared_converter._openapi_spec["info"]
    assert info["title"] == "Google Calendar API"
    assert info["description"] == "Accesses the Google Calendar API"
    assert info["version"] == "v3"
    assert info["termsOfService"] == "https://developers.google.com/calendar/"

    # Check external docs
    external_docs = prepared_converter._openapi_spec["externalDocs"]
    assert external_docs["url"] == "https://developers.google.com/calendar/"

  def test_convert_servers(self, prepared_converter):
    """Test conversion of server information."""
    # Call the method
    prepared_converter._convert_servers()

    # Verify the results
    servers = prepared_converter._openapi_spec["servers"]
    assert len(servers) == 1
    assert servers[0]["url"] == "https://www.googleapis.com/calendar/v3"
    assert servers[0]["description"] == "calendar v3 API"

  def test_convert_security_schemes(self, prepared_converter):
    """Test conversion of security schemes."""
    # Call the method
    prepared_converter._convert_security_schemes()

    # Verify the results
    security_schemes = prepared_converter._openapi_spec["components"][
        "securitySchemes"
    ]

    # Check OAuth2 configuration
    assert "oauth2" in security_schemes
    oauth2 = security_schemes["oauth2"]
    assert oauth2["type"] == "oauth2"

    # Check OAuth2 scopes
    scopes = oauth2["flows"]["authorizationCode"]["scopes"]
    assert "https://www.googleapis.com/auth/calendar" in scopes
    assert "https://www.googleapis.com/auth/calendar.readonly" in scopes

    # Check API key configuration
    assert "apiKey" in security_schemes
    assert security_schemes["apiKey"]["type"] == "apiKey"
    assert security_schemes["apiKey"]["in"] == "query"
    assert security_schemes["apiKey"]["name"] == "key"

  def test_convert_schemas(self, prepared_converter):
    """Test conversion of schema definitions."""
    # Call the method
    prepared_converter._convert_schemas()

    # Verify the results
    schemas = prepared_converter._openapi_spec["components"]["schemas"]

    # Check Calendar schema
    assert "Calendar" in schemas
    calendar_schema = schemas["Calendar"]
    assert calendar_schema["type"] == "object"
    assert calendar_schema["description"] == "A calendar resource"

    # Check required properties
    assert "required" in calendar_schema
    assert "summary" in calendar_schema["required"]

    # Check Event schema references
    assert "Event" in schemas
    event_schema = schemas["Event"]
    assert (
        event_schema["properties"]["start"]["$ref"]
        == "#/components/schemas/EventDateTime"
    )

    # Check array type with references
    attendees_schema = event_schema["properties"]["attendees"]
    assert attendees_schema["type"] == "array"
    assert (
        attendees_schema["items"]["$ref"]
        == "#/components/schemas/EventAttendee"
    )

    # Check enum values
    attendee_schema = schemas["EventAttendee"]
    response_status = attendee_schema["properties"]["responseStatus"]
    assert "enum" in response_status
    assert "accepted" in response_status["enum"]

  @pytest.mark.parametrize(
      "schema_def, expected_type, expected_attrs",
      [
          # Test object type
          (
              {
                  "type": "object",
                  "description": "Test object",
                  "properties": {
                      "id": {"type": "string", "required": True},
                      "name": {"type": "string"},
                  },
              },
              "object",
              {"description": "Test object", "required": ["id"]},
          ),
          # Test array type
          (
              {
                  "type": "array",
                  "description": "Test array",
                  "items": {"type": "string"},
              },
              "array",
              {"description": "Test array", "items": {"type": "string"}},
          ),
          # Test reference conversion
          (
              {"$ref": "Calendar"},
              None,  # No type for references
              {"$ref": "#/components/schemas/Calendar"},
          ),
          # Test enum conversion
          (
              {"type": "string", "enum": ["value1", "value2"]},
              "string",
              {"enum": ["value1", "value2"]},
          ),
      ],
  )
  def test_convert_schema_object(
      self, converter, schema_def, expected_type, expected_attrs
  ):
    """Test conversion of individual schema objects with different input variations."""
    converted = converter._convert_schema_object(schema_def)

    # Check type if expected
    if expected_type:
      assert converted["type"] == expected_type

    # Check other expected attributes
    for key, value in expected_attrs.items():
      assert converted[key] == value

  @pytest.mark.parametrize(
      "path, expected_params",
      [
          # Path with parameters
          (
              "/calendars/{calendarId}/events/{eventId}",
              ["calendarId", "eventId"],
          ),
          # Path without parameters
          ("/calendars/events", []),
          # Mixed path
          ("/users/{userId}/calendars/default", ["userId"]),
      ],
  )
  def test_extract_path_parameters(self, converter, path, expected_params):
    """Test extraction of path parameters from URL path with various inputs."""
    params = converter._extract_path_parameters(path)
    assert set(params) == set(expected_params)
    assert len(params) == len(expected_params)

  @pytest.mark.parametrize(
      "param_data, expected_result",
      [
          # String parameter
          (
              {
                  "type": "string",
                  "description": "String parameter",
                  "pattern": "^[a-z]+$",
              },
              {"type": "string", "pattern": "^[a-z]+$"},
          ),
          # Integer parameter with format
          (
              {"type": "integer", "format": "int32", "default": "10"},
              {"type": "integer", "format": "int32", "default": "10"},
          ),
          # Enum parameter
          (
              {"type": "string", "enum": ["option1", "option2"]},
              {"type": "string", "enum": ["option1", "option2"]},
          ),
      ],
  )
  def test_convert_parameter_schema(
      self, converter, param_data, expected_result
  ):
    """Test conversion of parameter definitions to OpenAPI schemas."""
    converted = converter._convert_parameter_schema(param_data)

    # Check all expected attributes
    for key, value in expected_result.items():
      assert converted[key] == value

  def test_convert(self, converter_with_patched_build):
    """Test the complete conversion process."""
    # Call the method
    result = converter_with_patched_build.convert()

    # Verify basic structure
    assert result["openapi"] == "3.0.0"
    assert "info" in result
    assert "servers" in result
    assert "paths" in result
    assert "components" in result

    # Verify paths
    paths = result["paths"]
    assert "/calendars/{calendarId}" in paths
    assert "get" in paths["/calendars/{calendarId}"]

    # Verify nested resources
    assert "/calendars/{calendarId}/events" in paths

    # Verify method details
    get_calendar = paths["/calendars/{calendarId}"]["get"]
    assert get_calendar["operationId"] == "calendar.calendars.get"
    assert "parameters" in get_calendar

    # Verify request body
    insert_calendar = paths["/calendars"]["post"]
    assert "requestBody" in insert_calendar
    request_schema = insert_calendar["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert request_schema["$ref"] == "#/components/schemas/Calendar"

    # Verify response body
    assert "responses" in get_calendar
    response_schema = get_calendar["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert response_schema["$ref"] == "#/components/schemas/Calendar"

  def test_convert_methods(self, prepared_converter, calendar_api_spec):
    """Test conversion of API methods."""
    # Convert methods
    methods = calendar_api_spec["resources"]["calendars"]["methods"]
    prepared_converter._convert_methods(methods, "/calendars")

    # Verify the results
    paths = prepared_converter._openapi_spec["paths"]

    # Check GET method
    assert "/calendars/{calendarId}" in paths
    get_method = paths["/calendars/{calendarId}"]["get"]
    assert get_method["operationId"] == "calendar.calendars.get"

    # Check parameters
    params = get_method["parameters"]
    param_names = [p["name"] for p in params]
    assert "calendarId" in param_names

    # Check POST method
    assert "/calendars" in paths
    post_method = paths["/calendars"]["post"]
    assert post_method["operationId"] == "calendar.calendars.insert"

    # Check request body
    assert "requestBody" in post_method
    assert (
        post_method["requestBody"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/Calendar"
    )

    # Check response
    assert (
        post_method["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        == "#/components/schemas/Calendar"
    )

  def test_convert_resources(self, prepared_converter, calendar_api_spec):
    """Test conversion of nested resources."""
    # Convert resources
    resources = calendar_api_spec["resources"]
    prepared_converter._convert_resources(resources)

    # Verify the results
    paths = prepared_converter._openapi_spec["paths"]

    # Check top-level resource methods
    assert "/calendars/{calendarId}" in paths

    # Check nested resource methods
    assert "/calendars/{calendarId}/events" in paths
    events_method = paths["/calendars/{calendarId}/events"]["get"]
    assert events_method["operationId"] == "calendar.events.list"

    # Check parameters in nested resource
    params = events_method["parameters"]
    param_names = [p["name"] for p in params]
    assert "calendarId" in param_names
    assert "maxResults" in param_names
    assert "orderBy" in param_names

  def test_integration_calendar_api(self, converter_with_patched_build):
    """Integration test using Calendar API specification."""
    # Create and run the converter
    openapi_spec = converter_with_patched_build.convert()

    # Verify conversion results
    assert openapi_spec["info"]["title"] == "Google Calendar API"
    assert (
        openapi_spec["servers"][0]["url"]
        == "https://www.googleapis.com/calendar/v3"
    )

    # Check security schemes
    security_schemes = openapi_spec["components"]["securitySchemes"]
    assert "oauth2" in security_schemes
    assert "apiKey" in security_schemes

    # Check schemas
    schemas = openapi_spec["components"]["schemas"]
    assert "Calendar" in schemas
    assert "Event" in schemas
    assert "EventDateTime" in schemas

    # Check paths
    paths = openapi_spec["paths"]
    assert "/calendars/{calendarId}" in paths
    assert "/calendars" in paths
    assert "/calendars/{calendarId}/events" in paths

    # Check method details
    get_events = paths["/calendars/{calendarId}/events"]["get"]
    assert get_events["operationId"] == "calendar.events.list"

    # Check parameter details
    param_dict = {p["name"]: p for p in get_events["parameters"]}
    assert "maxResults" in param_dict
    max_results = param_dict["maxResults"]
    assert max_results["in"] == "query"
    assert max_results["schema"]["type"] == "integer"
    assert max_results["schema"]["default"] == "250"


@pytest.fixture
def conftest_content():
  """Returns content for a conftest.py file to help with testing."""
  return """
import pytest
from unittest.mock import MagicMock

# This file contains fixtures that can be shared across multiple test modules

@pytest.fixture
def mock_google_response():
    \"\"\"Fixture that provides a mock response from Google's API.\"\"\"
    return {"key": "value", "items": [{"id": 1}, {"id": 2}]}

@pytest.fixture
def mock_http_error():
    \"\"\"Fixture that provides a mock HTTP error.\"\"\"
    mock_resp = MagicMock()
    mock_resp.status = 404
    return HttpError(resp=mock_resp, content=b'Not Found')
"""


def test_generate_conftest_example(conftest_content):
  """This is a meta-test that demonstrates how to generate a conftest.py file.

  In a real project, you would create a separate conftest.py file.
  """
  # In a real scenario, you would write this to a file named conftest.py
  # This test just verifies the conftest content is not empty
  assert len(conftest_content) > 0

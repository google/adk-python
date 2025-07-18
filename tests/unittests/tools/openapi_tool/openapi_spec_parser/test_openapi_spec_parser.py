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

from typing import Any
from typing import Dict

from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_spec_parser import OpenApiSpecParser
import pytest


def create_minimal_openapi_spec() -> Dict[str, Any]:
  """Creates a minimal valid OpenAPI spec."""
  return {
      "openapi": "3.1.0",
      "info": {"title": "Minimal API", "version": "1.0.0"},
      "paths": {
          "/test": {
              "get": {
                  "summary": "Test GET endpoint",
                  "operationId": "testGet",
                  "responses": {
                      "200": {
                          "description": "Successful response",
                          "content": {
                              "application/json": {"schema": {"type": "string"}}
                          },
                      }
                  },
              }
          }
      },
  }


@pytest.fixture
def openapi_spec_generator():
  """Fixture for creating an OperationGenerator instance."""
  return OpenApiSpecParser()


def test_parse_minimal_spec(openapi_spec_generator):
  """Test parsing a minimal OpenAPI specification."""
  openapi_spec = create_minimal_openapi_spec()

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  op = parsed_operations[0]

  assert len(parsed_operations) == 1
  assert op.name == "test_get"
  assert op.endpoint.path == "/test"
  assert op.endpoint.method == "get"
  assert op.return_value.type_value == str


def test_parse_spec_with_no_operation_id(openapi_spec_generator):
  """Test parsing a spec where operationId is missing (auto-generation)."""
  openapi_spec = create_minimal_openapi_spec()
  del openapi_spec["paths"]["/test"]["get"]["operationId"]  # Remove operationId

  parsed_operations = openapi_spec_generator.parse(openapi_spec)

  assert len(parsed_operations) == 1
  # Check if operationId is auto generated based on path and method.
  assert parsed_operations[0].name == "test_get"


def test_parse_spec_with_multiple_methods(openapi_spec_generator):
  """Test parsing a spec with multiple HTTP methods for the same path."""
  openapi_spec = create_minimal_openapi_spec()
  openapi_spec["paths"]["/test"]["post"] = {
      "summary": "Test POST endpoint",
      "operationId": "testPost",
      "responses": {"200": {"description": "Successful response"}},
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  operation_names = {op.name for op in parsed_operations}

  assert len(parsed_operations) == 2
  assert "test_get" in operation_names
  assert "test_post" in operation_names


def test_parse_spec_with_parameters(openapi_spec_generator):
  openapi_spec = create_minimal_openapi_spec()
  openapi_spec["paths"]["/test"]["get"]["parameters"] = [
      {"name": "param1", "in": "query", "schema": {"type": "string"}},
      {"name": "param2", "in": "header", "schema": {"type": "integer"}},
  ]

  parsed_operations = openapi_spec_generator.parse(openapi_spec)

  assert len(parsed_operations[0].parameters) == 2
  assert parsed_operations[0].parameters[0].original_name == "param1"
  assert parsed_operations[0].parameters[0].param_location == "query"
  assert parsed_operations[0].parameters[1].original_name == "param2"
  assert parsed_operations[0].parameters[1].param_location == "header"


def test_parse_spec_with_request_body(openapi_spec_generator):
  openapi_spec = create_minimal_openapi_spec()
  openapi_spec["paths"]["/test"]["post"] = {
      "summary": "Endpoint with request body",
      "operationId": "testPostWithBody",
      "requestBody": {
          "content": {
              "application/json": {
                  "schema": {
                      "type": "object",
                      "properties": {"name": {"type": "string"}},
                  }
              }
          }
      },
      "responses": {"200": {"description": "OK"}},
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  post_operations = [
      op for op in parsed_operations if op.endpoint.method == "post"
  ]
  op = post_operations[0]

  assert len(post_operations) == 1
  assert op.name == "test_post_with_body"
  assert len(op.parameters) == 1
  assert op.parameters[0].original_name == "name"
  assert op.parameters[0].type_value == str


def test_parse_spec_with_reference(openapi_spec_generator):
  """Test parsing a specification with $ref."""
  openapi_spec = {
      "openapi": "3.1.0",
      "info": {"title": "API with Refs", "version": "1.0.0"},
      "paths": {
          "/test_ref": {
              "get": {
                  "summary": "Endpoint with ref",
                  "operationId": "testGetRef",
                  "responses": {
                      "200": {
                          "description": "Success",
                          "content": {
                              "application/json": {
                                  "schema": {
                                      "$ref": "#/components/schemas/MySchema"
                                  }
                              }
                          },
                      }
                  },
              }
          }
      },
      "components": {
          "schemas": {
              "MySchema": {
                  "type": "object",
                  "properties": {"name": {"type": "string"}},
              }
          }
      },
  }
  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  op = parsed_operations[0]

  assert len(parsed_operations) == 1
  assert op.return_value.type_value.__origin__ is dict


def test_parse_spec_with_circular_reference(openapi_spec_generator):
  """Test correct handling of circular $ref (important!)."""
  openapi_spec = {
      "openapi": "3.1.0",
      "info": {"title": "Circular Ref API", "version": "1.0.0"},
      "paths": {
          "/circular": {
              "get": {
                  "responses": {
                      "200": {
                          "description": "OK",
                          "content": {
                              "application/json": {
                                  "schema": {"$ref": "#/components/schemas/A"}
                              }
                          },
                      }
                  }
              }
          }
      },
      "components": {
          "schemas": {
              "A": {
                  "type": "object",
                  "properties": {"b": {"$ref": "#/components/schemas/B"}},
              },
              "B": {
                  "type": "object",
                  "properties": {"a": {"$ref": "#/components/schemas/A"}},
              },
          }
      },
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  assert len(parsed_operations) == 1

  op = parsed_operations[0]
  assert op.return_value.type_value.__origin__ is dict
  assert op.return_value.type_hint == "Dict[str, Any]"


def test_parse_no_paths(openapi_spec_generator):
  """Test with a spec that has no paths defined."""
  openapi_spec = {
      "openapi": "3.1.0",
      "info": {"title": "No Paths API", "version": "1.0.0"},
  }
  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  assert len(parsed_operations) == 0  # Should be empty


def test_parse_empty_path_item(openapi_spec_generator):
  """Test a path item that is present but empty."""
  openapi_spec = {
      "openapi": "3.1.0",
      "info": {"title": "Empty Path Item API", "version": "1.0.0"},
      "paths": {"/empty": None},
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)

  assert len(parsed_operations) == 0


def test_parse_spec_with_global_auth_scheme(openapi_spec_generator):
  """Test parsing with a global security scheme."""
  openapi_spec = create_minimal_openapi_spec()
  openapi_spec["security"] = [{"api_key": []}]
  openapi_spec["components"] = {
      "securitySchemes": {
          "api_key": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
      }
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  op = parsed_operations[0]

  assert len(parsed_operations) == 1
  assert op.auth_scheme is not None
  assert op.auth_scheme.type_.value == "apiKey"


def test_parse_spec_with_local_auth_scheme(openapi_spec_generator):
  """Test parsing with a local (operation-level) security scheme."""
  openapi_spec = create_minimal_openapi_spec()
  openapi_spec["paths"]["/test"]["get"]["security"] = [{"local_auth": []}]
  openapi_spec["components"] = {
      "securitySchemes": {"local_auth": {"type": "http", "scheme": "bearer"}}
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  op = parsed_operations[0]

  assert op.auth_scheme is not None
  assert op.auth_scheme.type_.value == "http"
  assert op.auth_scheme.scheme == "bearer"


def test_parse_spec_with_servers(openapi_spec_generator):
  """Test parsing with server URLs."""
  openapi_spec = create_minimal_openapi_spec()
  openapi_spec["servers"] = [
      {"url": "https://api.example.com"},
      {"url": "http://localhost:8000"},
  ]

  parsed_operations = openapi_spec_generator.parse(openapi_spec)

  assert len(parsed_operations) == 1
  assert parsed_operations[0].endpoint.base_url == "https://api.example.com"


def test_parse_spec_with_no_servers(openapi_spec_generator):
  """Test with no servers defined (should default to empty string)."""
  openapi_spec = create_minimal_openapi_spec()
  if "servers" in openapi_spec:
    del openapi_spec["servers"]

  parsed_operations = openapi_spec_generator.parse(openapi_spec)

  assert len(parsed_operations) == 1
  assert parsed_operations[0].endpoint.base_url == ""


def test_parse_spec_with_description(openapi_spec_generator):
  openapi_spec = create_minimal_openapi_spec()
  expected_description = "This is a test description."
  openapi_spec["paths"]["/test"]["get"]["description"] = expected_description

  parsed_operations = openapi_spec_generator.parse(openapi_spec)

  assert len(parsed_operations) == 1
  assert parsed_operations[0].description == expected_description


def test_parse_spec_with_empty_description(openapi_spec_generator):
  openapi_spec = create_minimal_openapi_spec()
  openapi_spec["paths"]["/test"]["get"]["description"] = ""
  openapi_spec["paths"]["/test"]["get"]["summary"] = ""

  parsed_operations = openapi_spec_generator.parse(openapi_spec)

  assert len(parsed_operations) == 1
  assert parsed_operations[0].description == ""


def test_parse_spec_with_no_description(openapi_spec_generator):
  openapi_spec = create_minimal_openapi_spec()

  # delete description
  if "description" in openapi_spec["paths"]["/test"]["get"]:
    del openapi_spec["paths"]["/test"]["get"]["description"]
  if "summary" in openapi_spec["paths"]["/test"]["get"]:
    del openapi_spec["paths"]["/test"]["get"]["summary"]

  parsed_operations = openapi_spec_generator.parse(openapi_spec)

  assert len(parsed_operations) == 1
  assert (
      parsed_operations[0].description == ""
  )  # it should be initialized with empty string


def test_parse_invalid_openapi_spec_type(openapi_spec_generator):
  """Test that passing a non-dict object to parse raises TypeError"""
  with pytest.raises(AttributeError):
    openapi_spec_generator.parse(123)  # type: ignore

  with pytest.raises(AttributeError):
    openapi_spec_generator.parse("openapi_spec")  # type: ignore

  with pytest.raises(AttributeError):
    openapi_spec_generator.parse([])  # type: ignore


def test_parse_external_ref_raises_error(openapi_spec_generator):
  """Check that external references (not starting with #) raise ValueError."""
  openapi_spec = {
      "openapi": "3.1.0",
      "info": {"title": "External Ref API", "version": "1.0.0"},
      "paths": {
          "/external": {
              "get": {
                  "responses": {
                      "200": {
                          "description": "OK",
                          "content": {
                              "application/json": {
                                  "schema": {
                                      "$ref": "external_file.json#/components/schemas/ExternalSchema"
                                  }
                              }
                          },
                      }
                  }
              }
          }
      },
  }
  with pytest.raises(ValueError):
    openapi_spec_generator.parse(openapi_spec)


def test_parse_spec_with_multiple_paths_deep_refs(openapi_spec_generator):
  """Test specs with multiple paths, request/response bodies using deep refs."""
  openapi_spec = {
      "openapi": "3.1.0",
      "info": {"title": "Multiple Paths Deep Refs API", "version": "1.0.0"},
      "paths": {
          "/path1": {
              "post": {
                  "operationId": "postPath1",
                  "requestBody": {
                      "content": {
                          "application/json": {
                              "schema": {
                                  "$ref": "#/components/schemas/Request1"
                              }
                          }
                      }
                  },
                  "responses": {
                      "200": {
                          "description": "OK",
                          "content": {
                              "application/json": {
                                  "schema": {
                                      "$ref": "#/components/schemas/Response1"
                                  }
                              }
                          },
                      }
                  },
              }
          },
          "/path2": {
              "put": {
                  "operationId": "putPath2",
                  "requestBody": {
                      "content": {
                          "application/json": {
                              "schema": {
                                  "$ref": "#/components/schemas/Request2"
                              }
                          }
                      }
                  },
                  "responses": {
                      "200": {
                          "description": "OK",
                          "content": {
                              "application/json": {
                                  "schema": {
                                      "$ref": "#/components/schemas/Response2"
                                  }
                              }
                          },
                      }
                  },
              },
              "get": {
                  "operationId": "getPath2",
                  "responses": {
                      "200": {
                          "description": "OK",
                          "content": {
                              "application/json": {
                                  "schema": {
                                      "$ref": "#/components/schemas/Response2"
                                  }
                              }
                          },
                      }
                  },
              },
          },
      },
      "components": {
          "schemas": {
              "Request1": {
                  "type": "object",
                  "properties": {
                      "req1_prop1": {"$ref": "#/components/schemas/Level1_1"}
                  },
              },
              "Response1": {
                  "type": "object",
                  "properties": {
                      "res1_prop1": {"$ref": "#/components/schemas/Level1_2"}
                  },
              },
              "Request2": {
                  "type": "object",
                  "properties": {
                      "req2_prop1": {"$ref": "#/components/schemas/Level1_1"}
                  },
              },
              "Response2": {
                  "type": "object",
                  "properties": {
                      "res2_prop1": {"$ref": "#/components/schemas/Level1_2"}
                  },
              },
              "Level1_1": {
                  "type": "object",
                  "properties": {
                      "level1_1_prop1": {
                          "$ref": "#/components/schemas/Level2_1"
                      }
                  },
              },
              "Level1_2": {
                  "type": "object",
                  "properties": {
                      "level1_2_prop1": {
                          "$ref": "#/components/schemas/Level2_2"
                      }
                  },
              },
              "Level2_1": {
                  "type": "object",
                  "properties": {
                      "level2_1_prop1": {"$ref": "#/components/schemas/Level3"}
                  },
              },
              "Level2_2": {
                  "type": "object",
                  "properties": {"level2_2_prop1": {"type": "string"}},
              },
              "Level3": {"type": "integer"},
          }
      },
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  assert len(parsed_operations) == 3

  # Verify Path 1
  path1_ops = [op for op in parsed_operations if op.endpoint.path == "/path1"]
  assert len(path1_ops) == 1
  path1_op = path1_ops[0]
  assert path1_op.name == "post_path1"

  assert len(path1_op.parameters) == 1
  assert path1_op.parameters[0].original_name == "req1_prop1"
  assert (
      path1_op.parameters[0]
      .param_schema.properties["level1_1_prop1"]
      .properties["level2_1_prop1"]
      .type
      == "integer"
  )
  assert (
      path1_op.return_value.param_schema.properties["res1_prop1"]
      .properties["level1_2_prop1"]
      .properties["level2_2_prop1"]
      .type
      == "string"
  )

  # Verify Path 2
  path2_ops = [
      op
      for op in parsed_operations
      if op.endpoint.path == "/path2" and op.name == "put_path2"
  ]
  path2_op = path2_ops[0]
  assert path2_op is not None
  assert len(path2_op.parameters) == 1
  assert path2_op.parameters[0].original_name == "req2_prop1"
  assert (
      path2_op.parameters[0]
      .param_schema.properties["level1_1_prop1"]
      .properties["level2_1_prop1"]
      .type
      == "integer"
  )
  assert (
      path2_op.return_value.param_schema.properties["res2_prop1"]
      .properties["level1_2_prop1"]
      .properties["level2_2_prop1"]
      .type
      == "string"
  )


def test_parse_spec_with_duplicate_parameter_names(openapi_spec_generator):
  """Test handling of duplicate parameter names (one in query, one in body).

  The expected behavior is that both parameters should be captured but with
  different suffix, and
  their `original_name` attributes should reflect their origin (query or body).
  """
  openapi_spec = {
      "openapi": "3.1.0",
      "info": {"title": "Duplicate Parameter Names API", "version": "1.0.0"},
      "paths": {
          "/duplicate": {
              "post": {
                  "operationId": "createWithDuplicate",
                  "parameters": [{
                      "name": "name",
                      "in": "query",
                      "schema": {"type": "string"},
                  }],
                  "requestBody": {
                      "content": {
                          "application/json": {
                              "schema": {
                                  "type": "object",
                                  "properties": {"name": {"type": "integer"}},
                              }
                          }
                      }
                  },
                  "responses": {"200": {"description": "OK"}},
              }
          }
      },
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  assert len(parsed_operations) == 1
  op = parsed_operations[0]
  assert op.name == "create_with_duplicate"
  assert len(op.parameters) == 2

  query_param = None
  body_param = None
  for param in op.parameters:
    if param.param_location == "query" and param.original_name == "name":
      query_param = param
    elif param.param_location == "body" and param.original_name == "name":
      body_param = param

  assert query_param is not None
  assert query_param.original_name == "name"
  assert query_param.py_name == "name"

  assert body_param is not None
  assert body_param.original_name == "name"
  assert body_param.py_name == "name_0"


def test_parse_spec_with_path_level_parameters(openapi_spec_generator):
  """Test that operation parameters are correctly combined with path-level parameters."""
  openapi_spec = {
      "openapi": "3.1.0",
      "info": {"title": "Combine Parameters API", "version": "1.0.0"},
      "paths": {
          "/test": {
              "parameters": [{
                  "name": "global_param",
                  "in": "query",
                  "schema": {"type": "string"},
              }],
              "get": {
                  "parameters": [{
                      "name": "local_param",
                      "in": "header",
                      "schema": {"type": "integer"},
                  }],
                  "operationId": "testGet",
                  "responses": {
                      "200": {
                          "description": "Successful response",
                          "content": {
                              "application/json": {"schema": {"type": "string"}}
                          },
                      }
                  },
              },
          }
      },
  }

  parsed_operations = openapi_spec_generator.parse(openapi_spec)
  assert len(parsed_operations) == 1

  operation = parsed_operations[0]
  assert len(operation.parameters) == 2

  # Verify the combined parameters
  global_param = next(
      (p for p in operation.parameters if p.original_name == "global_param"),
      None,
  )
  local_param = next(
      (p for p in operation.parameters if p.original_name == "local_param"),
      None,
  )

  assert global_param is not None
  assert global_param.param_location == "query"
  assert global_param.type_value is str

  assert local_param is not None
  assert local_param.param_location == "header"
  assert local_param.type_value is int

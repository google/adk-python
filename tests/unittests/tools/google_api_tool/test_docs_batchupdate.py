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
from unittest.mock import patch

from google.adk.tools.google_api_tool.googleapi_to_openapi_converter import GoogleApiToOpenApiConverter
import pytest


@pytest.fixture
def docs_api_spec():
  """Fixture that provides a mock Google Docs API spec for testing."""
  return {
      "kind": "discovery#restDescription",
      "id": "docs:v1",
      "name": "docs",
      "version": "v1",
      "title": "Google Docs API",
      "description": "Reads and writes Google Docs documents.",
      "documentationLink": "https://developers.google.com/docs/",
      "protocol": "rest",
      "rootUrl": "https://docs.googleapis.com/",
      "servicePath": "",
      "auth": {
          "oauth2": {
              "scopes": {
                  "https://www.googleapis.com/auth/documents": {
                      "description": (
                          "See, edit, create, and delete all of your Google"
                          " Docs documents"
                      )
                  },
                  "https://www.googleapis.com/auth/documents.readonly": {
                      "description": "View your Google Docs documents"
                  },
                  "https://www.googleapis.com/auth/drive": {
                      "description": (
                          "See, edit, create, and delete all of your Google"
                          " Drive files"
                      )
                  },
                  "https://www.googleapis.com/auth/drive.file": {
                      "description": (
                          "View and manage Google Drive files and folders that"
                          " you have opened or created with this app"
                      )
                  },
              }
          }
      },
      "schemas": {
          "Document": {
              "type": "object",
              "description": "A Google Docs document",
              "properties": {
                  "documentId": {
                      "type": "string",
                      "description": "The ID of the document",
                  },
                  "title": {
                      "type": "string",
                      "description": "The title of the document",
                  },
                  "body": {"$ref": "Body", "description": "The document body"},
                  "revisionId": {
                      "type": "string",
                      "description": "The revision ID of the document",
                  },
              },
          },
          "Body": {
              "type": "object",
              "description": "The document body",
              "properties": {
                  "content": {
                      "type": "array",
                      "description": "The content of the body",
                      "items": {"$ref": "StructuralElement"},
                  }
              },
          },
          "StructuralElement": {
              "type": "object",
              "description": "A structural element of a document",
              "properties": {
                  "startIndex": {
                      "type": "integer",
                      "description": "The zero-based start index",
                  },
                  "endIndex": {
                      "type": "integer",
                      "description": "The zero-based end index",
                  },
              },
          },
          "BatchUpdateDocumentRequest": {
              "type": "object",
              "description": "Request to batch update a document",
              "properties": {
                  "requests": {
                      "type": "array",
                      "description": (
                          "A list of updates to apply to the document"
                      ),
                      "items": {"$ref": "Request"},
                  },
                  "writeControl": {
                      "$ref": "WriteControl",
                      "description": (
                          "Provides control over how write requests are"
                          " executed"
                      ),
                  },
              },
          },
          "Request": {
              "type": "object",
              "description": "A single kind of update to apply to a document",
              "properties": {
                  "insertText": {"$ref": "InsertTextRequest"},
                  "updateTextStyle": {"$ref": "UpdateTextStyleRequest"},
                  "replaceAllText": {"$ref": "ReplaceAllTextRequest"},
              },
          },
          "InsertTextRequest": {
              "type": "object",
              "description": "Inserts text into the document",
              "properties": {
                  "location": {
                      "$ref": "Location",
                      "description": "The location to insert text",
                  },
                  "text": {
                      "type": "string",
                      "description": "The text to insert",
                  },
              },
          },
          "UpdateTextStyleRequest": {
              "type": "object",
              "description": "Updates the text style of the specified range",
              "properties": {
                  "range": {
                      "$ref": "Range",
                      "description": "The range to update",
                  },
                  "textStyle": {
                      "$ref": "TextStyle",
                      "description": "The text style to apply",
                  },
                  "fields": {
                      "type": "string",
                      "description": "The fields that should be updated",
                  },
              },
          },
          "ReplaceAllTextRequest": {
              "type": "object",
              "description": "Replaces all instances of text matching criteria",
              "properties": {
                  "containsText": {"$ref": "SubstringMatchCriteria"},
                  "replaceText": {
                      "type": "string",
                      "description": (
                          "The text that will replace the matched text"
                      ),
                  },
              },
          },
          "Location": {
              "type": "object",
              "description": "A particular location in the document",
              "properties": {
                  "index": {
                      "type": "integer",
                      "description": "The zero-based index",
                  },
                  "tabId": {
                      "type": "string",
                      "description": "The tab the location is in",
                  },
              },
          },
          "Range": {
              "type": "object",
              "description": "Specifies a contiguous range of text",
              "properties": {
                  "startIndex": {
                      "type": "integer",
                      "description": "The zero-based start index",
                  },
                  "endIndex": {
                      "type": "integer",
                      "description": "The zero-based end index",
                  },
              },
          },
          "TextStyle": {
              "type": "object",
              "description": (
                  "Represents the styling that can be applied to text"
              ),
              "properties": {
                  "bold": {
                      "type": "boolean",
                      "description": "Whether or not the text is bold",
                  },
                  "italic": {
                      "type": "boolean",
                      "description": "Whether or not the text is italic",
                  },
                  "fontSize": {
                      "$ref": "Dimension",
                      "description": "The size of the text's font",
                  },
              },
          },
          "SubstringMatchCriteria": {
              "type": "object",
              "description": (
                  "A criteria that matches a specific string of text in the"
                  " document"
              ),
              "properties": {
                  "text": {
                      "type": "string",
                      "description": "The text to search for",
                  },
                  "matchCase": {
                      "type": "boolean",
                      "description": (
                          "Indicates whether the search should respect case"
                      ),
                  },
              },
          },
          "WriteControl": {
              "type": "object",
              "description": (
                  "Provides control over how write requests are executed"
              ),
              "properties": {
                  "requiredRevisionId": {
                      "type": "string",
                      "description": "The required revision ID",
                  },
                  "targetRevisionId": {
                      "type": "string",
                      "description": "The target revision ID",
                  },
              },
          },
          "BatchUpdateDocumentResponse": {
              "type": "object",
              "description": "Response from a BatchUpdateDocument request",
              "properties": {
                  "documentId": {
                      "type": "string",
                      "description": "The ID of the document",
                  },
                  "replies": {
                      "type": "array",
                      "description": "The reply of the updates",
                      "items": {"$ref": "Response"},
                  },
                  "writeControl": {
                      "$ref": "WriteControl",
                      "description": "The updated write control",
                  },
              },
          },
          "Response": {
              "type": "object",
              "description": "A single response from an update",
              "properties": {
                  "replaceAllText": {"$ref": "ReplaceAllTextResponse"},
              },
          },
          "ReplaceAllTextResponse": {
              "type": "object",
              "description": "The result of replacing text",
              "properties": {
                  "occurrencesChanged": {
                      "type": "integer",
                      "description": "The number of occurrences changed",
                  },
              },
          },
      },
      "resources": {
          "documents": {
              "methods": {
                  "get": {
                      "id": "docs.documents.get",
                      "path": "v1/documents/{documentId}",
                      "flatPath": "v1/documents/{documentId}",
                      "httpMethod": "GET",
                      "description": (
                          "Gets the latest version of the specified document."
                      ),
                      "parameters": {
                          "documentId": {
                              "type": "string",
                              "description": (
                                  "The ID of the document to retrieve"
                              ),
                              "required": True,
                              "location": "path",
                          }
                      },
                      "response": {"$ref": "Document"},
                      "scopes": [
                          "https://www.googleapis.com/auth/documents",
                          "https://www.googleapis.com/auth/documents.readonly",
                          "https://www.googleapis.com/auth/drive",
                          "https://www.googleapis.com/auth/drive.file",
                      ],
                  },
                  "create": {
                      "id": "docs.documents.create",
                      "path": "v1/documents",
                      "httpMethod": "POST",
                      "description": (
                          "Creates a blank document using the title given in"
                          " the request."
                      ),
                      "request": {"$ref": "Document"},
                      "response": {"$ref": "Document"},
                      "scopes": [
                          "https://www.googleapis.com/auth/documents",
                          "https://www.googleapis.com/auth/drive",
                          "https://www.googleapis.com/auth/drive.file",
                      ],
                  },
                  "batchUpdate": {
                      "id": "docs.documents.batchUpdate",
                      "path": "v1/documents/{documentId}:batchUpdate",
                      "flatPath": "v1/documents/{documentId}:batchUpdate",
                      "httpMethod": "POST",
                      "description": (
                          "Applies one or more updates to the document."
                      ),
                      "parameters": {
                          "documentId": {
                              "type": "string",
                              "description": "The ID of the document to update",
                              "required": True,
                              "location": "path",
                          }
                      },
                      "request": {"$ref": "BatchUpdateDocumentRequest"},
                      "response": {"$ref": "BatchUpdateDocumentResponse"},
                      "scopes": [
                          "https://www.googleapis.com/auth/documents",
                          "https://www.googleapis.com/auth/drive",
                          "https://www.googleapis.com/auth/drive.file",
                      ],
                  },
              },
          }
      },
  }


@pytest.fixture
def docs_converter():
  """Fixture that provides a basic docs converter instance."""
  return GoogleApiToOpenApiConverter("docs", "v1")


@pytest.fixture
def mock_docs_api_resource(docs_api_spec):
  """Fixture that provides a mock API resource with the docs test spec."""
  mock_resource = MagicMock()
  mock_resource._rootDesc = docs_api_spec
  return mock_resource


@pytest.fixture
def prepared_docs_converter(docs_converter, docs_api_spec):
  """Fixture that provides a converter with the Docs API spec already set."""
  docs_converter._google_api_spec = docs_api_spec
  return docs_converter


@pytest.fixture
def docs_converter_with_patched_build(monkeypatch, mock_docs_api_resource):
  """Fixture that provides a converter with the build function patched.

  This simulates a successful API spec fetch.
  """
  # Create a mock for the build function
  mock_build = MagicMock(return_value=mock_docs_api_resource)

  # Patch the build function in the target module
  monkeypatch.setattr(
      "google.adk.tools.google_api_tool.googleapi_to_openapi_converter.build",
      mock_build,
  )

  # Create and return a converter instance
  return GoogleApiToOpenApiConverter("docs", "v1")


class TestDocsApiBatchUpdate:
  """Test suite for the Google Docs API batchUpdate endpoint conversion."""

  def test_batch_update_method_conversion(
      self, prepared_docs_converter, docs_api_spec
  ):
    """Test conversion of the batchUpdate method specifically."""
    # Convert methods from the documents resource
    methods = docs_api_spec["resources"]["documents"]["methods"]
    prepared_docs_converter._convert_methods(methods, "/v1/documents")

    # Verify the results
    paths = prepared_docs_converter._openapi_spec["paths"]

    # Check that batchUpdate POST method exists
    assert "/v1/documents/{documentId}:batchUpdate" in paths
    batch_update_method = paths["/v1/documents/{documentId}:batchUpdate"][
        "post"
    ]

    # Verify method details
    assert batch_update_method["operationId"] == "docs.documents.batchUpdate"
    assert (
        batch_update_method["summary"]
        == "Applies one or more updates to the document."
    )

    # Check parameters exist
    params = batch_update_method["parameters"]
    param_names = [p["name"] for p in params]
    assert "documentId" in param_names

    # Check request body
    assert "requestBody" in batch_update_method
    request_body = batch_update_method["requestBody"]
    assert request_body["required"] is True
    request_schema = request_body["content"]["application/json"]["schema"]
    assert (
        request_schema["$ref"]
        == "#/components/schemas/BatchUpdateDocumentRequest"
    )

    # Check response
    assert "responses" in batch_update_method
    response_schema = batch_update_method["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert (
        response_schema["$ref"]
        == "#/components/schemas/BatchUpdateDocumentResponse"
    )

    # Check security/scopes
    assert "security" in batch_update_method
    # Should have OAuth2 scopes for documents access

  def test_batch_update_request_schema_conversion(
      self, prepared_docs_converter, docs_api_spec
  ):
    """Test that BatchUpdateDocumentRequest schema is properly converted."""
    # Convert schemas using the actual method signature
    prepared_docs_converter._convert_schemas()

    schemas = prepared_docs_converter._openapi_spec["components"]["schemas"]

    # Check BatchUpdateDocumentRequest schema
    assert "BatchUpdateDocumentRequest" in schemas
    batch_request_schema = schemas["BatchUpdateDocumentRequest"]

    assert batch_request_schema["type"] == "object"
    assert "properties" in batch_request_schema
    assert "requests" in batch_request_schema["properties"]
    assert "writeControl" in batch_request_schema["properties"]

    # Check requests array property
    requests_prop = batch_request_schema["properties"]["requests"]
    assert requests_prop["type"] == "array"
    assert requests_prop["items"]["$ref"] == "#/components/schemas/Request"

  def test_batch_update_response_schema_conversion(
      self, prepared_docs_converter, docs_api_spec
  ):
    """Test that BatchUpdateDocumentResponse schema is properly converted."""
    # Convert schemas using the actual method signature
    prepared_docs_converter._convert_schemas()

    schemas = prepared_docs_converter._openapi_spec["components"]["schemas"]

    # Check BatchUpdateDocumentResponse schema
    assert "BatchUpdateDocumentResponse" in schemas
    batch_response_schema = schemas["BatchUpdateDocumentResponse"]

    assert batch_response_schema["type"] == "object"
    assert "properties" in batch_response_schema
    assert "documentId" in batch_response_schema["properties"]
    assert "replies" in batch_response_schema["properties"]
    assert "writeControl" in batch_response_schema["properties"]

    # Check replies array property
    replies_prop = batch_response_schema["properties"]["replies"]
    assert replies_prop["type"] == "array"
    assert replies_prop["items"]["$ref"] == "#/components/schemas/Response"

  def test_batch_update_request_types_conversion(
      self, prepared_docs_converter, docs_api_spec
  ):
    """Test that various request types are properly converted."""
    # Convert schemas using the actual method signature
    prepared_docs_converter._convert_schemas()

    schemas = prepared_docs_converter._openapi_spec["components"]["schemas"]

    # Check Request schema (union of different request types)
    assert "Request" in schemas
    request_schema = schemas["Request"]
    assert "properties" in request_schema

    # Should contain different request types
    assert "insertText" in request_schema["properties"]
    assert "updateTextStyle" in request_schema["properties"]
    assert "replaceAllText" in request_schema["properties"]

    # Check InsertTextRequest
    assert "InsertTextRequest" in schemas
    insert_text_schema = schemas["InsertTextRequest"]
    assert "location" in insert_text_schema["properties"]
    assert "text" in insert_text_schema["properties"]

    # Check UpdateTextStyleRequest
    assert "UpdateTextStyleRequest" in schemas
    update_style_schema = schemas["UpdateTextStyleRequest"]
    assert "range" in update_style_schema["properties"]
    assert "textStyle" in update_style_schema["properties"]
    assert "fields" in update_style_schema["properties"]

  def test_convert_methods(self, prepared_docs_converter, docs_api_spec):
    """Test conversion of API methods."""
    # Convert methods
    methods = docs_api_spec["resources"]["documents"]["methods"]
    prepared_docs_converter._convert_methods(methods, "/v1/documents")

    # Verify the results
    paths = prepared_docs_converter._openapi_spec["paths"]

    # Check GET method
    assert "/v1/documents/{documentId}" in paths
    get_method = paths["/v1/documents/{documentId}"]["get"]
    assert get_method["operationId"] == "docs.documents.get"

    # Check parameters
    params = get_method["parameters"]
    param_names = [p["name"] for p in params]
    assert "documentId" in param_names

    # Check POST method (create)
    assert "/v1/documents" in paths
    post_method = paths["/v1/documents"]["post"]
    assert post_method["operationId"] == "docs.documents.create"

    # Check request body
    assert "requestBody" in post_method
    assert (
        post_method["requestBody"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        == "#/components/schemas/Document"
    )

    # Check response
    assert (
        post_method["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        == "#/components/schemas/Document"
    )

    # Check batchUpdate POST method
    assert "/v1/documents/{documentId}:batchUpdate" in paths
    batch_update_method = paths["/v1/documents/{documentId}:batchUpdate"][
        "post"
    ]
    assert batch_update_method["operationId"] == "docs.documents.batchUpdate"

  def test_complete_docs_api_conversion(
      self, docs_converter_with_patched_build
  ):
    """Integration test for complete Docs API conversion including batchUpdate."""
    # Call the method
    result = docs_converter_with_patched_build.convert()

    # Verify basic structure
    assert result["openapi"] == "3.0.0"
    assert "info" in result
    assert "servers" in result
    assert "paths" in result
    assert "components" in result

    # Verify paths
    paths = result["paths"]
    assert "/v1/documents/{documentId}" in paths
    assert "get" in paths["/v1/documents/{documentId}"]

    # Verify batchUpdate endpoint
    assert "/v1/documents/{documentId}:batchUpdate" in paths
    assert "post" in paths["/v1/documents/{documentId}:batchUpdate"]

    # Verify method details
    get_document = paths["/v1/documents/{documentId}"]["get"]
    assert get_document["operationId"] == "docs.documents.get"
    assert "parameters" in get_document

    # Verify batchUpdate method
    batch_update = paths["/v1/documents/{documentId}:batchUpdate"]["post"]
    assert batch_update["operationId"] == "docs.documents.batchUpdate"

    # Verify request body
    assert "requestBody" in batch_update
    request_schema = batch_update["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert (
        request_schema["$ref"]
        == "#/components/schemas/BatchUpdateDocumentRequest"
    )

    # Verify response body
    assert "responses" in batch_update
    response_schema = batch_update["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert (
        response_schema["$ref"]
        == "#/components/schemas/BatchUpdateDocumentResponse"
    )

    # Verify schemas exist
    schemas = result["components"]["schemas"]
    assert "Document" in schemas
    assert "BatchUpdateDocumentRequest" in schemas
    assert "BatchUpdateDocumentResponse" in schemas
    assert "InsertTextRequest" in schemas
    assert "UpdateTextStyleRequest" in schemas
    assert "ReplaceAllTextRequest" in schemas

  def test_batch_update_example_request_structure(
      self, prepared_docs_converter, docs_api_spec
  ):
    """Test that the converted schema can represent a realistic batchUpdate request."""
    # Convert schemas using the actual method signature
    prepared_docs_converter._convert_schemas()

    schemas = prepared_docs_converter._openapi_spec["components"]["schemas"]

    # Verify that we can represent a realistic batch update request like:
    # {
    #   "requests": [
    #     {
    #       "insertText": {
    #         "location": {"index": 1},
    #         "text": "Hello World"
    #       }
    #     },
    #     {
    #       "updateTextStyle": {
    #         "range": {"startIndex": 1, "endIndex": 6},
    #         "textStyle": {"bold": true},
    #         "fields": "bold"
    #       }
    #     }
    #   ],
    #   "writeControl": {
    #     "requiredRevisionId": "some-revision-id"
    #   }
    # }

    # Check that all required schemas exist for this structure
    assert "BatchUpdateDocumentRequest" in schemas
    assert "Request" in schemas
    assert "InsertTextRequest" in schemas
    assert "UpdateTextStyleRequest" in schemas
    assert "Location" in schemas
    assert "Range" in schemas
    assert "TextStyle" in schemas
    assert "WriteControl" in schemas

    # Verify Location schema has required properties
    location_schema = schemas["Location"]
    assert "index" in location_schema["properties"]
    assert location_schema["properties"]["index"]["type"] == "integer"

    # Verify Range schema has required properties
    range_schema = schemas["Range"]
    assert "startIndex" in range_schema["properties"]
    assert "endIndex" in range_schema["properties"]

    # Verify TextStyle schema has formatting properties
    text_style_schema = schemas["TextStyle"]
    assert "bold" in text_style_schema["properties"]
    assert text_style_schema["properties"]["bold"]["type"] == "boolean"

  def test_integration_docs_api(self, docs_converter_with_patched_build):
    """Integration test using Google Docs API specification."""
    # Create and run the converter
    openapi_spec = docs_converter_with_patched_build.convert()

    # Verify conversion results
    assert openapi_spec["info"]["title"] == "Google Docs API"
    assert openapi_spec["servers"][0]["url"] == "https://docs.googleapis.com"

    # Check security schemes
    security_schemes = openapi_spec["components"]["securitySchemes"]
    assert "oauth2" in security_schemes
    assert "apiKey" in security_schemes

    # Check schemas
    schemas = openapi_spec["components"]["schemas"]
    assert "Document" in schemas
    assert "BatchUpdateDocumentRequest" in schemas
    assert "BatchUpdateDocumentResponse" in schemas
    assert "InsertTextRequest" in schemas
    assert "UpdateTextStyleRequest" in schemas
    assert "ReplaceAllTextRequest" in schemas

    # Check paths
    paths = openapi_spec["paths"]
    assert "/v1/documents/{documentId}" in paths
    assert "/v1/documents" in paths
    assert "/v1/documents/{documentId}:batchUpdate" in paths

    # Check method details
    get_document = paths["/v1/documents/{documentId}"]["get"]
    assert get_document["operationId"] == "docs.documents.get"

    # Check batchUpdate method details
    batch_update = paths["/v1/documents/{documentId}:batchUpdate"]["post"]
    assert batch_update["operationId"] == "docs.documents.batchUpdate"

    # Check parameter details
    param_dict = {p["name"]: p for p in get_document["parameters"]}
    assert "documentId" in param_dict
    document_id = param_dict["documentId"]
    assert document_id["required"] is True
    assert document_id["schema"]["type"] == "string"

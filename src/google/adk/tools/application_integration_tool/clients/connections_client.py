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

import json
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import google.auth
from google.auth import default as default_service_credential
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import requests


class ConnectionsClient:
  """Utility class for interacting with Google Cloud Connectors API."""

  def __init__(
      self,
      project: str,
      location: str,
      connection: str,
      service_account_json: Optional[str] = None,
  ):
    """Initializes the ConnectionsClient.

    Args:
      project: The Google Cloud project ID.
      location: The Google Cloud location (e.g., us-central1).
      connection: The connection name.
      service_account_json: The service account configuration as a dictionary.
        Required if not using default service credential. Used for fetching
        connection details.
    """
    self.project = project
    self.location = location
    self.connection = connection
    self.connector_url = "https://connectors.googleapis.com"
    self.service_account_json = service_account_json
    self.credential_cache = None

  def get_connection_details(self) -> Dict[str, Any]:
    """Retrieves service details (service name and host) for a given connection.

    Also returns if auth override is enabled for the connection.

    Returns:
        tuple: A tuple containing (service_name, host).

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error.
        Exception: For any other unexpected errors.
    """
    url = f"{self.connector_url}/v1/projects/{self.project}/locations/{self.location}/connections/{self.connection}?view=BASIC"

    response = self._execute_api_call(url)

    connection_data = response.json()
    connection_name = connection_data.get("name", "")
    service_name = connection_data.get("serviceDirectory", "")
    host = connection_data.get("host", "")
    if host:
      service_name = connection_data.get("tlsServiceDirectory", "")
    auth_override_enabled = connection_data.get("authOverrideEnabled", False)
    return {
        "name": connection_name,
        "serviceName": service_name,
        "host": host,
        "authOverrideEnabled": auth_override_enabled,
    }

  def get_entity_schema_and_operations(
      self, entity: str
  ) -> Tuple[Dict[str, Any], List[str]]:
    """Retrieves the JSON schema for a given entity in a connection.

    Args:
        entity (str): The entity name.

    Returns:
        tuple: A tuple containing (schema, operations).

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request or processing error.
        Exception: For any other unexpected errors.
    """
    url = f"{self.connector_url}/v1/projects/{self.project}/locations/{self.location}/connections/{self.connection}/connectionSchemaMetadata:getEntityType?entityId={entity}"

    response = self._execute_api_call(url)
    operation_id = response.json().get("name")

    if not operation_id:
      raise ValueError(
          f"Failed to get entity schema and operations for entity: {entity}"
      )

    operation_response = self._poll_operation(operation_id)

    schema = operation_response.get("response", {}).get("jsonSchema", {})
    operations = operation_response.get("response", {}).get("operations", [])
    return schema, operations

  def get_action_schema(self, action: str) -> Dict[str, Any]:
    """Retrieves the input and output JSON schema for a given action in a connection.

    Args:
        action (str): The action name.

    Returns:
        tuple: A tuple containing (input_schema, output_schema).

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request or processing error.
        Exception: For any other unexpected errors.
    """
    url = f"{self.connector_url}/v1/projects/{self.project}/locations/{self.location}/connections/{self.connection}/connectionSchemaMetadata:getAction?actionId={action}"

    response = self._execute_api_call(url)

    operation_id = response.json().get("name")

    if not operation_id:
      raise ValueError(f"Failed to get action schema for action: {action}")

    operation_response = self._poll_operation(operation_id)

    input_schema = operation_response.get("response", {}).get(
        "inputJsonSchema", {}
    )
    output_schema = operation_response.get("response", {}).get(
        "outputJsonSchema", {}
    )
    description = operation_response.get("response", {}).get("description", "")
    display_name = operation_response.get("response", {}).get("displayName", "")
    return {
        "inputSchema": input_schema,
        "outputSchema": output_schema,
        "description": description,
        "displayName": display_name,
    }

  @staticmethod
  def get_connector_base_spec() -> Dict[str, Any]:
    return {
        "openapi": "3.0.1",
        "info": {
            "title": "ExecuteConnection",
            "description": "This tool can execute a query on connection",
            "version": "4",
        },
        "servers": [{"url": "https://integrations.googleapis.com"}],
        "security": [
            {"google_auth": ["https://www.googleapis.com/auth/cloud-platform"]}
        ],
        "paths": {},
        "components": {
            "schemas": {
                "operation": {
                    "type": "string",
                    "default": "LIST_ENTITIES",
                    "description": (
                        "Operation to execute. Possible values are"
                        " LIST_ENTITIES, GET_ENTITY, CREATE_ENTITY,"
                        " UPDATE_ENTITY, DELETE_ENTITY in case of entities."
                        " EXECUTE_ACTION in case of actions. and EXECUTE_QUERY"
                        " in case of custom queries."
                    ),
                },
                "entityId": {
                    "type": "string",
                    "description": "Name of the entity",
                },
                "connectorInputPayload": {"type": "object"},
                "filterClause": {
                    "type": "string",
                    "default": "",
                    "description": "WHERE clause in SQL query",
                },
                "pageSize": {
                    "type": "integer",
                    "default": 50,
                    "description": (
                        "Number of entities to return in the response"
                    ),
                },
                "pageToken": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Page token to return the next page of entities"
                    ),
                },
                "connectionName": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Connection resource name to run the query for"
                    ),
                },
                "serviceName": {
                    "type": "string",
                    "default": "",
                    "description": "Service directory for the connection",
                },
                "host": {
                    "type": "string",
                    "default": "",
                    "description": "Host name incase of tls service directory",
                },
                "entity": {
                    "type": "string",
                    "default": "Issues",
                    "description": "Entity to run the query for",
                },
                "action": {
                    "type": "string",
                    "default": "ExecuteCustomQuery",
                    "description": "Action to run the query for",
                },
                "query": {
                    "type": "string",
                    "default": "",
                    "description": "Custom Query to execute on the connection",
                },
                "dynamicAuthConfig": {
                    "type": "object",
                    "default": {},
                    "description": "Dynamic auth config for the connection",
                },
                "timeout": {
                    "type": "integer",
                    "default": 120,
                    "description": (
                        "Timeout in seconds for execution of custom query"
                    ),
                },
                "sortByColumns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Column to sort the results by",
                },
                "connectorOutputPayload": {"type": "object"},
                "nextPageToken": {"type": "string"},
                "execute-connector_Response": {
                    "required": ["connectorOutputPayload"],
                    "type": "object",
                    "properties": {
                        "connectorOutputPayload": {
                            "$ref": (
                                "#/components/schemas/connectorOutputPayload"
                            )
                        },
                        "nextPageToken": {
                            "$ref": "#/components/schemas/nextPageToken"
                        },
                    },
                },
            },
            "securitySchemes": {
                "google_auth": {
                    "type": "oauth2",
                    "flows": {
                        "implicit": {
                            "authorizationUrl": (
                                "https://accounts.google.com/o/oauth2/auth"
                            ),
                            "scopes": {
                                "https://www.googleapis.com/auth/cloud-platform": (
                                    "Auth for google cloud services"
                                )
                            },
                        }
                    },
                }
            },
        },
    }

  @staticmethod
  def get_action_operation(
      action: str,
      operation: str,
      action_display_name: str,
      tool_name: str = "",
      tool_instructions: str = "",
  ) -> Dict[str, Any]:
    description = f"Use this tool to execute {action}"
    if operation == "EXECUTE_QUERY":
      description += (
          " Use pageSize = 50 and timeout = 120 until user specifies a"
          " different value otherwise. If user provides a query in natural"
          " language, convert it to SQL query and then execute it using the"
          " tool."
      )
    return {
        "post": {
            "summary": f"{action_display_name}",
            "description": f"{description} {tool_instructions}",
            "operationId": f"{tool_name}_{action_display_name}",
            "x-action": f"{action}",
            "x-operation": f"{operation}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": f"#/components/schemas/{action_display_name}_Request"
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": f"#/components/schemas/{action_display_name}_Response",
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def list_operation(
      entity: str,
      schema_as_string: str = "",
      tool_name: str = "",
      tool_instructions: str = "",
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"List {entity}",
            "description": f"""Returns the list of {entity} data. If the page token was available in the response, let users know there are more records available. Ask if the user wants to fetch the next page of results. When passing filter use the
                following format: `field_name1='value1' AND field_name2='value2'
                `. {tool_instructions}""",
            "x-operation": "LIST_ENTITIES",
            "x-entity": f"{entity}",
            "operationId": f"{tool_name}_list_{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": (
                                f"#/components/schemas/list_{entity}_Request"
                            )
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "description": (
                                    f"Returns a list of {entity} of json"
                                    f" schema: {schema_as_string}"
                                ),
                                "$ref": "#/components/schemas/execute-connector_Response",
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def get_operation(
      entity: str,
      schema_as_string: str = "",
      tool_name: str = "",
      tool_instructions: str = "",
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"Get {entity}",
            "description": (
                f"Returns the details of the {entity}. {tool_instructions}"
            ),
            "operationId": f"{tool_name}_get_{entity}",
            "x-operation": "GET_ENTITY",
            "x-entity": f"{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": f"#/components/schemas/get_{entity}_Request"
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "description": (
                                    f"Returns {entity} of json schema:"
                                    f" {schema_as_string}"
                                ),
                                "$ref": "#/components/schemas/execute-connector_Response",
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def create_operation(
      entity: str, tool_name: str = "", tool_instructions: str = ""
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"Creates a new {entity}",
            "description": f"Creates a new {entity}. {tool_instructions}",
            "x-operation": "CREATE_ENTITY",
            "x-entity": f"{entity}",
            "operationId": f"{tool_name}_create_{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": (
                                f"#/components/schemas/create_{entity}_Request"
                            )
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/execute-connector_Response"
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def update_operation(
      entity: str, tool_name: str = "", tool_instructions: str = ""
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"Updates the {entity}",
            "description": f"Updates the {entity}. {tool_instructions}",
            "x-operation": "UPDATE_ENTITY",
            "x-entity": f"{entity}",
            "operationId": f"{tool_name}_update_{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": (
                                f"#/components/schemas/update_{entity}_Request"
                            )
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/execute-connector_Response"
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def delete_operation(
      entity: str, tool_name: str = "", tool_instructions: str = ""
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"Delete the {entity}",
            "description": f"Deletes the {entity}. {tool_instructions}",
            "x-operation": "DELETE_ENTITY",
            "x-entity": f"{entity}",
            "operationId": f"{tool_name}_delete_{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": (
                                f"#/components/schemas/delete_{entity}_Request"
                            )
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/execute-connector_Response"
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def create_operation_request(entity: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "connectorInputPayload",
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "connectorInputPayload": {
                "$ref": f"#/components/schemas/connectorInputPayload_{entity}"
            },
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
            "dynamicAuthConfig": {
                "$ref": "#/components/schemas/dynamicAuthConfig"
            },
        },
    }

  @staticmethod
  def update_operation_request(entity: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "connectorInputPayload",
            "entityId",
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "connectorInputPayload": {
                "$ref": f"#/components/schemas/connectorInputPayload_{entity}"
            },
            "entityId": {"$ref": "#/components/schemas/entityId"},
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
            "dynamicAuthConfig": {
                "$ref": "#/components/schemas/dynamicAuthConfig"
            },
            "filterClause": {"$ref": "#/components/schemas/filterClause"},
        },
    }

  @staticmethod
  def get_operation_request() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "entityId",
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "entityId": {"$ref": "#/components/schemas/entityId"},
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
            "dynamicAuthConfig": {
                "$ref": "#/components/schemas/dynamicAuthConfig"
            },
        },
    }

  @staticmethod
  def delete_operation_request() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "entityId",
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "entityId": {"$ref": "#/components/schemas/entityId"},
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
            "dynamicAuthConfig": {
                "$ref": "#/components/schemas/dynamicAuthConfig"
            },
            "filterClause": {"$ref": "#/components/schemas/filterClause"},
        },
    }

  @staticmethod
  def list_operation_request() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "filterClause": {"$ref": "#/components/schemas/filterClause"},
            "pageSize": {"$ref": "#/components/schemas/pageSize"},
            "pageToken": {"$ref": "#/components/schemas/pageToken"},
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
            "sortByColumns": {"$ref": "#/components/schemas/sortByColumns"},
            "dynamicAuthConfig": {
                "$ref": "#/components/schemas/dynamicAuthConfig"
            },
        },
    }

  @staticmethod
  def action_request(action: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "action",
            "connectorInputPayload",
        ],
        "properties": {
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "action": {"$ref": "#/components/schemas/action"},
            "connectorInputPayload": {
                "$ref": f"#/components/schemas/connectorInputPayload_{action}"
            },
            "dynamicAuthConfig": {
                "$ref": "#/components/schemas/dynamicAuthConfig"
            },
        },
    }

  @staticmethod
  def action_response(action: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "connectorOutputPayload": {
                "$ref": f"#/components/schemas/connectorOutputPayload_{action}"
            },
        },
    }

  @staticmethod
  def execute_custom_query_request() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "action",
            "query",
            "timeout",
            "pageSize",
        ],
        "properties": {
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "action": {"$ref": "#/components/schemas/action"},
            "query": {"$ref": "#/components/schemas/query"},
            "timeout": {"$ref": "#/components/schemas/timeout"},
            "pageSize": {"$ref": "#/components/schemas/pageSize"},
            "dynamicAuthConfig": {
                "$ref": "#/components/schemas/dynamicAuthConfig"
            },
        },
    }

  def connector_payload(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
    return self._convert_json_schema_to_openapi_schema(json_schema)

  def _convert_json_schema_to_openapi_schema(self, json_schema):
    """Converts a JSON schema dictionary to an OpenAPI schema dictionary, handling variable types, properties, items, nullable, and description.

    Args:
        json_schema (dict): The input JSON schema dictionary.

    Returns:
        dict: The converted OpenAPI schema dictionary.
    """
    openapi_schema = {}

    if "description" in json_schema:
      openapi_schema["description"] = json_schema["description"]

    if "type" in json_schema:
      if isinstance(json_schema["type"], list):
        if "null" in json_schema["type"]:
          openapi_schema["nullable"] = True
          other_types = [t for t in json_schema["type"] if t != "null"]
          if other_types:
            openapi_schema["type"] = other_types[0]
        else:
          openapi_schema["type"] = json_schema["type"][0]
      else:
        openapi_schema["type"] = json_schema["type"]

    if openapi_schema.get("type") == "object" and "properties" in json_schema:
      openapi_schema["properties"] = {}
      for prop_name, prop_schema in json_schema["properties"].items():
        openapi_schema["properties"][prop_name] = (
            self._convert_json_schema_to_openapi_schema(prop_schema)
        )

    elif openapi_schema.get("type") == "array" and "items" in json_schema:
      if isinstance(json_schema["items"], list):
        openapi_schema["items"] = [
            self._convert_json_schema_to_openapi_schema(item)
            for item in json_schema["items"]
        ]
      else:
        openapi_schema["items"] = self._convert_json_schema_to_openapi_schema(
            json_schema["items"]
        )

    return openapi_schema

  def _get_access_token(self) -> str:
    """Gets the access token for the service account.

    Returns:
        The access token.
    """
    if self.credential_cache and not self.credential_cache.expired:
      return self.credential_cache.token

    if self.service_account_json:
      credentials = service_account.Credentials.from_service_account_info(
          json.loads(self.service_account_json),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
    else:
      try:
        credentials, _ = default_service_credential(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
      except:
        credentials = None

    if not credentials:
      raise ValueError(
          "Please provide a service account that has the required permissions"
          " to access the connection."
      )

    credentials.refresh(Request())
    self.credential_cache = credentials
    return credentials.token

  def _execute_api_call(self, url):
    """Executes an API call to the given URL.

    Args:
        url (str): The URL to call.

    Returns:
        requests.Response: The response object from the API call.

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error.
        Exception: For any other unexpected errors.
    """
    try:
      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {self._get_access_token()}",
      }

      response = requests.get(url, headers=headers)
      response.raise_for_status()
      return response

    except google.auth.exceptions.DefaultCredentialsError as e:
      raise PermissionError(f"Credentials error: {e}") from e

    except requests.exceptions.RequestException as e:
      if (
          "404" in str(e)
          or "Not found" in str(e)
          or "400" in str(e)
          or "Bad request" in str(e)
      ):
        raise ValueError(
            "Invalid request. Please check the provided"
            f" values of project({self.project}), location({self.location}),"
            f" connection({self.connection})."
        ) from e
      raise ValueError(f"Request error: {e}") from e

    except Exception as e:
      raise Exception(f"An unexpected error occurred: {e}") from e

  def _poll_operation(self, operation_id: str) -> Dict[str, Any]:
    """Polls an operation until it is done.

    Args:
        operation_id: The ID of the operation to poll.

    Returns:
        The final response of the operation.

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error.
        Exception: For any other unexpected errors.
    """
    operation_done: bool = False
    operation_response: Dict[str, Any] = {}
    while not operation_done:
      get_operation_url = f"{self.connector_url}/v1/{operation_id}"
      response = self._execute_api_call(get_operation_url)
      operation_response = response.json()
      operation_done = operation_response.get("done", False)
      time.sleep(1)
    return operation_response

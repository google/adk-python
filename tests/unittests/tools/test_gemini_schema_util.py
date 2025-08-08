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

from google.adk.tools._gemini_schema_util import _sanitize_schema_formats_for_gemini
from google.adk.tools._gemini_schema_util import _to_gemini_schema
from google.adk.tools._gemini_schema_util import _to_snake_case
from google.genai.types import Schema
from google.genai.types import Type
import pytest


class TestToGeminiSchema:

  def test_to_gemini_schema_none(self):
    assert _to_gemini_schema(None) is None

  def test_to_gemini_schema_not_dict(self):
    with pytest.raises(TypeError, match="openapi_schema must be a dictionary"):
      _to_gemini_schema("not a dict")

  def test_to_gemini_schema_empty_dict(self):
    result = _to_gemini_schema({})
    assert isinstance(result, Schema)
    assert result.type is Type.OBJECT
    assert result.properties is None

  def test_to_gemini_schema_dict_with_only_object_type(self):
    result = _to_gemini_schema({"type": "object"})
    assert isinstance(result, Schema)
    assert result.type == Type.OBJECT
    assert result.properties is None

  def test_to_gemini_schema_basic_types(self):
    openapi_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert isinstance(gemini_schema, Schema)
    assert gemini_schema.type == Type.OBJECT
    assert gemini_schema.properties["name"].type == Type.STRING
    assert gemini_schema.properties["age"].type == Type.INTEGER
    assert gemini_schema.properties["is_active"].type == Type.BOOLEAN

  def test_to_gemini_schema_array_string_types(self):
    openapi_schema = {
        "type": "object",
        "properties": {
            "boolean_field": {"type": "boolean"},
            "nonnullable_string": {"type": ["string"]},
            "nullable_string": {"type": ["string", "null"]},
            "nullable_number": {"type": ["null", "integer"]},
            "nullable_object": {"type": ["object", "null"]},
            "multi_types_nullable": {"type": ["string", "null", "integer"]},
            "only_null": {"type": "null"},
            "empty_default_object": {},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert isinstance(gemini_schema, Schema)
    assert gemini_schema.type == Type.OBJECT
    assert gemini_schema.properties["boolean_field"].type == Type.BOOLEAN

    assert gemini_schema.properties["nonnullable_string"].type == Type.STRING
    assert not gemini_schema.properties["nonnullable_string"].nullable

    assert gemini_schema.properties["nullable_string"].type == Type.STRING
    assert gemini_schema.properties["nullable_string"].nullable

    assert gemini_schema.properties["nullable_number"].type == Type.INTEGER
    assert gemini_schema.properties["nullable_number"].nullable

    assert gemini_schema.properties["nullable_object"].type == Type.OBJECT
    assert gemini_schema.properties["nullable_object"].nullable

    assert gemini_schema.properties["multi_types_nullable"].any_of == [
        Schema(type=Type.STRING),
        Schema(type=Type.INTEGER),
    ]
    assert gemini_schema.properties["multi_types_nullable"].nullable

    assert gemini_schema.properties["only_null"].type is None
    assert gemini_schema.properties["only_null"].nullable

    assert gemini_schema.properties["empty_default_object"].type == Type.OBJECT
    assert gemini_schema.properties["empty_default_object"].nullable is None

  def test_to_gemini_schema_nested_objects(self):
    openapi_schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
            }
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.properties["address"].type == Type.OBJECT
    assert (
        gemini_schema.properties["address"].properties["street"].type
        == Type.STRING
    )
    assert (
        gemini_schema.properties["address"].properties["city"].type
        == Type.STRING
    )

  def test_to_gemini_schema_array(self):
    openapi_schema = {
        "type": "array",
        "items": {"type": "string"},
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.type == Type.ARRAY
    assert gemini_schema.items.type == Type.STRING

  def test_to_gemini_schema_nested_array(self):
    openapi_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.items.properties["name"].type == Type.STRING

  def test_to_gemini_schema_any_of(self):
    openapi_schema = {
        "anyOf": [{"type": "string"}, {"type": "integer"}],
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert len(gemini_schema.any_of) == 2
    assert gemini_schema.any_of[0].type == Type.STRING
    assert gemini_schema.any_of[1].type == Type.INTEGER

  def test_to_gemini_schema_any_of_nullable(self):
    openapi_schema = {
        "anyOf": [{"type": "string"}, {"type": "null"}],
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.type == Type.STRING
    assert gemini_schema.nullable

  def test_to_gemini_schema_general_list(self):
    openapi_schema = {
        "type": "array",
        "properties": {
            "list_field": {"type": "array", "items": {"type": "string"}},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.properties["list_field"].type == Type.ARRAY
    assert gemini_schema.properties["list_field"].items.type == Type.STRING

  def test_to_gemini_schema_enum(self):
    openapi_schema = {"type": "string", "enum": ["a", "b", "c"]}
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.enum == ["a", "b", "c"]

  def test_to_gemini_schema_required(self):
    openapi_schema = {
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.required == ["name"]

  def test_to_gemini_schema_nested_dict(self):
    openapi_schema = {
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {
                    "key1": {"type": "object"},
                    "key2": {"type": "string"},
                },
            }
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    # Since metadata is neither properties nor item, it will call to_gemini_schema recursively.
    assert isinstance(gemini_schema.properties["metadata"], Schema)
    assert (
        gemini_schema.properties["metadata"].type == Type.OBJECT
    )  # add object type by default
    assert len(gemini_schema.properties["metadata"].properties) == 2
    assert (
        gemini_schema.properties["metadata"].properties["key1"].type
        == Type.OBJECT
    )
    assert (
        gemini_schema.properties["metadata"].properties["key2"].type
        == Type.STRING
    )

  def test_to_gemini_schema_converts_property_dict(self):
    openapi_schema = {
        "properties": {
            "name": {"type": "string", "description": "The property key"},
            "value": {"type": "string", "description": "The property value"},
        },
        "type": "object",
        "description": "A single property entry in the Properties message.",
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.type == Type.OBJECT
    assert gemini_schema.properties["name"].type == Type.STRING
    assert gemini_schema.properties["value"].type == Type.STRING

  def test_to_gemini_schema_remove_unrecognized_fields(self):
    openapi_schema = {
        "type": "string",
        "description": "A single date string.",
        "format": "date",
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.type == Type.STRING
    assert not gemini_schema.format

  def test_to_gemini_schema_nested_dict_with_defs_and_ref(self):
    """Test that nested dict with $defs and $refs is converted correctly."""
    openapi_schema = {
        "$defs": {
            "DeviceEnum": {
                "enum": ["GLOBAL", "desktop", "mobile"],
                "title": "DeviceEnum",
                "type": "string",
            },
            "DomainPayload": {
                "properties": {
                    "adDomain": {
                        "description": "List of one or many domains.",
                        "items": {"type": "string"},
                        "title": "Addomain",
                        "type": "array",
                    },
                    "device": {
                        "$ref": "#/$defs/DeviceEnum",
                        "default": "GLOBAL",
                        "description": (
                            "Filter by device. All devices are returned by"
                            " default."
                        ),
                    },
                },
                "required": ["adDomain"],
                "title": "DomainPayload",
                "type": "object",
            },
        },
        "properties": {"payload": {"$ref": "#/$defs/DomainPayload"}},
        "required": ["payload"],
        "title": "query_domainsArguments",
        "type": "object",
    }
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.type == Type.OBJECT
    assert gemini_schema.properties["payload"].type == Type.OBJECT
    assert (
        gemini_schema.properties["payload"].properties["adDomain"].type
        == Type.ARRAY
    )
    assert (
        gemini_schema.properties["payload"].properties["adDomain"].items.type
        == Type.STRING
    )
    assert (
        gemini_schema.properties["payload"].properties["device"].type
        == Type.STRING
    )
    assert gemini_schema.properties["payload"].properties["device"].enum == [
        "GLOBAL",
        "desktop",
        "mobile",
    ]
    assert gemini_schema.properties["payload"].required == ["adDomain"]

  def test_sanitize_integer_formats(self):
    """Test that int32 and int64 formats are preserved for integer types"""
    openapi_schema = {
        "type": "object",
        "properties": {
            "int32_field": {"type": "integer", "format": "int32"},
            "int64_field": {"type": "integer", "format": "int64"},
            "invalid_int_format": {"type": "integer", "format": "unsigned"},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)

    # int32 and int64 should be preserved
    assert gemini_schema.properties["int32_field"].format == "int32"
    assert gemini_schema.properties["int64_field"].format == "int64"
    # Invalid format should be removed
    assert gemini_schema.properties["invalid_int_format"].format is None

  def test_sanitize_string_formats(self):
    """Test that only date-time and enum formats are preserved for string types"""
    openapi_schema = {
        "type": "object",
        "properties": {
            "datetime_field": {"type": "string", "format": "date-time"},
            "enum_field": {
                "type": "string",
                "format": "enum",
                "enum": ["a", "b"],
            },
            "date_field": {"type": "string", "format": "date"},
            "email_field": {"type": "string", "format": "email"},
            "byte_field": {"type": "string", "format": "byte"},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)

    # date-time and enum should be preserved
    assert gemini_schema.properties["datetime_field"].format == "date-time"
    assert gemini_schema.properties["enum_field"].format == "enum"
    # Other formats should be removed
    assert gemini_schema.properties["date_field"].format is None
    assert gemini_schema.properties["email_field"].format is None
    assert gemini_schema.properties["byte_field"].format is None

  def test_sanitize_number_formats(self):
    """Test format handling for number types"""
    openapi_schema = {
        "type": "object",
        "properties": {
            "float_field": {"type": "number", "format": "float"},
            "double_field": {"type": "number", "format": "double"},
            "int32_number": {"type": "number", "format": "int32"},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)

    # float and double should be removed for number type
    assert gemini_schema.properties["float_field"].format is None
    assert gemini_schema.properties["double_field"].format is None
    # int32 should be preserved even for number type
    assert gemini_schema.properties["int32_number"].format == "int32"

  def test_sanitize_nested_formats(self):
    """Test format sanitization in nested structures"""
    openapi_schema = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {
                    "date_str": {"type": "string", "format": "date"},
                    "int_field": {"type": "integer", "format": "int64"},
                },
            },
            "array_field": {
                "type": "array",
                "items": {"type": "string", "format": "uri"},
            },
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)

    # Check nested object
    assert (
        gemini_schema.properties["nested"].properties["date_str"].format is None
    )
    assert (
        gemini_schema.properties["nested"].properties["int_field"].format
        == "int64"
    )
    # Check array items
    assert gemini_schema.properties["array_field"].items.format is None

  def test_sanitize_anyof_formats(self):
    """Test format sanitization in anyOf structures"""
    openapi_schema = {
        "anyOf": [
            {"type": "string", "format": "email"},
            {"type": "integer", "format": "int32"},
            {"type": "string", "format": "date-time"},
        ],
    }
    gemini_schema = _to_gemini_schema(openapi_schema)

    # First anyOf should have format removed (email)
    assert gemini_schema.any_of[0].format is None
    # Second anyOf should preserve int32
    assert gemini_schema.any_of[1].format == "int32"
    # Third anyOf should preserve date-time
    assert gemini_schema.any_of[2].format == "date-time"

  def test_camel_case_to_snake_case_conversion(self):
    """Test that camelCase keys are converted to snake_case"""
    openapi_schema = {
        "type": "object",
        "minProperties": 1,
        "maxProperties": 10,
        "properties": {
            "firstName": {"type": "string", "minLength": 1, "maxLength": 50},
            "lastName": {"type": "string", "minLength": 1, "maxLength": 50},
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)

    # Check snake_case conversion
    assert gemini_schema.min_properties == 1
    assert gemini_schema.max_properties == 10
    assert gemini_schema.properties["firstName"].min_length == 1
    assert gemini_schema.properties["firstName"].max_length == 50

  def test_preserve_valid_formats_without_type(self):
    """Test behavior when format is specified but type is missing"""
    openapi_schema = {
        "format": "date-time",  # No type specified
        "properties": {
            "field1": {"format": "int32"},  # No type
        },
    }
    gemini_schema = _to_gemini_schema(openapi_schema)

    # Format should be removed when type is not specified
    assert gemini_schema.format is None
    assert gemini_schema.properties["field1"].format is None

  def test_to_gemini_schema_property_ordering(self):
    openapi_schema = {
        "type": "object",
        "propertyOrdering": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }

    gemini_schema = _to_gemini_schema(openapi_schema)
    assert gemini_schema.property_ordering == ["name", "age"]

  def test_sanitize_schema_formats_for_gemini(self):
    schema = {
        "type": "object",
        "description": "Test schema",  # Top-level description
        "properties": {
            "valid_int": {"type": "integer", "format": "int32"},
            "invalid_format_prop": {"type": "integer", "format": "unsigned"},
            "valid_string": {"type": "string", "format": "date-time"},
            "camelCaseKey": {"type": "string"},
            "prop_with_extra_key": {
                "type": "boolean",
                "unknownInternalKey": "discard_this_value",
            },
        },
        "required": ["valid_int"],
        "additionalProperties": False,  # This is an unsupported top-level key
        "unknownTopLevelKey": (
            "discard_me_too"
        ),  # Another unsupported top-level key
    }
    sanitized = _sanitize_schema_formats_for_gemini(schema)

    # Check description is preserved
    assert sanitized["description"] == "Test schema"

    # Check properties and their sanitization
    assert "properties" in sanitized
    sanitized_props = sanitized["properties"]

    assert "valid_int" in sanitized_props
    assert sanitized_props["valid_int"]["type"] == "integer"
    assert sanitized_props["valid_int"]["format"] == "int32"

    assert "invalid_format_prop" in sanitized_props
    assert sanitized_props["invalid_format_prop"]["type"] == "integer"
    assert (
        "format" not in sanitized_props["invalid_format_prop"]
    )  # Invalid format removed

    assert "valid_string" in sanitized_props
    assert sanitized_props["valid_string"]["type"] == "string"
    assert sanitized_props["valid_string"]["format"] == "date-time"

    # Check camelCase keys not changed for properties
    assert "camel_case_key" not in sanitized_props
    assert "camelCaseKey" in sanitized_props
    assert sanitized_props["camelCaseKey"]["type"] == "string"

    # Check removal of unsupported keys within a property definition
    assert "prop_with_extra_key" in sanitized_props
    assert sanitized_props["prop_with_extra_key"]["type"] == "boolean"
    assert (
        "unknown_internal_key"  # snake_cased version of unknownInternalKey
        not in sanitized_props["prop_with_extra_key"]
    )

    # Check removal of unsupported top-level fields (after snake_casing)
    assert "additional_properties" not in sanitized
    assert "unknown_top_level_key" not in sanitized

    # Check original unsupported top-level field names are not there either
    assert "additionalProperties" not in sanitized
    assert "unknownTopLevelKey" not in sanitized

    # Check required is preserved
    assert sanitized["required"] == ["valid_int"]

    # Test with a schema that has a list of types for a property
    schema_with_list_type = {
        "type": "object",
        "properties": {
            "nullable_field": {"type": ["string", "null"], "format": "uuid"}
        },
    }
    sanitized_list_type = _sanitize_schema_formats_for_gemini(
        schema_with_list_type
    )
    # format should be removed because 'uuid' is not supported for string
    assert "format" not in sanitized_list_type["properties"]["nullable_field"]
    # type should be processed by _sanitize_schema_type and preserved
    assert sanitized_list_type["properties"]["nullable_field"]["type"] == [
        "string",
        "null",
    ]

  def test_sanitize_schema_formats_for_gemini_nullable(self):
    openapi_schema = {
        "properties": {
            "case_id": {
                "description": "The ID of the case.",
                "title": "Case ID",
                "type": "string",
            },
            "next_page_token": {
                "any_of": [
                    {"type": "string"},
                    {"type": ["object", "null"]},
                ],
                "description": (
                    "The nextPageToken to fetch the next page of results."
                ),
                "title": "Next Page Token",
            },
        },
        "required": ["case_id"],
        "title": "list_alerts_by_caseArguments",
        "type": "object",
    }
    openapi_schema = _sanitize_schema_formats_for_gemini(openapi_schema)
    assert openapi_schema == {
        "properties": {
            "case_id": {
                "description": "The ID of the case.",
                "title": "Case ID",
                "type": "string",
            },
            "next_page_token": {
                "any_of": [
                    {"type": "string"},
                    {"type": ["object", "null"]},
                ],
                "description": (
                    "The nextPageToken to fetch the next page of results."
                ),
                "title": "Next Page Token",
            },
        },
        "required": ["case_id"],
        "title": "list_alerts_by_caseArguments",
        "type": "object",
    }

  def test_to_gemini_schema_properties_is_none(self):
    """Tests schema conversion when 'properties' field is None."""
    openapi_schema = {"type": "object", "properties": None}
    gemini_schema = _to_gemini_schema(openapi_schema)
    assert isinstance(gemini_schema, Schema)
    assert gemini_schema.type == Type.OBJECT
    assert gemini_schema.properties is None


class TestToSnakeCase:

  @pytest.mark.parametrize(
      "input_str, expected_output",
      [
          ("lowerCamelCase", "lower_camel_case"),
          ("UpperCamelCase", "upper_camel_case"),
          ("space separated", "space_separated"),
          ("REST API", "rest_api"),
          ("Mixed_CASE with_Spaces", "mixed_case_with_spaces"),
          ("__init__", "init"),
          ("APIKey", "api_key"),
          ("SomeLongURL", "some_long_url"),
          ("CONSTANT_CASE", "constant_case"),
          ("already_snake_case", "already_snake_case"),
          ("single", "single"),
          ("", ""),
          ("  spaced  ", "spaced"),
          ("with123numbers", "with123numbers"),
          ("With_Mixed_123_and_SPACES", "with_mixed_123_and_spaces"),
          ("HTMLParser", "html_parser"),
          ("HTTPResponseCode", "http_response_code"),
          ("a_b_c", "a_b_c"),
          ("A_B_C", "a_b_c"),
          ("fromAtoB", "from_ato_b"),
          ("XMLHTTPRequest", "xmlhttp_request"),
          ("_leading", "leading"),
          ("trailing_", "trailing"),
          ("  leading_and_trailing_  ", "leading_and_trailing"),
          ("Multiple___Underscores", "multiple_underscores"),
          ("  spaces_and___underscores  ", "spaces_and_underscores"),
          ("  _mixed_Case  ", "mixed_case"),
          ("123Start", "123_start"),
          ("End123", "end123"),
          ("Mid123dle", "mid123dle"),
      ],
  )
  def test_to_snake_case(self, input_str, expected_output):
    assert _to_snake_case(input_str) == expected_output

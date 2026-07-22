# mypy: ignore-errors
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


import base64
import datetime
import json
import unittest
from unittest import mock
import uuid

from google.adk.integrations.eventarc import _config as config
from google.adk.integrations.eventarc import _message_tool as message_tool
import google.oauth2.credentials


class TestMessageTool(unittest.TestCase):

  def setUp(self):
    self.mock_client_module = mock.patch.object(
        message_tool, "eventarc_client", autospec=True
    ).start()
    self.mock_publisher_client = mock.MagicMock(spec=["publish"])
    self.mock_client_module.get_publisher_client.return_value = (
        self.mock_publisher_client
    )
    self.mock_eventarc_v1 = mock.patch.object(
        message_tool, "eventarc_publishing_v1", autospec=True
    ).start()

    self.settings = config.EventarcToolConfig(project_id="test-project")
    self.credentials = google.oauth2.credentials.Credentials(token="fake")

  def tearDown(self):
    mock.patch.stopall()

  def test_publish_message_success_text(self):
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data="hello world",
    )
    self.assertEqual(res["status"], "SUCCESS")
    self.assertIn("message_id", res)

    # Verify get_publisher_client was called
    self.mock_client_module.get_publisher_client.assert_called_once_with(
        credentials=self.credentials, project_id="test-project"
    )

  def test_publish_message_custom_timeout(self):
    custom_settings = config.EventarcToolConfig(
        project_id="test-project", publish_timeout=30.0
    )
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=custom_settings,
        data="hello world",
    )
    self.assertEqual(res["status"], "SUCCESS")
    self.mock_publisher_client.publish.assert_called_once()
    call_kwargs = self.mock_publisher_client.publish.call_args.kwargs
    self.assertEqual(call_kwargs.get("timeout"), 30.0)

  def test_publish_message_success_json(self):
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data={"foo": "bar"},
    )
    self.assertEqual(res["status"], "SUCCESS")

  def test_publish_message_base64_encoded(self):
    encoded_data = base64.b64encode(b"binary data").decode("utf-8")
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data=encoded_data,
        is_base64_encoded=True,
    )
    self.assertEqual(res["status"], "SUCCESS")

  def test_publish_message_invalid_base64(self):
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data="not-base64-!@#",
        is_base64_encoded=True,
    )
    self.assertEqual(res["status"], "ERROR")
    self.assertIn("Invalid base64", res["error_details"])

  def test_publish_message_unserializable_json(self):
    class CustomClass:
      pass

    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data={"foo": CustomClass()},
    )
    self.assertEqual(res["status"], "ERROR")
    self.assertIn("Failed to serialize data", res["error_details"])

  def test_publish_message_invalid_inputs(self):
    cases = [
        {
            "name": "invalid_type",
            "update_kwargs": {"type": ""},
            "expected_error": "type must be a non-empty string",
        },
        {
            "name": "invalid_source",
            "update_kwargs": {"source": ""},
            "expected_error": "source must be a non-empty string",
        },
        {
            "name": "invalid_id",
            "update_kwargs": {"id": "   "},
            "expected_error": "id, if provided, must be a non-empty string",
        },
        {
            "name": "invalid_base64_data_type",
            "update_kwargs": {"data": 123, "is_base64_encoded": True},
            "expected_error": (
                "data must be a string when is_base64_encoded is True"
            ),
        },
        {
            "name": "invalid_custom_attributes_type",
            "update_kwargs": {"custom_attributes": "not a dict"},
            "expected_error": "custom_attributes must be a dict",
        },
        {
            "name": "invalid_custom_attributes_keys",
            "update_kwargs": {"custom_attributes": {"InvalidKey!": "val"}},
            "expected_error": "Invalid custom attribute key",
        },
        {
            "name": "invalid_time_type",
            "update_kwargs": {"time": 12345},
            "expected_error": "time must be a string",
        },
        {
            "name": "invalid_time_format",
            "update_kwargs": {"time": "invalid-time"},
            "expected_error": "Invalid RFC 3339",
        },
    ]

    for case in cases:
      with self.subTest(case["name"]):
        kwargs = {
            "bus": "bus",
            "type": "type",
            "source": "source",
            "credentials": self.credentials,
            "settings": self.settings,
        }
        kwargs.update(case["update_kwargs"])
        res = message_tool.publish_message(**kwargs)
        self.assertEqual(res["status"], "ERROR")
        self.assertIn(case["expected_error"], res["error_details"])

  def test_publish_message_time_valid_rfc3339(self):
    valid_times = [
        "2026-06-03T12:00:00Z",
        "2026-06-03T12:00:00.123456Z",
        "2026-06-03T12:00:00+00:00",
        "2026-06-03T12:00:00-07:00",
        "2026-06-03T12:00:00.123+02:00",
    ]
    for valid_time in valid_times:
      with self.subTest(time=valid_time):
        self.mock_eventarc_v1.reset_mock()
        res = message_tool.publish_message(
            bus="bus",
            type="type",
            source="source",
            credentials=self.credentials,
            settings=self.settings,
            time=valid_time,
        )
        self.assertEqual(res["status"], "SUCCESS")
        event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
        attributes = event_kwargs.get("attributes", {})
        self.assertIn("time", attributes)
        self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
            ce_string=valid_time
        )

  def test_publish_message_exception_eviction(self):
    self.mock_publisher_client.publish.side_effect = RuntimeError("API failed")
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
    )
    self.assertEqual(res["status"], "ERROR")
    self.assertIn("API failed", res["error_details"])

    # Verify remove_publisher_client was called
    self.mock_client_module.remove_publisher_client.assert_called_once_with(
        credentials=self.credentials, project_id="test-project"
    )

  @mock.patch.object(message_tool, "opentelemetry", autospec=True)
  def test_publish_message_tracing(self, mock_opentelemetry):
    def inject_mock(carrier):
      carrier["traceparent"] = "00-testtrace-testid-01"
      carrier["tracestate"] = "teststate=1"

    mock_opentelemetry.propagate.get_global_textmap.return_value.inject = (
        inject_mock
    )

    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        include_tracing_extension=True,
    )
    self.assertEqual(res["status"], "SUCCESS")

    # Verify custom attributes are appended to CloudEvent
    self.mock_eventarc_v1.types.CloudEvent.assert_called_once()
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    attributes = event_kwargs.get("attributes", {})
    self.assertIn("traceparent", attributes)
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="00-testtrace-testid-01"
    )

  def test_publish_message_empty_string_data(self):
    # Act
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data="",
    )
    # Assert
    self.assertEqual(res["status"], "SUCCESS")

  def test_publish_message_empty_dict_data(self):
    # Act
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data={},
    )
    # Assert
    self.assertEqual(res["status"], "SUCCESS")

  def test_publish_message_missing_library(self):
    with mock.patch.object(message_tool, "eventarc_publishing_v1", None):
      res = message_tool.publish_message(
          bus="bus",
          type="type",
          source="source",
          credentials=self.credentials,
          settings=self.settings,
      )
      self.assertEqual(res["status"], "ERROR")
      self.assertIn("not installed", res["error_details"])

  def test_publish_message_time_empty_string(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        time="",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertNotIn("time", event_kwargs.get("attributes", {}))

  def test_publish_message_explicit_datacontenttype(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="<xml/>",
        datacontenttype="application/xml",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("text_data"), "<xml/>")
    attributes = event_kwargs.get("attributes", {})
    self.assertIn("datacontenttype", attributes)
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/xml"
    )

  def test_publish_message_image_payload(self):
    # Simulate an agent sending an image
    # "iVBORw0KGgo=" is a valid base64 snippet (e.g. PNG header)
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="iVBORw0KGgo=",
        is_base64_encoded=True,
        datacontenttype="image/png",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("binary_data"), b"\x89PNG\r\n\x1a\n")
    attributes = event_kwargs.get("attributes", {})
    self.assertIn("datacontenttype", attributes)
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="image/png"
    )

  def test_publish_message_explicit_datacontenttype_json_with_binary_data(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="e30=",  # base64 for {}
        is_base64_encoded=True,
        datacontenttype="application/json",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("binary_data"), b"{}")
    attributes = event_kwargs.get("attributes", {})
    self.assertIn("datacontenttype", attributes)
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  def test_publish_message_explicit_datacontenttype_xml_with_dict_data(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data={"foo": "bar"},
        datacontenttype="application/xml",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("text_data"), '{"foo": "bar"}')
    attributes = event_kwargs.get("attributes", {})
    self.assertIn("datacontenttype", attributes)
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/xml"
    )

  def test_publish_message_empty_datacontenttype(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="hello",
        datacontenttype="",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertNotIn("datacontenttype", event_kwargs.get("attributes", {}))

  def test_publish_message_with_subject(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="hello",
        subject="test-subject",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("subject"), "test-subject")

  def test_publish_message_data_integer(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=12345,
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("text_data"), "12345")
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  def test_publish_message_data_boolean(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=True,
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("text_data"), "true")
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  def test_publish_message_data_list_of_dicts(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=[{"a": 1}, {"b": 2}],
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("text_data"), '[{"a": 1}, {"b": 2}]')
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  def test_publish_message_data_unicode(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="Hello 🌍!",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("text_data"), "Hello 🌍!")
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="text/plain"
    )

  def test_publish_message_custom_attributes_type_casting(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        custom_attributes={"isvalid": True, "count": 42},
    )
    self.assertEqual(res["status"], "SUCCESS")
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="True"
    )
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="42"
    )

  def test_publish_message_explicit_specversion(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        specversion="1.1",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("spec_version"), "1.1")

  def test_publish_message_explicit_id(self):
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        id="custom-event-id-99",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("id"), "custom-event-id-99")
    self.assertEqual(res["message_id"], "custom-event-id-99")

  def test_publish_message_base64_without_datacontenttype(self):
    # Simulate an agent sending base64 but forgetting the datacontenttype
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="YmluYXJ5",  # 'binary'
        is_base64_encoded=True,
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("binary_data"), b"binary")
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/octet-stream"
    )

  def test_publish_message_data_deeply_nested_dict(self):
    nested_data = {
        "user": {
            "id": 101,
            "profile": {
                "name": "Alice",
                "preferences": {
                    "notifications": {"email": True, "sms": False},
                    "tags": ["premium", "beta-tester"],
                },
            },
            "history": [
                {"action": "login", "timestamp": "2026-06-04T00:00:00Z"},
                {
                    "action": "purchase",
                    "details": {"item_id": 999, "amount": 42.5},
                },
            ],
        },
        "metadata": {
            "source": "mobile-app",
            "version": [1, 2, {"build": "rc1"}],
        },
    }
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=nested_data,
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("text_data"), json.dumps(nested_data))
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  def test_publish_message_data_deeply_nested_list(self):
    nested_list = [
        [1, 2, [3, 4, [5, {"six": 6}]]],
        {"seven": [8, 9]},
        "ten",
        True,
        None,
        [{"eleven": {"twelve": [13, 14]}}],
    ]
    res = message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=nested_list,
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    self.assertEqual(event_kwargs.get("text_data"), json.dumps(nested_list))
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  def test_publish_message_auto_generated_attributes(self):
    res = message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data="hello world",
    )
    self.assertEqual(res["status"], "SUCCESS")
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs

    # Assert ID is a valid UUIDv4
    generated_id = event_kwargs.get("id")
    self.assertIsNotNone(generated_id)
    uuid_obj = uuid.UUID(generated_id, version=4)
    self.assertEqual(str(uuid_obj), generated_id)

    # Assert Time is auto-generated and valid RFC 3339
    attributes = event_kwargs.get("attributes", {})
    self.assertIn("time", attributes)

    # We need to find the specific CloudEventAttributeValue mock call that corresponds to the time attribute.
    # The actual implementation in message_tool.py populates it in custom_attr["time"] = time_attr
    # Let's inspect the attributes dictionary passed to CloudEvent.
    # We just need to check if ANY of the calls to CloudEventAttributeValue contain a valid RFC 3339 string
    # that could be the time. A simpler approach is to check if it parses via fromisoformat after replacing Z.

    time_val = None
    for (
        call
    ) in (
        self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.mock_calls
    ):
      ce_string = call.kwargs.get("ce_string")
      if ce_string and (
          "T" in ce_string
          and ("Z" in ce_string or "+" in ce_string or "-" in ce_string)
      ):
        # Attempt to parse it
        try:
          dt = datetime.datetime.fromisoformat(ce_string.replace("Z", "+00:00"))
          time_val = ce_string
          break
        except ValueError:
          continue

    self.assertIsNotNone(
        time_val,
        "Failed to find a valid RFC 3339 auto-generated time string in the"
        " attributes.",
    )


if __name__ == "__main__":
  unittest.main()

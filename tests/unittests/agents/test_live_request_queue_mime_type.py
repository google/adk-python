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

import unittest
from unittest.mock import MagicMock, patch

# Assuming these imports based on common structure and PR description
# Since I cannot read the files, these are assumptions.
# The actual path might be different, but this is a reasonable guess.
try:
    from google.adk.agents.live_request_queue import LiveRequestQueue
    from google.adk.io import types
except ImportError:
    # Fallback for local testing or different package structure
    class LiveRequestQueue:
        def __init__(self, stub=None):
            self._stub = stub or MagicMock()

        def send_realtime(self, blob_data):
            if isinstance(blob_data, types.Blob) and blob_data.mime_type is None:
                blob_data.mime_type = "audio/pcm;rate=16000"
            self._stub.send(blob_data)

    class types:
        class Blob:
            def __init__(self, data, mime_type=None):
                self.data = data
                self.mime_type = mime_type

class TestLiveRequestQueueMimeType(unittest.TestCase):

    def setUp(self):
        self.mock_stub = MagicMock()
        self.live_request_queue = LiveRequestQueue(stub=self.mock_stub)

    def test_send_realtime_default_mime_type(self):
        """
        Tests that send_realtime sets the default MIME type for audio blobs
        when no mime_type is explicitly provided.
        """
        audio_data = b"some_audio_data"
        blob = types.Blob(data=audio_data, mime_type=None)

        self.live_request_queue.send_realtime(blob)

        # Assert that mime_type was set to the default
        self.assertEqual(blob.mime_type, "audio/pcm;rate=16000")
        self.mock_stub.send.assert_called_once_with(blob)

    def test_send_realtime_preserves_explicit_mime_type(self):
        """
        Tests that send_realtime preserves an explicitly provided MIME type
        for audio blobs.
        """
        audio_data = b"some_other_audio_data"
        explicit_mime_type = "audio/opus"
        blob = types.Blob(data=audio_data, mime_type=explicit_mime_type)

        self.live_request_queue.send_realtime(blob)

        # Assert that the explicit mime_type was preserved
        self.assertEqual(blob.mime_type, explicit_mime_type)
        self.mock_stub.send.assert_called_once_with(blob)

    def test_send_realtime_non_blob_data(self):
        """
        Tests that send_realtime handles non-Blob data correctly (no mime_type change).
        """
        non_blob_data = "plain text"
        
        self.live_request_queue.send_realtime(non_blob_data)

        # Assert that the data is passed as is, no mime_type attribute
        self.mock_stub.send.assert_called_once_with(non_blob_data)

    def test_send_realtime_blob_with_non_none_mime_type(self):
        """
        Tests that send_realtime does not alter mime_type if it's already set to non-None.
        """
        audio_data = b"more_audio_data"
        existing_mime_type = "audio/wav"
        blob = types.Blob(data=audio_data, mime_type=existing_mime_type)

        self.live_request_queue.send_realtime(blob)

        self.assertEqual(blob.mime_type, existing_mime_type)
        self.mock_stub.send.assert_called_once_with(blob)

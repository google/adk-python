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
import asyncio
from unittest.mock import MagicMock, patch

from google.adk.agents.live_request_queue import LiveRequestQueue, LiveRequest
from google.genai import types


class TestLiveRequestQueueMimeType(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.live_request_queue = LiveRequestQueue()

    async def test_send_realtime_default_mime_type(self):
        """
        Tests that send_realtime sets the default MIME type for audio blobs
        when no mime_type is explicitly provided.
        """
        audio_data = b"some_audio_data"
        blob = types.Blob(data=audio_data, mime_type=None)

        self.live_request_queue.send_realtime(blob)

        # Retrieve the request from the queue
        request = await self.live_request_queue.get()

        # Assert that mime_type was set to the default
        self.assertEqual(request.blob.mime_type, "audio/pcm;rate=16000")
        self.assertEqual(request.blob.data, audio_data)

    async def test_send_realtime_preserves_explicit_mime_type(self):
        """
        Tests that send_realtime preserves an explicitly provided MIME type
        for audio blobs.
        """
        audio_data = b"some_other_audio_data"
        explicit_mime_type = "audio/opus"
        blob = types.Blob(data=audio_data, mime_type=explicit_mime_type)

        self.live_request_queue.send_realtime(blob)

        # Retrieve the request from the queue
        request = await self.live_request_queue.get()

        # Assert that the explicit mime_type was preserved
        self.assertEqual(request.blob.mime_type, explicit_mime_type)
        self.assertEqual(request.blob.data, audio_data)

    # Removed test_send_realtime_non_blob_data as send_realtime expects types.Blob

    async def test_send_realtime_blob_with_non_none_mime_type(self):
        """
        Tests that send_realtime does not alter mime_type if it's already set to non-None.
        """
        audio_data = b"more_audio_data"
        existing_mime_type = "audio/wav"
        blob = types.Blob(data=audio_data, mime_type=existing_mime_type)

        self.live_request_queue.send_realtime(blob)

        # Retrieve the request from the queue
        request = await self.live_request_queue.get()

        self.assertEqual(request.blob.mime_type, existing_mime_type)
        self.assertEqual(request.blob.data, audio_data)

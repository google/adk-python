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

import asyncio
import unittest
from unittest.mock import MagicMock

from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.genai import types


class TestLiveRequestQueueIssue5584(unittest.IsolatedAsyncioTestCase):

  async def test_send_realtime_default_mime_type(self):
    queue = LiveRequestQueue()
    test_data = b"test_audio_data"
    blob = types.Blob(data=test_data)

    queue.send_realtime(blob)
    retrieved_request = await queue.get()

    self.assertIsNotNone(retrieved_request.blob)
    self.assertEqual(retrieved_request.blob.data, test_data)
    self.assertEqual(retrieved_request.blob.mime_type, "audio/pcm;rate=16000")

  async def test_send_realtime_explicit_mime_type(self):
    queue = LiveRequestQueue()
    test_data = b"test_audio_data_opus"
    explicit_mime_type = "audio/opus"
    blob = types.Blob(data=test_data, mime_type=explicit_mime_type)

    queue.send_realtime(blob)
    retrieved_request = await queue.get()

    self.assertIsNotNone(retrieved_request.blob)
    self.assertEqual(retrieved_request.blob.data, test_data)
    self.assertEqual(retrieved_request.blob.mime_type, explicit_mime_type)

if __name__ == '__main__':
  unittest.main()

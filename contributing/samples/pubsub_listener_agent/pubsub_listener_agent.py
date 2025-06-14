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

"""
pubsub_listener_agent.py

An ADK BaseAgent implementation that listens to Google Cloud Pub/Sub subscription messages,
acknowledges them, and emits corresponding ADK Events asynchronously.

Intended to be run as part of an ADK Runner session loop.

Environment variables required:
- APP_NAME
- PROJECT_ID
- SUBSCRIPTION_NAME

"""

import asyncio
import os
import json
import signal
import sys
from typing import AsyncGenerator
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import Runner, InMemorySessionService
from google.adk.events import Event
from google.genai.types import UserContent, ModelContent
from google.cloud import pubsub_v1
from google.api_core.exceptions import DeadlineExceeded
from dotenv import load_dotenv

load_dotenv()

required_env_vars = ["APP_NAME", "PROJECT_ID", "SUBSCRIPTION_NAME"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}. "
        "Please set them in your environment or .env file."
    )

# Pub/Sub config (make sure these are set in your env or manually here)
APP_NAME = os.getenv("APP_NAME", "default_app_name")
PROJECT_ID = os.getenv("PROJECT_ID", "default_project_id")
SUBSCRIPTION_NAME = os.getenv("SUBSCRIPTION_NAME", "default_subscription_name")

SUBSCRIPTION_ID = f"{SUBSCRIPTION_NAME}-{APP_NAME}"
completion_subscription_path = f"projects/{PROJECT_ID}/subscriptions/{SUBSCRIPTION_ID}"

class PubSubListenerAgent(BaseAgent):
    def __init__(self):
        print("[DEBUG] Initializing listener agent")
        super().__init__(name="PubSubBuildListenerAgent", description="Listens to GCP Pub/Sub and emits ADK messages.")

        # Initialize custom vars after base Pydantic setup
        object.__setattr__(self, 'subscriber', pubsub_v1.SubscriberClient())
        object.__setattr__(self, 'subscription_path', self.subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID))
        self._stop_event = asyncio.Event()

    def close(self):
        """Explicitly close the subscriber client."""
        if hasattr(self, 'subscriber') and self.subscriber:
            print("[DEBUG] Closing subscriber client.")
            self.subscriber.close()
            object.__setattr__(self, 'subscriber', None)

    def __del__(self):
        """Destructor to ensure subscriber client is closed."""
        try:
            self.close()
        except Exception as e:
            print(f"[WARN] Exception during PubSubListenerAgent cleanup: {e}")

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """Listen to Pub/Sub messages and yield ADK Events asynchronously."""
        print("[DEBUG] Starting pub/sub listener...")

        loop = asyncio.get_running_loop()
        # Register signal handlers to set stop event
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._stop_event.set)
        else:
            # Windows: fallback - no add_signal_handler, you can rely on external cancellation or other mechanism
            print("[WARN] Signal handlers not supported on Windows; manual shutdown required.")

        while True:
            if self._stop_event.is_set():
                print("[DEBUG] Shutting down PubSubBuildListenerAgent gracefully.")
                self.close()
                break
            else: 
                try:
                    # Pull messages synchronously but with timeout to avoid blocking forever
                    response = self.subscriber.pull(
                        request={
                            "subscription": self.subscription_path,
                            "max_messages": 1,
                        },
                        timeout=5.0  # non-blocking
                    )

                    if not response.received_messages:
                        # No messages, just wait a short bit before retrying
                        await asyncio.sleep(1)
                        continue

                    for received_message in response.received_messages:
                        try:
                            payload = json.loads(received_message.message.data.decode("utf-8"))
                            #print(f"[DEBUG] Received message: {payload}")

                            # Acknowledge the message
                            self.subscriber.acknowledge(
                                request={
                                    "subscription": self.subscription_path,
                                    "ack_ids": [received_message.ack_id],
                                }
                            )

                            # Emit event for this message
                            yield Event(
                                content=ModelContent(f"🎉 Received pubsub notification: {payload}"),
                                author=self.name 
                            )

                        except Exception as e:
                            print(f"[DEBUG] Error processing message: {e}")

                        
                except DeadlineExceeded:
                    # This happens if the server times out waiting for messages; just retry. expected behavior
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"[DEBUG] Exception in run_async loop: {e}")
                    await asyncio.sleep(5)  # Backoff on error

async def main():
    pubsub_agent = PubSubListenerAgent()
    runner = Runner(app_name=APP_NAME, agent=pubsub_agent, session_service=InMemorySessionService())

    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id="dev_user_01",
        session_id="build_listener_session"
    )

    async_gen = runner.run_async(
        user_id="dev_user_01",
        session_id="build_listener_session",
        new_message=UserContent("Starting pub/sub listener agent")
    )
    async for event in async_gen:
        print("Runner received event:", event.content.parts[0].text)

if __name__ == "__main__":
    asyncio.run(main())

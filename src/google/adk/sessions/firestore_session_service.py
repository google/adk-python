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
Implements a session service using Google Cloud Firestore for storage.
This version uses the synchronous Firestore client to prevent event loop errors.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from typing_extensions import override

from google.genai.types import Content

from ..events.event import Event
from ..events.event_actions import EventActions
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State

logger = logging.getLogger("google_adk." + __name__)
# Set the level to INFO to make sure our logs are captured.
logger.setLevel(logging.INFO)

SESSIONS_COLLECTION = "adk_sessions"
EVENTS_SUBCOLLECTION = "events"


class FirestoreSessionService(BaseSessionService):
    def __init__(self, project: Optional[str] = None, database: Optional[str] = None):
        """Initializes the FirestoreSessionService with the synchronous client."""
        # Use the standard synchronous client instead of the AsyncClient
        self._db = firestore.Client(project=project, database=database)

    @override
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """Creates a new session document in Firestore using a thread."""
        # If a session_id is provided, we should ideally use it or handle it according to the base class contract.
        # For now, if Firestore always generates IDs, consider removing session_id from the signature or ignoring it.
        # raise ValueError("User-provided session ID is not supported.")

        def _create_in_firestore():
            session_data = {
                "app_name": app_name,
                "user_id": user_id,
                "state": state or {},
                "createTime": firestore.SERVER_TIMESTAMP,
                "updateTime": firestore.SERVER_TIMESTAMP,
            }
            # These are now synchronous calls
            _, doc_ref = self._db.collection(SESSIONS_COLLECTION).add(session_data)
            doc = doc_ref.get()
            doc_dict = doc.to_dict()
            return Session(
                app_name=doc_dict["app_name"],
                user_id=doc_dict["user_id"],
                id=doc.id,
                state=doc_dict.get("state", {}),
                last_update_time=doc_dict["updateTime"].timestamp(),
            )

        # Run the synchronous DB calls in a separate thread to not block the async server
        return await asyncio.to_thread(_create_in_firestore)

    @override
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """Retrieves a session and its events from Firestore using a thread."""

        def _get_from_firestore():
            session_ref = self._db.collection(SESSIONS_COLLECTION).document(session_id)
            session_doc = session_ref.get()

            if not session_doc.exists:
                return None

            session_dict = session_doc.to_dict()
            if (
                session_dict.get("app_name") != app_name
                or session_dict.get("user_id") != user_id
            ):
                return None

            update_timestamp = session_dict["updateTime"].timestamp()
            session = Session(
                app_name=session_dict["app_name"],
                user_id=session_dict["user_id"],
                id=session_doc.id,
                state=session_dict.get("state", {}),
                last_update_time=update_timestamp,
            )

            # Fetch events without ordering from the database to avoid index requirements.
            events_ref = session_ref.collection(EVENTS_SUBCOLLECTION)
            event_docs = events_ref.stream()
            events_list = [_from_firestore_doc_to_event(doc) for doc in event_docs]
            # Sort the events in the application code instead.
            events_list.sort(key=lambda e: e.timestamp)
            session.events = events_list

            if config:
                if config.num_recent_events:
                    session.events = session.events[-config.num_recent_events :]
                elif config.after_timestamp:
                    session.events = [e for e in session.events if e.timestamp > config.after_timestamp]
            
            return session

        return await asyncio.to_thread(_get_from_firestore)

    @override
    async def list_sessions(self, *, app_name: str, user_id: str) -> ListSessionsResponse:
        """Lists all sessions for a given user and app from Firestore using a thread."""
        def _list_from_firestore():
            query = self._db.collection(SESSIONS_COLLECTION).where(
                filter=FieldFilter("app_name", "==", app_name)
            ).where(filter=FieldFilter("user_id", "==", user_id))
            
            sessions = []
            for doc in query.stream():
                session_dict = doc.to_dict()
                session = Session(
                    app_name=session_dict["app_name"],
                    user_id=session_dict["user_id"],
                    id=doc.id,
                    state=session_dict.get("state", {}),
                    last_update_time=session_dict["updateTime"].timestamp(),
                )
                sessions.append(session)
            return ListSessionsResponse(sessions=sessions)

        return await asyncio.to_thread(_list_from_firestore)

    @override
    async def delete_session(self, *, app_name: str, user_id: str, session_id: str) -> None:
        """Deletes a session and all its events from Firestore using a thread."""
        def _delete_in_firestore():
            session_ref = self._db.collection(SESSIONS_COLLECTION).document(session_id)
            if not session_doc.exists or session_doc.to_dict().get("user_id") != user_id or session_doc.to_dict().get("app_name") != app_name:
                return

            events_ref = session_ref.collection(EVENTS_SUBCOLLECTION)
            event_docs = events_ref.list_documents()
            
            batch = self._db.batch()
            for doc in event_docs:
                batch.delete(doc)
            batch.delete(session_ref)
            batch.commit()
        
        await asyncio.to_thread(_delete_in_firestore)

    async def update_session_state(self, session_id: str, state_delta: Dict[str, Any]):
        """Updates the state of a session document in Firestore."""
        def _update_in_firestore():
            session_ref = self._db.collection(SESSIONS_COLLECTION).document(session_id)
            # Firestore's update with dot notation is perfect for this
            update_data = {f"state.{key}": value for key, value in state_delta.items()}
            update_data["updateTime"] = firestore.SERVER_TIMESTAMP
            session_ref.update(update_data)

        await asyncio.to_thread(_update_in_firestore)

    @override
    async def append_event(self, session: Session, event: Event) -> Event:
        """Appends an event to the session's event subcollection in Firestore using a thread."""
        event = await super().append_event(session=session, event=event)
        if event.actions and event.actions.state_delta:
            await self.update_session_state(session.id, event.actions.state_delta)
        
        def _append_in_firestore():
            logger.info("Starting _append_in_firestore for session '%s'", session.id)
            try:
                # Use a batch for atomic writes.
                batch = self._db.batch()
                
                session_ref = self._db.collection(SESSIONS_COLLECTION).document(session.id)
                
                # Create a new document for the event in the subcollection.
                event_doc_ref = session_ref.collection(EVENTS_SUBCOLLECTION).document()
                event.id = event_doc_ref.id # Assign the new ID to the event object
                
                event_data_dict = _convert_event_to_json(event)
                logger.info("Appending event data: %s", event_data_dict)
                
                # Add the event creation to the batch.
                batch.set(event_doc_ref, event_data_dict)

                # Update the session document's timestamp. State is no longer saved here.
                batch.update(session_ref, {"updateTime": firestore.SERVER_TIMESTAMP})
                # Commit the batch.
                logger.info("Committing batch to Firestore for session '%s'...", session.id)
                batch.commit()
                logger.info("Batch committed successfully for session '%s'.", session.id)
            except Exception as e:
                # Log any exception that occurs during the process.
                logger.error(
                    "!!! Exception in _append_in_firestore for session '%s': %s",
                    session.id,
                    e,
                    exc_info=True
                )
        
        await asyncio.to_thread(_append_in_firestore)
        return event

def _convert_event_to_json(event: Event) -> Dict[str, Any]:
  """Serializes an Event object into a JSON-compatible dictionary."""
  metadata_json = {
      'partial': event.partial,
      'turn_complete': event.turn_complete,
      'interrupted': event.interrupted,
      'branch': event.branch,
      'long_running_tool_ids': (
          list(event.long_running_tool_ids)
          if event.long_running_tool_ids
          else None
      ),
  }
  if event.grounding_metadata:
    metadata_json['grounding_metadata'] = event.grounding_metadata.model_dump(
        exclude_none=True, mode='json'
    )

  event_json = {
      'author': event.author,
      'invocation_id': event.invocation_id,
      'timestamp': {
          'seconds': int(event.timestamp),
          'nanos': int(
              (event.timestamp - int(event.timestamp)) * 1_000_000_000
          ),
      },
      'error_code': event.error_code,
      'error_message': event.error_message,
      'event_metadata': metadata_json,
  }

  if event.actions:
    actions_json = {
        'skip_summarization': event.actions.skip_summarization,
        'state_delta': event.actions.state_delta,
        'artifact_delta': event.actions.artifact_delta,
        'transfer_agent': event.actions.transfer_to_agent,
        'escalate': event.actions.escalate,
        'requested_auth_configs': event.actions.requested_auth_configs,
    }
    event_json['actions'] = actions_json
  if event.content:
    content_dict = event.content.model_dump(exclude_none=True, mode='json')
    if event.author == "user" and 'parts' in content_dict:
        new_parts = []
        for part in content_dict['parts']:
            # Check for inline_data which is how `Part.from_data` stores the PDF
            if 'inline_data' in part and part['inline_data'].get('mime_type') == 'application/pdf':
                # Replace the large PDF data with a placeholder text
                new_parts.append({'text': '[PDF content omitted from history]'})
            else:
                new_parts.append(part)
        content_dict['parts'] = new_parts
    event_json['content'] = content_dict
  if event.error_code:
    event_json['error_code'] = event.error_code
  if event.error_message:
    event_json['error_message'] = event.error_message
  return event_json


def _from_firestore_doc_to_event(doc: firestore.DocumentSnapshot) -> Event:
    """Deserializes a Firestore document into an Event object."""
    event_dict = doc.to_dict()
    event_actions = EventActions()
    if event_dict.get("actions", None):
        actions_data = event_dict["actions"]
        event_actions = EventActions(
            skip_summarization=actions_data.get("skipSummarization", None),
            state_delta=actions_data.get("stateDelta", {}),
            artifact_delta=actions_data.get("artifactDelta", {}),
            transfer_to_agent=actions_data.get("transferAgent", None),
            escalate=actions_data.get("escalate", None),
            requested_auth_configs=actions_data.get("requestedAuthConfigs", {}),
        )

    ts_map = event_dict["timestamp"]
    timestamp_float = ts_map["seconds"] + ts_map.get("nanos", 0) / 1_000_000_000

    content_dict = event_dict.get("content", None)
    content = Content(**content_dict) if content_dict else None

    event = Event(
        id=doc.id,
        invocation_id=event_dict["invocation_id"],
        author=event_dict["author"],
        actions=event_actions,
        content=content,
        timestamp=timestamp_float,
        error_code=event_dict.get("error_code", None),
        error_message=event_dict.get("error_message", None),
    )

    if event_dict.get("event_metadata", None):
        metadata = event_dict["event_metadata"]
        long_running_tool_ids_list = metadata.get("long_running_tool_ids", None)
        event.partial = metadata.get("partial", None)
        event.turn_complete = metadata.get("turn_complete", None)
        event.interrupted = metadata.get("interrupted", None)
        event.branch = metadata.get("branch", None)
        # grounding_metadata_dict = metadata.get("grounding_metadata", None)
        # if grounding_metadata_dict:
        #     event.grounding_metadata = GroundingMetadata(**grounding_metadata_dict)
        event.long_running_tool_ids = (
            set(long_running_tool_ids_list) if long_running_tool_ids_list else None
        )

    return event

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

import chromadb
import json
from typing_extensions import override

from ..events.event import Event
from ..sessions.session import Session
from .base_memory_service import BaseMemoryService
from .base_memory_service import MemoryResult
from .base_memory_service import SearchMemoryResponse
from google.genai import types
from ..events.event_actions import EventActions
from chromadb.config import Settings


class ChromaMemoryService(BaseMemoryService):
    """A memory service that uses Chroma for storage and retrieval."""
    
    def __init__(self, top_k: int = 10, is_persistent: bool = False):
        """Initializes a ChromaMemoryService.
        Args:
            collection_name: The name of the Chroma collection to use.
        """
        settings = Settings(
            allow_reset=True
        )
        self.client = chromadb.EphemeralClient(settings=settings) if not is_persistent else chromadb.PersistentClient(settings=settings)
        self.top_k = top_k
        
    def _clean_metadata_value(self, value):
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        else:
            try:
                return json.dumps(value)
            except:
                return str(value)
        
    @staticmethod
    def _parse_metadata_field_value(value, type_caster=None, is_json_load=False):
        """Helper to parse metadata values for Event reconstruction."""
        if value == "":
            return None
        if value is None:
            return None
        
        if is_json_load:
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    return parsed_value
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON string: {e}")

            return value
        
        if type_caster:
            try:
                casted_value = type_caster(value)
                return casted_value
            except ValueError as e:
                return None 
            
        return value
        
    @override
    async def add_session_to_memory(self, session: Session):
        """Adds a session to the memory service.
        
        Args:
            session: The session to add.
        """
        collection = self.client.get_or_create_collection(f"{session.app_name}_{session.user_id}")
        
        for event in session.events:
            if not event.content or not event.content.parts:
                continue
                
            text_parts = []
            for part in event.content.parts:
                if part.text:
                    text_parts.append(part.text.lower())
            
            if not text_parts:
                continue
                
            text = " ".join(text_parts)
            
            collection.add(
                ids=[f"{session.id}_{event.id}"],
                documents=[text],
                metadatas=[{
                    "session_id": self._clean_metadata_value(session.id),
                    "event_id": self._clean_metadata_value(event.id),
                    "invocation_id": self._clean_metadata_value(event.invocation_id),
                    "author": self._clean_metadata_value(event.author),
                    "timestamp": self._clean_metadata_value(event.timestamp),
                    "branch": self._clean_metadata_value(event.branch or ""),
                    "actions": self._clean_metadata_value(event.actions.model_dump_json() if event.actions else None),
                    "long_running_tool_ids": self._clean_metadata_value(str(list(event.long_running_tool_ids)) if event.long_running_tool_ids else None),
                    "grounding_metadata": self._clean_metadata_value(event.grounding_metadata.model_dump_json() if event.grounding_metadata else None),
                    "partial": self._clean_metadata_value(event.partial),
                    "turn_complete": self._clean_metadata_value(event.turn_complete),
                    "error_code": self._clean_metadata_value(event.error_code),
                    "error_message": self._clean_metadata_value(event.error_message),
                    "interrupted": self._clean_metadata_value(event.interrupted),
                    "custom_metadata": self._clean_metadata_value(event.custom_metadata),
                }],
            )

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Searches for sessions that match the query using both semantic and keyword search.

        Args:
            app_name: The name of the application.
            user_id: The id of the user.
            query: The query to search for.

        Returns:
            A SearchMemoryResponse containing the matching memories.
        """
        try:
            collection = self.client.get_collection(f"{app_name}_{user_id}")
        except Exception as e:
            return SearchMemoryResponse(memories=[])
        
        # Perform hybrid search
        keywords = set(query.lower().split())
        if len(keywords) == 0:
            where_document = {}
        elif len(keywords) == 1:
            keyword_val = keywords.pop()
            where_document = {"$contains": keyword_val}
        else:
            where_document = {"$or": [{"$contains": keyword} for keyword in keywords]}
        
        results = collection.query(
            query_texts=[query] if query else None,
            n_results=self.top_k,
            where_document=where_document if where_document else None,
            include=['metadatas', 'documents']
        )
        
        session_events = {}
        if not (results and results.get("ids") and results["ids"] and \
                isinstance(results["ids"][0], list) and results["ids"][0]):
            return SearchMemoryResponse(memories=[])

        doc_ids = results["ids"][0]
        all_metadatas = results.get("metadatas", [[]])[0]
        all_documents = results.get("documents", [[]])[0]

        if not (len(doc_ids) == len(all_metadatas) == len(all_documents)):
            return SearchMemoryResponse(memories=[])

        for i, doc_id_str in enumerate(doc_ids):
            if not isinstance(doc_id_str, str):
                continue

            try:
                id_parts = doc_id_str.rsplit('_', 1)
                if len(id_parts) == 2:
                    session_id, event_id = id_parts[0], id_parts[1]
                else:
                    continue 
            except Exception as e:
                continue
            
            current_metadata = all_metadatas[i]
            current_document_text = all_documents[i]
            
            event = Event(
                id=event_id, # Use the correctly parsed event_id
                invocation_id=current_metadata.get("invocation_id"),
                author=current_metadata.get("author"),
                timestamp=ChromaMemoryService._parse_metadata_field_value(current_metadata.get("timestamp"), type_caster=float),
                content=types.Content(parts=[types.Part(text=current_document_text)]),
                branch=current_metadata.get("branch") if current_metadata.get("branch") else None,
                actions=EventActions.model_validate_json(current_metadata["actions"]) if current_metadata.get("actions") else None,
                long_running_tool_ids=set(eval(current_metadata["long_running_tool_ids"])) if current_metadata.get("long_running_tool_ids") else None,
                grounding_metadata=types.GroundingMetadata.model_validate_json(current_metadata["grounding_metadata"]) if current_metadata.get("grounding_metadata") else None,
                partial=ChromaMemoryService._parse_metadata_field_value(current_metadata.get("partial")),
                turn_complete=ChromaMemoryService._parse_metadata_field_value(current_metadata.get("turn_complete")),
                error_code=ChromaMemoryService._parse_metadata_field_value(current_metadata.get("error_code"), type_caster=int),
                error_message=current_metadata.get("error_message"),
                interrupted=ChromaMemoryService._parse_metadata_field_value(current_metadata.get("interrupted")),
                custom_metadata=ChromaMemoryService._parse_metadata_field_value(current_metadata.get("custom_metadata"), is_json_load=True)
            )
                
            if session_id not in session_events:
                session_events[session_id] = []
            session_events[session_id].append(event)
        
        # Create memory results
        memory_results = []
        for session_id, events in session_events.items():
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            memory_results.append(
                MemoryResult(session_id=session_id, events=sorted_events)
            )
        
        return SearchMemoryResponse(memories=memory_results)

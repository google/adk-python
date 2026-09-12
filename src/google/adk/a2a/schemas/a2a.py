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

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class AgentResponseStatus(str, Enum):
    SUCCESS = 'SUCCESS'
    ERROR = 'ERROR'
    ELICITATION_REQUIRED = 'ELICITATION_REQUIRED'

class ElicitationData(BaseModel):
    question: str = Field(..., description="The clarification question to ask the user.")
    options: Optional[List[str]] = Field(None, description="Optional list of selectable options for the user.")
    missing_entities: List[str] = Field(..., description="List of parameters or entities that are missing or ambiguous.")
    context_snapshot: Optional[dict] = Field(None, description="Snapshot of the state to be rehydrated.")

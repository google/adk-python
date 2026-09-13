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

"""LiveKit integration.

Bridges a LiveKit room to an ADK live agent, giving an unmodified agent
telephony (SIP/PSTN), WebRTC, and Unity/gaming ingress. Install with:
pip install "google-adk[livekit]"
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
  from ._livekit_call import current_call
  from ._livekit_call import LiveKitCall
  from ._livekit_runner import DATA_TOPIC
  from ._livekit_runner import LiveKitRunner
  from ._livekit_runner import LK_CHAT_TOPIC
  from ._livekit_tools import end_call
  from ._livekit_tools import send_dtmf
  from ._livekit_tools import transfer_call
  from ._livekit_toolset import LiveKitToolset
  from ._transcripts import LK_TRANSCRIPTION_TOPIC

_lazy_imports = {
    "DATA_TOPIC": "._livekit_runner",
    "LK_CHAT_TOPIC": "._livekit_runner",
    "LK_TRANSCRIPTION_TOPIC": "._transcripts",
    "LiveKitCall": "._livekit_call",
    "LiveKitRunner": "._livekit_runner",
    "LiveKitToolset": "._livekit_toolset",
    "current_call": "._livekit_call",
    "end_call": "._livekit_tools",
    "send_dtmf": "._livekit_tools",
    "transfer_call": "._livekit_tools",
}

__all__ = sorted(_lazy_imports)


def __getattr__(name: str) -> typing.Any:
  if name in _lazy_imports:
    import importlib

    module = importlib.import_module(_lazy_imports[name], __name__)
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return list(__all__)

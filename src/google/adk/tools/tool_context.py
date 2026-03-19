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

from __future__ import annotations

# Keep CallbackContext for backward compatibility
from ..agents.callback_context import CallbackContext
from ..agents.context import Context
# Keep AuthCredential for backward compatibility
from ..auth.auth_credential import AuthCredential
# Keep AuthHandler for backward compatibility
from ..auth.auth_handler import AuthHandler
# Keep AuthConfig for backward compatibility
from ..auth.auth_tool import AuthConfig
# Keep ToolConfirmation for backward compatibility
from .tool_confirmation import ToolConfirmation

# ToolContext is unified into Context
ToolContext = Context

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

from __future__ import annotations

import inspect
from typing import Any
from typing import Callable

from typing_extensions import override

from ..auth.auth_credential import AuthCredential
from ..auth.auth_tool import AuthConfig
from ..auth.credential_manager import CredentialManager
from ..utils.feature_decorator import experimental
from .function_tool import FunctionTool
from .tool_context import ToolContext


@experimental
class AuthenticatedFunctionTool(FunctionTool):
  """A tool that requires authentication and make sure it's already
  authenticated before actually getting called. (Experimental)"""

  def __init__(
      self,
      *,
      func: Callable[..., Any],
      auth_config: AuthConfig = None,
      unauthenticated_response: str = None,
  ):
    super().__init__(func=func)
    self._ignore_params.append("credential")

    if auth_config and auth_config.auth_scheme:
      self._credentials_manager = CredentialManager(auth_config=auth_config)
    else:
      self._credentials_manager = None
    self._no_credential_message = unauthenticated_response

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    credential = None
    if self._credentials_manager:
      credential = await self._credentials_manager.load_auth_credential(
          tool_context
      )
      if not credential:
        await self._credentials_manager.request_credential(tool_context)
        return self._no_credential_message or "Pending User Authorization."

    return await self._run_async_impl(
        args=args, tool_context=tool_context, credential=credential
    )

  async def _run_async_impl(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
      credential: AuthCredential,
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if "credential" in signature.parameters:
      args_to_call["credential"] = credential
    return await super().run_async(args=args_to_call, tool_context=tool_context)

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
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

from typing_extensions import override

from ..auth.auth_credential import AuthCredential
from ..auth.auth_tool import AuthConfig
from ..auth.credential_manager import CredentialManager
from ..utils.feature_decorator import experimental
from .function_tool import FunctionTool
from .tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


@experimental
class AuthenticatedFunctionTool(FunctionTool):
  """A FunctionTool that handles authentication before the actual tool logic
  gets called. Functions can accept a special `credential` argument which is the
  credential ready for use.(Experimental)
  """

  def __init__(
      self,
      *,
      func: Callable[..., Any],
      auth_config: AuthConfig = None,
      response_for_auth_required: Optional[Union[dict[str, Any], str]] = None,
  ):
    """Initializes the AuthenticatedFunctionTool.

    Args:
        func: The function to be called.
        auth_config: The authentication configuration.
        response_for_auth_required: The response to return when the tool is
          requesting auth credential from the client. There could be two case,
          the tool doesn't configure any credentials
          (auth_config.raw_auth_credential is missing) or the credentials
          configured is not enough to authenticate the tool (e.g. an OAuth
          client id and client secrect is configured.) and needs client input
          (e.g. client need to involve the end user in an oauth flow and get
          back the oauth response.)
    """
    super().__init__(func=func)
    self._ignore_params.append("credential")

    if auth_config and auth_config.auth_scheme:
      self._credentials_manager = CredentialManager(auth_config=auth_config)
    else:
      logger.warning(
          "auth_config or auth_config.auth_scheme is missing. Will skip"
          " authentication.Using FunctionTool instead if authentication is not"
          " required."
      )
      self._credentials_manager = None
    self._response_for_auth_required = response_for_auth_required

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    credential = None
    if self._credentials_manager:
      credential = await self._credentials_manager.get_auth_credential(
          tool_context
      )
      if not credential:
        await self._credentials_manager.request_credential(tool_context)
        return self._response_for_auth_required or "Pending User Authorization."

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

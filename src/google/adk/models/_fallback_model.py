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

"""A model that falls back to backup models when a call fails."""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextlib import AsyncExitStack
import copy
import logging
import sys
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import ClassVar
from typing import NamedTuple
from typing import TYPE_CHECKING
import weakref

from google.genai import types
from google.genai.errors import APIError
from pydantic import Field
from pydantic import model_validator
from pydantic import PrivateAttr
from typing_extensions import override

from ..features import experimental
from ..features import FeatureName
from ..utils.context_utils import Aclosing
from ._capabilities import LlmCapabilities
from .base_llm import BaseLlm
from .llm_response import LlmResponse
from .registry import LLMRegistry

if TYPE_CHECKING:
  from .base_llm_connection import BaseLlmConnection
  from .llm_request import LlmRequest

logger = logging.getLogger('google_adk.' + __name__)


def _model_name(entry: str | BaseLlm) -> str:
  """Returns the model name for an entry of :attr:`FallbackModel.models`."""
  return entry if isinstance(entry, str) else entry.model


_SNAPSHOT_PRIVATE = (
    '_dynamic_instructions',
    '_is_managed_agent',
    '_has_static_instruction',
    '_static_instruction_prefix_end_index',
)
"""The private attributes of ``LlmRequest`` a rollback restores.

Named rather than taken from ``__private_attributes__`` wholesale: copying
whatever happens to be there is what made ``tools_dict`` crash the wrapper, and
a private attribute added later could hold a lock or a session just as easily.
"""

_UNSNAPSHOTTED_PRIVATE: tuple[str, ...] = ()
"""The private attributes of ``LlmRequest`` a rollback deliberately leaves.

Somewhere for an attribute that must not be copied — one holding a lock, a
session or a connection — or that a model never edits and so needs no
restoring. Say which, and why, when adding one.

Between them the two tuples have to name every private attribute of
``LlmRequest``; ``test_every_private_attribute_is_accounted_for`` fails
otherwise. Adding an attribute over in ``llm_request.py`` therefore has to be
answered here, one way or the other, instead of silently widening what this
copies.
"""


class _RequestSnapshot(NamedTuple):
  """The parts of a request that a model edits, kept for rollback."""

  contents: list[types.Content]
  config: types.GenerateContentConfig
  live_connect_config: types.LiveConnectConfig
  private: dict[str, Any]

  @classmethod
  def of(cls, llm_request: LlmRequest) -> _RequestSnapshot:
    """Captures `llm_request`, copying only what a model can change.

    Copying the whole request is not an option: ``tools_dict`` holds live tool
    objects, and an MCP tool reaches a ``threading.Lock`` that deep copy
    refuses. Tools are a registry the models read, never something they edit,
    so they are left out along with everything else outside these four.
    """
    return cls(
        contents=[
            content.model_copy(deep=True) for content in llm_request.contents
        ],
        config=llm_request.config.model_copy(deep=True),
        live_connect_config=(
            llm_request.live_connect_config.model_copy(deep=True)
            if llm_request.live_connect_config
            else types.LiveConnectConfig()
        ),
        private=copy.deepcopy(
            {name: getattr(llm_request, name) for name in _SNAPSHOT_PRIVATE}
        ),
    )

  def restore(self, llm_request: LlmRequest) -> None:
    """Rolls `llm_request` back to this snapshot, in place.

    Args:
      llm_request: The request to roll back. The caller holds it, so it has to
        be the same object afterwards, not a replacement.
    """
    llm_request.contents = self.contents
    llm_request.config = self.config
    llm_request.live_connect_config = self.live_connect_config
    for name, value in self.private.items():
      setattr(llm_request, name, value)


def _resumption_handle(llm_request: LlmRequest) -> str | None:
  """The live session-resumption handle on `llm_request`, if it carries one."""
  live_config = llm_request.live_connect_config
  if live_config is None or live_config.session_resumption is None:
    return None
  return live_config.session_resumption.handle


_LITELLM_MISREPORTED_500S = (
    # Raised for a refused connection, a DNS failure or a dropped socket.
    'APIConnectionError',
    # Raised when litellm's own post-call rules reject a response that did
    # arrive.
    'APIResponseValidationError',
)


def _is_litellm_misreported_500(error: BaseException) -> bool:
  """Whether `error` is a litellm error whose status 500 describes nothing.

  litellm hard-codes ``status_code = 500`` on both of
  :data:`_LITELLM_MISREPORTED_500S`, and neither is a server error. One never
  reached the service; the other reports that a response did arrive and then
  failed a client-side check. Taking either at face value would fail over and
  re-send a prompt the service may already have charged for and acted on.

  litellm is an optional dependency, so it is consulted only when the process
  has already imported it. Nothing else can raise its exceptions.

  Args:
    error: The exception raised by a delegate model.

  Returns:
    True if the error is one whose 500 should not be read as a server error.
  """
  exceptions = getattr(sys.modules.get('litellm'), 'exceptions', None)
  if exceptions is None:
    return False
  misreported = tuple(
      candidate
      for name in _LITELLM_MISREPORTED_500S
      if isinstance(candidate := getattr(exceptions, name, None), type)
  )
  return isinstance(error, misreported)


def _status_code(error: BaseException) -> int | None:
  """Extracts the HTTP status code from a provider error, if it carries one.

  Every provider SDK spells this differently, and ``FallbackModel`` wraps all
  of them, so each shape is tried in turn.

  Args:
    error: The exception raised by a delegate model.

  Returns:
    The HTTP status code, or None if the error does not carry one.
  """
  # google-genai collapses every 4xx into ClientError and every 5xx into
  # ServerError, and carries the status on `code`.
  if isinstance(error, APIError):
    return error.code
  if _is_litellm_misreported_500(error):
    return None
  # litellm, openai and anthropic all expose `status_code` directly.
  status_code = getattr(error, 'status_code', None)
  if isinstance(status_code, int):
    return status_code
  # httpx, which ApigeeLlm raises through, nests it under the response.
  status_code = getattr(getattr(error, 'response', None), 'status_code', None)
  return status_code if isinstance(status_code, int) else None


@experimental(FeatureName.FALLBACK_MODEL)
class FallbackModel(BaseLlm):
  """Tries a sequence of models in order, moving on when one fails.

  Each model is tried exactly once. Retrying a single model is a separate
  concern, handled by that model's own retry configuration (for example
  ``Gemini(model=..., retry_options=...)``), so that a single failure is not
  retried twice over by two layers.

  Usage::

      from google.adk import Agent
      from google.adk.models import FallbackModel

      agent = Agent(
          name='reliable_agent',
          model=FallbackModel(
              models=['gemini-3.1-pro-preview', 'gemini-3.5-flash'],
          ),
      )

  Entries may also be model instances, which is how a backup gets settings of
  its own::

      FallbackModel(
          models=[
              'gemini-3.1-pro-preview',
              Gemini(model='gemini-3.5-flash', retry_options=...),
          ],
      )

  Only failures that carry one of :attr:`retriable_status_codes` move on to
  the next model. Anything else propagates to the caller unchanged, including
  a failure that never reached the service: those carry no status, except for
  the litellm errors that report one they did not get, which are recognised
  and treated as carrying none.

  Once a model has yielded its first response the turn belongs to it: a later
  failure propagates rather than falling back, because the caller already
  holds part of that model's output and splicing a second model onto it would
  corrupt the turn. This is what streaming needs, since a streaming call can
  fail partway through a turn after emitting several chunks. A non-streaming
  call yields once and so almost always fails before that point, leaving it
  free to fall back.

  Live connections follow the same rule at their own boundary. A model that
  fails to connect is passed over for the next one, since nothing has crossed
  the connection yet; a failure once the session is open belongs to that
  session and propagates.
  """

  DEFAULT_STATUS_CODES: ClassVar[frozenset[int]] = frozenset({
      429,  # Too many requests.
      500,  # Internal server error.
      502,  # Bad gateway.
      503,  # Service unavailable.
      504,  # Gateway timeout.
  })
  """HTTP status codes that move the request on to the next model by default.

  This is the set ADK already retries on — see ``_RETRY_HTTP_STATUS_CODES`` in
  ``evaluation._retry_options_utils`` and the defaults in ``ApigeeLlm`` — minus
  408. Those two retry the *same* model, where a request that timed out costs
  one more attempt at worst. Moving to a *different* model is a heavier
  commitment: 408 does not say whether the request was processed before the
  clock ran out, and litellm reports even a client-side timeout as 408, so
  failing over on it risks paying for and acting on one prompt twice. Add it
  back for a provider whose 408 is known to mean the request was dropped.
  """

  models: list[str | BaseLlm] = Field(min_length=1)
  """The models to try, in order. The first entry is the primary model."""

  retriable_status_codes: frozenset[int] = DEFAULT_STATUS_CODES
  """The HTTP status codes that cause the next model to be tried."""

  model: str = ''
  """The primary model's name. Derived from :attr:`models`; do not set it.

  ``BaseLlm`` declares this and ADK reads it all over — to name the model on a
  span, to fill in ``LlmRequest.model`` — so it cannot simply be dropped here.
  Configure this model with :attr:`models` instead; passing ``model`` to the
  constructor is rejected, because a name given there would be reported but
  would not select anything. Like every other field on a ``BaseLlm``, it is
  not re-validated when assigned after construction.
  """

  _resolved: dict[str, BaseLlm] = PrivateAttr(default_factory=dict)

  _live_owner: list[tuple[weakref.ref[LlmRequest], int]] = PrivateAttr(
      default_factory=list
  )
  """Which model opened the live session carried by each request.

  The live flow builds one request per ``run_live`` and reuses it for every
  reconnect, so the request identifies the session. Keying on the request
  rather than on this instance is what keeps concurrent live sessions on one
  agent from overwriting each other's entry, and the references are weak so an
  entry goes away with the session that owned it.

  Not synchronised: sharing one ``FallbackModel`` between live sessions on
  *different* event loops can drop an entry, which costs that session its pin
  and sends it back to matching on the name.
  """

  @model_validator(mode='after')
  def _derive_model_name_from_primary(self) -> FallbackModel:
    primary_name = _model_name(self.models[0])
    # An equal value round-trips model_dump(); anything else is a caller who
    # meant `models` and would otherwise be silently ignored.
    if self.model and self.model != primary_name:
      raise ValueError(
          'FallbackModel.model is derived from the first entry of `models`'
          f' and cannot be set directly: got {self.model!r}, expected'
          f' {primary_name!r}. List the models to try in `models`.'
      )
    self.model = primary_name
    return self

  def _delegate(self, entry: str | BaseLlm) -> BaseLlm:
    """Resolves an entry of :attr:`models` to a model instance."""
    if isinstance(entry, BaseLlm):
      return entry
    if entry not in self._resolved:
      self._resolved[entry] = LLMRegistry.new_llm(entry)
    return self._resolved[entry]

  def _should_fall_back(self, error: BaseException) -> bool:
    """Whether `error` is one that the next model should be tried for."""
    status_code = _status_code(error)
    return (
        status_code is not None and status_code in self.retriable_status_codes
    )

  @property
  @override
  def capabilities(self) -> LlmCapabilities:
    """The primary model's capabilities.

    A backup that reports different capabilities does not change what the
    request was built for, since the request is built before any call is made
    and so before it is known which model will serve it.
    """
    return self._delegate(self.models[0]).capabilities

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content, trying each model in turn until one succeeds.

    Args:
      llm_request: The request to send.
      stream: Whether to stream the response.

    Yields:
      The responses from the first model that succeeds.

    Raises:
      Exception: The last model's error, if every model failed. Also any error
        that is not retriable, and any error raised once the turn's first
        response has been yielded, which a streaming call can do partway
        through a turn.
    """
    last_index = len(self.models) - 1

    for index, entry in enumerate(self.models):
      # Outside the try on purpose: a delegate that cannot even be built is a
      # configuration error, and should be loud rather than fallen back from.
      delegate = self._delegate(entry)
      # Models edit the request in place before sending it — appending a user
      # turn, preprocessing tools — so a failed attempt would otherwise hand
      # its edits to the next model as if the caller had written them. Keep a
      # snapshot while another model could still be tried; whichever model
      # succeeds keeps its edits, which is what traces should show.
      pristine = (
          _RequestSnapshot.of(llm_request) if index < last_index else None
      )
      # Models read the name off the request, which the flow filled in from
      # this wrapper. Point it at the delegate or it calls the wrong model.
      llm_request.model = delegate.model
      response_yielded = False
      try:
        # Aclosing so that a caller who stops early — a callback that raises,
        # a client that goes away — closes the delegate's stream rather than
        # leaving the provider connection to the loop's finaliser. Every other
        # link in this chain does the same.
        async with Aclosing(
            delegate.generate_content_async(llm_request, stream)
        ) as agen:
          async for llm_response in agen:
            response_yielded = True
            yield llm_response
        return
      # Broad by necessity: every provider SDK raises its own error type and
      # this wraps all of them. `_should_fall_back` narrows it immediately,
      # and anything it rejects propagates untouched.
      except Exception as error:  # pylint: disable=broad-except
        # `response_yielded` comes first: a streaming call can fail partway
        # through a turn, and once part of this model's output has reached the
        # caller, falling back would splice two models into one turn.
        if response_yielded or not self._should_fall_back(error):
          raise
        # Non-None from here on: `_should_fall_back` matched it against
        # `retriable_status_codes`.
        status_code = _status_code(error)
        if index == last_index:
          # Every model has failed. Raising hands the last provider's error to
          # the flow, where `LlmAgent.on_model_error_callback` can turn it into
          # a response rather than end the invocation.
          raise
        logger.warning(
            'Model %s failed with status %s; falling back to the next model.',
            delegate.model,
            status_code,
        )
        if pristine is not None:
          pristine.restore(llm_request)

  @override
  @asynccontextmanager
  async def connect(
      self, llm_request: LlmRequest
  ) -> AsyncIterator[BaseLlmConnection]:
    """Opens a live connection, trying each model until one connects.

    Failing over is possible here for the same reason it is before a turn's
    first response: nothing has crossed the connection yet, so no session
    state is lost by handing the attempt to another model. Once a connection
    is open the session belongs to it — a backup cannot resume a
    bidirectional session already under way — so a failure after that point
    reaches the caller unchanged.

    A reconnect is the exception. A request carrying a session-resumption
    handle is pinned to the model that issued it, because replaying that
    handle against a different model resumes nothing; the failure reaches the
    caller so it can decide whether to start a fresh session instead.

    Args:
      llm_request: The request to open the connection with.

    Yields:
      The connection of the first model that accepts one. Errors are treated
      as in :meth:`generate_content_async`: a retriable status moves to the
      next model, anything else propagates.

    Raises:
      ValueError: If the request carries a session-resumption handle whose
        model does not name exactly one entry of :attr:`models`.
      Exception: The last model's error, if no model accepted a connection.
    """
    candidates = self._candidate_indexes(llm_request)
    last_position = len(candidates) - 1

    # The stack closes whichever connection was opened, on the normal path and
    # on one carrying an exception out of the caller's `async with` body. A
    # model that failed to connect registered nothing, so it needs no cleanup.
    async with AsyncExitStack() as stack:
      for position, index in enumerate(candidates):
        delegate = self._delegate(self.models[index])
        # Connecting edits the request as much as a turn does — Gemini writes
        # speech_config, system_instruction, tools, thinking_config,
        # safety_settings and http_options onto it. Some of those it writes
        # only when the model has them, so without a rollback a backup that
        # has none would inherit the primary's.
        pristine = (
            _RequestSnapshot.of(llm_request)
            if position < last_position
            else None
        )
        # Live models read the name off the request too, and Gemini rejects a
        # request that has none.
        llm_request.model = delegate.model
        try:
          connection = await stack.enter_async_context(
              delegate.connect(llm_request)
          )
        except Exception as error:  # pylint: disable=broad-except
          if position == last_position or not self._should_fall_back(error):
            raise
          logger.warning(
              'Model %s failed to connect with status %s; falling back to the'
              ' next model.',
              delegate.model,
              _status_code(error),
          )
          if pristine is not None:
            pristine.restore(llm_request)
          continue
        self._remember_live_owner(llm_request, index)
        yield connection
        return

  def _remember_live_owner(self, llm_request: LlmRequest, index: int) -> None:
    """Records that `index` opened the live session carried by `llm_request`."""
    self._live_owner = [
        (request, owner)
        for request, owner in self._live_owner
        if request() is not None and request() is not llm_request
    ]
    self._live_owner.append((weakref.ref(llm_request), index))

  def _recall_live_owner(self, llm_request: LlmRequest) -> int | None:
    """The model that opened the session on `llm_request`, if this one did.

    Matching is by object identity, not equality: two live sessions can hold
    requests that compare equal while belonging to different servers.
    """
    for request, owner in self._live_owner:
      if request() is llm_request:
        return owner
    return None

  def _owns_session(self, entry: str | BaseLlm, model_name: str | None) -> bool:
    """Whether `entry` is the model that `model_name` refers to."""
    if model_name is None:
      return False
    if _model_name(entry) == model_name:
      return True
    # A prefixed entry such as 'gemini:gemini-3.5-flash' loses its prefix when
    # the registry builds it, so the entry string and the delegate's name
    # differ. Only an entry that has already been built can have opened the
    # session, so the cache answers this without instantiating anything new.
    if isinstance(entry, str):
      built = self._resolved.get(entry)
      return built is not None and built.model == model_name
    return False

  def _candidate_indexes(self, llm_request: LlmRequest) -> list[int]:
    """The entries of :attr:`models` to attempt a connection with, in order.

    Normally every entry. A request carrying a session-resumption handle is a
    reconnect, though, and that handle was issued by whichever model opened
    the session — replaying it against a different one resumes nothing. Such a
    request is pinned to its own model, and failure reaches the caller so it
    can decide whether to start a fresh session.

    The owner is whichever model :attr:`_live_owner` recorded against this
    request. That follows the session rather than the model's name, so two
    entries may share a name — one model behind two keys or regions — and
    still be told apart.

    A handle on a request with no record came from an earlier run, since the
    live flow builds one request per run. Only the name is left to go on then,
    and the flow has already reset it to the agent's own, so such a resume
    pins to the primary whichever model opened the session.

    A handle whose owner cannot be identified at all is an error rather than a
    reason to fall back: trying the models in turn would offer one model's
    handle to every other one, which is the outcome the pin exists to prevent.

    Args:
      llm_request: The request to open the connection with.

    Returns:
      The indexes of the models to try, in order.

    Raises:
      ValueError: If the request carries a handle and its model does not name
        exactly one entry of :attr:`models`.
    """
    if _resumption_handle(llm_request) is None:
      return list(range(len(self.models)))

    remembered = self._recall_live_owner(llm_request)
    if remembered is not None:
      return [remembered]

    # Nothing remembered: the handle was issued under a different request, so
    # it came from an earlier run. All that is left to go on is the name the
    # flow put on the request, which is the agent's own — see above.
    owner = [
        index
        for index, entry in enumerate(self.models)
        if self._owns_session(entry, llm_request.model)
    ]
    if len(owner) != 1:
      raise ValueError(
          'Cannot resume a live session: the request carries a'
          f' session-resumption handle but names model {llm_request.model!r},'
          f" which matches {len(owner)} of this FallbackModel's models. A"
          ' handle is only meaningful to the model that issued it, and the'
          ' owner is identified by name, so two entries reporting the same'
          ' name cannot be told apart — one model behind two endpoints or'
          ' keys, for instance. Clear the handle to start a fresh session, or'
          ' keep such a pair out of a live agent that resumes sessions.'
      )
    return owner

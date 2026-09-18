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

import contextlib
import hashlib
import logging
from typing import Final
from typing import Iterator
from typing import Literal
from typing import NamedTuple
from typing import Optional

from pydantic import BaseModel

from ....auth.auth_credential import AuthCredential
from ....auth.auth_credential import AuthCredentialTypes
from ....auth.auth_schemes import AuthScheme
from ....auth.auth_schemes import AuthSchemeType
from ....auth.auth_tool import _stable_model_digest
from ....auth.auth_tool import AuthConfig
from ....auth.refresher.oauth2_credential_refresher import OAuth2CredentialRefresher
from ...tool_context import ToolContext
from ..auth.credential_exchangers.auto_auth_credential_exchanger import AutoAuthCredentialExchanger
from ..auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
from ..auth.credential_exchangers.base_credential_exchanger import BaseAuthCredentialExchanger

logger = logging.getLogger("google_adk." + __name__)

AuthPreparationState = Literal["pending", "done"]

# Outcome of ToolAuthHandler.handle_unauthorized_error():
#   "refreshed": a new access token was obtained and stored; retry the call.
#   "reauth_requested": the stale credential was evicted and the user was asked
#     to re-authorize; the caller should report a pending result.
#   "reauth_limit_reached": the user has already been asked to re-authorize
#     this credential and the API still rejected it; the caller should surface
#     the API error and tell the model not to retry.
#   "failed": nothing could be done; the caller should surface the API error.
UnauthorizedRecoveryState = Literal[
    "refreshed", "reauth_requested", "reauth_limit_reached", "failed"
]

# How many times one credential may interrupt the user with a re-authorization
# request before recovery stops asking. The budget refills as soon as a call
# using that credential succeeds, so an ordinary expiry can ask again later in
# the session; it is only an endpoint that keeps rejecting a freshly authorized
# token that runs out. Set to 1 because a request the user has already
# answered, against an endpoint that still returns 401, has proven it does not
# help.
_MAX_REAUTH_REQUESTS: Final[int] = 1

# How many times recovery may refresh one credential before it gives up on
# refreshing. Spent only by a refresh that produced a new token but did not
# lead to a working call, and refilled as soon as one does, so an ordinary
# expiry is always refreshed; it is only a credential whose refreshes keep
# producing tokens the API still rejects that runs out.
_MAX_RECOVERY_REFRESHES: Final[int] = 1


class AuthPreparationResult(BaseModel):
  """Result of the credential preparation process."""

  state: AuthPreparationState
  auth_scheme: Optional[AuthScheme] = None
  auth_credential: Optional[AuthCredential] = None


class ToolContextCredentialStore:
  """Handles storage and retrieval of credentials within a ToolContext."""

  def __init__(self, tool_context: ToolContext):
    self.tool_context = tool_context

  def _legacy_stable_digest(self, text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

  def _get_legacy_credential_key(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> str:
    if auth_credential and auth_credential.oauth2:
      auth_credential = auth_credential.model_copy(deep=True)
      if auth_credential.oauth2:
        auth_credential.oauth2.auth_uri = None
        auth_credential.oauth2.state = None
        auth_credential.oauth2.auth_response_uri = None
        auth_credential.oauth2.auth_code = None
        auth_credential.oauth2.access_token = None
        auth_credential.oauth2.refresh_token = None
        auth_credential.oauth2.expires_at = None
        auth_credential.oauth2.expires_in = None
        auth_credential.oauth2.redirect_uri = None
    scheme_name = (
        f"{auth_scheme.type_.name}_{self._legacy_stable_digest(auth_scheme.model_dump_json())}"
        if auth_scheme
        else ""
    )
    credential_name = (
        f"{auth_credential.auth_type.value}_{self._legacy_stable_digest(auth_credential.model_dump_json())}"
        if auth_credential
        else ""
    )
    return f"{scheme_name}_{credential_name}_existing_exchanged_credential"

  def get_credential_key(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> str:
    """Generates a unique key for the given auth scheme and credential."""

    if auth_credential and auth_credential.oauth2:
      auth_credential = auth_credential.model_copy(deep=True)
      if auth_credential.oauth2:
        auth_credential.oauth2.auth_uri = None
        auth_credential.oauth2.state = None
        auth_credential.oauth2.auth_response_uri = None
        auth_credential.oauth2.auth_code = None
        auth_credential.oauth2.access_token = None
        auth_credential.oauth2.refresh_token = None
        auth_credential.oauth2.expires_at = None
        auth_credential.oauth2.expires_in = None
        auth_credential.oauth2.redirect_uri = None
    scheme_name = (
        f"{auth_scheme.type_.name}_{_stable_model_digest(auth_scheme)}"
        if auth_scheme
        else ""
    )
    credential_name = (
        f"{auth_credential.auth_type.value}_{_stable_model_digest(auth_credential)}"
        if auth_credential
        else ""
    )
    # no need to prepend temp: namespace, session state is a copy, changes to
    # it won't be persisted , only changes in event_action.state_delta will be
    # persisted. temp: namespace will be cleared after current run. but tool
    # want access token to be there stored across runs

    return f"{scheme_name}_{credential_name}_existing_exchanged_credential"

  def get_credential(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> Optional[AuthCredential]:
    if not self.tool_context:
      return None

    token_key = self.get_credential_key(auth_scheme, auth_credential)
    # TODO try not to use session state, this looks a hacky way, depend on
    # session implementation, we don't want session to persist the token,
    # meanwhile we want the token shared across runs.
    serialized_credential = self.tool_context.state.get(token_key)
    if serialized_credential:
      return AuthCredential.model_validate(serialized_credential)

    legacy_key = self._get_legacy_credential_key(auth_scheme, auth_credential)
    if legacy_key == token_key:
      return None
    serialized_legacy_credential = self.tool_context.state.get(legacy_key)
    if not serialized_legacy_credential:
      return None

    # Migrate to the current key for future lookups.
    self.tool_context.state[token_key] = serialized_legacy_credential
    return AuthCredential.model_validate(serialized_legacy_credential)

  def store_credential(
      self,
      key: str,
      auth_credential: Optional[AuthCredential],
  ):
    if self.tool_context:
      self.tool_context.state[key] = auth_credential.model_dump(
          exclude_none=True
      )

  def remove_credential(self, key: str) -> None:
    """Removes a stored credential.

    The key is set to None rather than deleted: None serializes cleanly across
    every session backend and reads back as "no credential", while a deletion
    has no representation in a state delta. Keys that are not present are left
    alone, so eviction never creates an entry that was never stored.
    """
    if not self.tool_context:
      return
    if key not in self.tool_context.state:
      return
    self.tool_context.state[key] = None


class _RefreshOutcome(NamedTuple):
  credential: AuthCredential
  new_access_token: bool
  rotated_refresh_token: bool


class ToolAuthHandler:
  """Handles the preparation and exchange of authentication credentials for tools."""

  def __init__(
      self,
      tool_context: ToolContext,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
      credential_exchanger: Optional[BaseAuthCredentialExchanger] = None,
      credential_store: Optional["ToolContextCredentialStore"] = None,
      *,
      credential_key: Optional[str] = None,
  ):
    self.tool_context = tool_context
    self.auth_scheme = (
        auth_scheme.model_copy(deep=True) if auth_scheme else None
    )
    self.auth_credential = (
        auth_credential.model_copy(deep=True) if auth_credential else None
    )
    self._credential_key = credential_key
    self.credential_exchanger = (
        credential_exchanger or AutoAuthCredentialExchanger()
    )
    self.credential_store = credential_store
    self.should_store_credential = True

  def _get_credential_key_override(self) -> Optional[str]:
    """Returns a user-provided credential_key if available."""
    if self._credential_key:
      return self._credential_key

    for obj in (self.auth_credential, self.auth_scheme):
      if not obj or not obj.model_extra:
        continue
      for key in ("credential_key", "credentialKey"):
        value = obj.model_extra.get(key)
        if isinstance(value, str) and value:
          return value

    return None

  def _build_auth_config(self) -> AuthConfig:
    return AuthConfig(
        auth_scheme=self.auth_scheme,
        raw_auth_credential=self.auth_credential,
        credential_key=self._get_credential_key_override(),
    )

  @classmethod
  def from_tool_context(
      cls,
      tool_context: ToolContext,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
      credential_exchanger: Optional[BaseAuthCredentialExchanger] = None,
      *,
      credential_key: Optional[str] = None,
  ) -> "ToolAuthHandler":
    """Creates a ToolAuthHandler instance from a ToolContext."""
    credential_store = ToolContextCredentialStore(tool_context)
    return cls(
        tool_context,
        auth_scheme,
        auth_credential,
        credential_key=credential_key,
        credential_exchanger=credential_exchanger,
        credential_store=credential_store,
    )

  async def _get_existing_credential(
      self,
  ) -> Optional[AuthCredential]:
    """Checks for and returns an existing, exchanged credential."""
    if self.credential_store:
      existing_credential = self.credential_store.get_credential(
          self.auth_scheme, self.auth_credential
      )
      if existing_credential:
        if existing_credential.oauth2:
          # The helper persists what it obtains, so the next invocation reads
          # the new tokens instead of the stale pre-refresh ones. Without that,
          # providers that rotate refresh tokens on each refresh (e.g.
          # Salesforce, many OIDC providers) fail because the old refresh
          # token has already been invalidated.
          outcome = await self._refresh_and_store(existing_credential)
          existing_credential = outcome.credential
        return existing_credential
    return None

  def _exchange_credential(
      self, auth_credential: AuthCredential
  ) -> Optional[AuthPreparationResult]:
    """Handles an OpenID Connect authorization response."""

    exchanged_credential = None
    try:
      exchanged_credential = self.credential_exchanger.exchange_credential(
          self.auth_scheme, auth_credential
      )
    except Exception as e:
      logger.error("Failed to exchange credential: %s", e)
    return exchanged_credential

  def _store_credential(self, auth_credential: AuthCredential) -> None:
    """stores the auth_credential."""

    if self.credential_store:
      key = self.credential_store.get_credential_key(
          self.auth_scheme, self.auth_credential
      )
      self.credential_store.store_credential(key, auth_credential)

  async def _refresh_and_store(
      self,
      credential: AuthCredential,
      *,
      force_expiry: bool = False,
  ) -> _RefreshOutcome:
    """Refreshes an OAuth2 credential and persists what comes back.

    Both the scheduled refresh in _get_existing_credential and the reactive one
    in handle_unauthorized_error go through here, so the refresh token survives
    either way. Keeping them apart is what let the scheduled path drop it while
    the reactive path preserved it.

    Args:
      credential: The stored credential to refresh. The refresher mutates it in
        place.
      force_expiry: Whether to mark the credential expired before refreshing.
        The refresher only contacts the token endpoint for a credential that
        already reads as expired, so a caller reacting to a 401, where the
        server has rejected a token the credential still believes in, has to say
        so explicitly. A caller refreshing on schedule leaves this False and
        lets the recorded expiry decide.

    Returns:
      A _RefreshOutcome naming the updated credential, whether a new access
      token was obtained, and whether the refresh token rotated. An unchanged
      access token means the provider gave us nothing new to call the API with,
      so whatever triggered the refresh would fail the same way again; a
      refresh token rotated alongside it is still persisted, since the old one
      is dead.
    """
    refresher = OAuth2CredentialRefresher()
    if not credential.oauth2:
      return _RefreshOutcome(credential, False, False)

    if force_expiry:
      # The value must be truthy, not merely non-None: authlib < 1.7 bails out
      # of is_expired() with "if not expires_at: return None", so 0 would read
      # as "no expiry known" and skip the refresh. A timestamp of 1 (Jan 1,
      # 1970 00:00:01 UTC) is expired under every version the
      # authlib>=1.6.6,<2 pin allows.
      credential.oauth2.expires_at = 1
      credential.oauth2.expires_in = 0
    elif not await refresher.is_refresh_needed(credential):
      return _RefreshOutcome(credential, False, False)

    old_access_token = credential.oauth2.access_token
    old_refresh_token = credential.oauth2.refresh_token
    try:
      refreshed = await refresher.refresh(credential, self.auth_scheme)
    except Exception as e:
      # Scoped to the token endpoint round trip, which is the only step here
      # expected to fail. Covering the store below as well would report a
      # rejected write as a failed refresh, and the caller would ask the user
      # to re-authorize a refresh token that had in fact just been spent -- and
      # under rotation the old one is dead by then, with the new credential
      # never persisted.
      logger.warning("OAuth2 token refresh failed: %s", e)
      return _RefreshOutcome(credential, False, False)

    # update_credential_with_tokens assigns tokens.get("refresh_token")
    # unconditionally, so a provider that does not rotate loses its refresh
    # token to None. RFC 6749 section 6 leaves issuing a new one optional and
    # keeps the submitted one valid until one is issued, so carry it forward.
    # Without this the credential ends up with nothing to refresh from and an
    # interactive re-authorization becomes the only way back.
    if refreshed.oauth2 and not refreshed.oauth2.refresh_token:
      refreshed.oauth2.refresh_token = old_refresh_token

    if not (refreshed.oauth2 and refreshed.oauth2.access_token):
      # Nothing usable came back, so there is nothing worth storing.
      return _RefreshOutcome(credential, False, False)

    new_access_token = refreshed.oauth2.access_token != old_access_token
    # A rotated refresh token has to be persisted even when the access token is
    # unchanged, which is what a provider returns when it refreshes before full
    # expiry or rotates on a sliding window. The authorization server
    # invalidates the old refresh token as it issues the new one, so a stored
    # copy still holding the old one has nothing left to refresh from.
    rotated_refresh_token = bool(
        refreshed.oauth2.refresh_token
        and refreshed.oauth2.refresh_token != old_refresh_token
    )

    if not new_access_token and not rotated_refresh_token:
      # The refresher reports failure by returning the credential it was given,
      # so tokens that have not moved are the only signal available. This is
      # also what keeps a forced expiry out of session state, which would
      # otherwise make every later call attempt a refresh that cannot work: the
      # forced value only survives a refresh that never reached the token
      # endpoint, and that is exactly the case where neither token moved.
      return _RefreshOutcome(credential, False, False)

    self._store_credential(refreshed)
    return _RefreshOutcome(refreshed, new_access_token, rotated_refresh_token)

  def _evict_credentials(self) -> None:
    """Evicts stored credentials from session state."""
    if not self.credential_store:
      return
    token_key = self.credential_store.get_credential_key(
        self.auth_scheme, self.auth_credential
    )
    self.credential_store.remove_credential(token_key)
    legacy_key = self.credential_store._get_legacy_credential_key(
        self.auth_scheme, self.auth_credential
    )
    if legacy_key != token_key:
      self.credential_store.remove_credential(legacy_key)

  def _recovery_scope(self) -> str:
    """Returns the identifier that 401 recovery bookkeeping is keyed by.

    The key names the credential, not the tool. Sibling tools of a toolset
    share one stored credential, so keying per tool would let them recover in
    parallel and refresh the same refresh token at once: under refresh token
    rotation the loser spends a token the winner has already invalidated, and
    then evicts the credential the winner just stored. Per-tool keying would
    also budget the user's re-authorization requests once per tool rather than
    once per broken credential, which is no bound at all for a toolset with
    thirty tools behind one credential.

    The credential key is a truncated SHA-256 digest computed with every token
    field cleared, and it is already the key this credential is stored under in
    session state, so reusing it here reveals nothing new.
    """
    if not self.credential_store:
      return ""
    return self.credential_store.get_credential_key(
        self.auth_scheme, self.auth_credential
    )

  def _recovery_lock_key(self) -> str:
    """Returns the key holding the recovery lock.

    In the temp namespace deliberately: the lock guards one call chain, and
    session services drop temp keys before writing, so a process that dies
    mid-recovery cannot leave the lock set and disable recovery for good.
    """
    return f"temp:_adk_reauth_attempted_{self._recovery_scope()}"

  def _reauth_budget_key(self) -> str:
    """Returns the key holding the re-authorization budget.

    Deliberately NOT in the temp namespace, which the session services strip
    from the delta before it reaches storage. The budget exists to remember
    across turns that the user has already answered a re-authorization request
    for this credential, and a turn boundary is exactly where a temp key is
    lost, so a temp budget would reset before it was ever consulted and bound
    nothing.

    The colon is load-bearing: State exempts any key containing one from
    state_schema validation, so an app that declares a schema does not have to
    declare this internal key.
    """
    return f"_adk_reauth_requests:{self._recovery_scope()}"

  def _refresh_budget_key(self) -> str:
    """Returns the key holding the reactive refresh budget.

    Outside the temp namespace for the same reason as the re-authorization
    budget: what it bounds is a sequence of calls, not a single call chain, and
    a temp key does not survive the turn boundary between them.
    """
    return f"_adk_recovery_refreshes:{self._recovery_scope()}"

  @contextlib.contextmanager
  def claim_recovery(self) -> Iterator[bool]:
    """Claims the exclusive right to recover this credential.

    The claim is released on exit, including on exception, so no caller can
    strand it and leave this credential unrecoverable for the rest of the
    invocation. Hold it across any retry of the API call, not just across the
    recovery itself: the retried call builds its own handler, which must find
    the claim taken so that a persistently failing endpoint cannot recurse.

    Synchronous on purpose. Nothing here awaits, and that is what makes the
    check-and-set below atomic, so an async context manager would advertise a
    suspension point that must not exist.

    Yields:
      True if the claim was taken and recovery may proceed, False if another
      caller is already recovering this credential.
    """
    acquired = self._acquire_recovery_lock()
    try:
      yield acquired
    finally:
      # Only the winner may clear the flag. A loser that released it would
      # hand a second caller the claim the winner still holds.
      if acquired:
        self._release_recovery_lock()

  def _acquire_recovery_lock(self) -> bool:
    """Sets the recovery flag for this credential, if no one else holds it.

    The flag lives in shared session state rather than in a local so that it
    serializes concurrent calls: no await separates the read from the write, so
    the check-and-set is atomic under asyncio and exactly one caller wins. The
    losers report the ordinary retryable API error and succeed when the model
    retries them, by which point the winner has stored a fresh token. Holding
    the lock across the retry is also what bounds recursion, since the retried
    call builds a new handler that finds the lock already held.

    Returns:
      True if the lock was acquired, False if recovery is already in progress
      for this credential.
    """
    if not self.tool_context or not self.credential_store:
      return False
    lock_key = self._recovery_lock_key()
    if self.tool_context.state.get(lock_key, False):
      return False
    self.tool_context.state[lock_key] = True
    return True

  def _release_recovery_lock(self) -> None:
    """Clears the flag set by _acquire_recovery_lock().

    Reset rather than removed: State has no __delitem__, and a deletion cannot
    be expressed in a state delta. Leaving it set would disable recovery for
    this credential for the rest of the invocation, since the key stays
    readable in the live state object until the turn ends.
    """
    if not self.tool_context or not self.credential_store:
      return
    self.tool_context.state[self._recovery_lock_key()] = False

  def note_successful_call(self) -> None:
    """Refills this credential's recovery budgets.

    Called by the tool after a request that the API did not reject. A working
    call is the only evidence that recovery actually helped, so it is what
    restores the budgets. Without it, exhausted budgets would silence refresh
    and re-authorization for this credential for the rest of the session,
    including for an ordinary token expiry hours later.
    """
    if not self.tool_context or not self.credential_store:
      return
    for budget_key in (self._reauth_budget_key(), self._refresh_budget_key()):
      # Written only when actually set, so an ordinary successful call does
      # not emit a state delta on every request.
      if self.tool_context.state.get(budget_key):
        self.tool_context.state[budget_key] = 0

  async def handle_unauthorized_error(self) -> UnauthorizedRecoveryState:
    """Recovers from an HTTP 401 Unauthorized error returned by an API call.

    Attempts a reactive background token refresh if a refresh token is present.
    If that is not possible or fails, and the scheme is one the user can
    interactively re-authorize (OAuth2 or OpenID Connect), the stale credential
    is evicted from session state and re-authorization is requested. For every
    other scheme a 401 is not recoverable here, so the caller is left to report
    the API error.

    Asking the user to re-authorize is budgeted per credential, because an
    endpoint that keeps returning 401 would otherwise ask on every call for the
    rest of the session. A refresh is never budgeted: it is silent and costs
    the user nothing.

    Returns:
      "refreshed" if a new access token was obtained and stored, in which case
      the caller should retry the API call once; "reauth_requested" if the
      credential was evicted and a re-authorization request was issued to the
      user; "reauth_limit_reached" if re-authorization would have been
      requested but this credential has spent its budget; "failed" if no
      recovery was possible.
    """
    # Budget bookkeeping needs the store to name the credential and the context
    # to read and write the counts. Without both there is nowhere to record an
    # attempt, so the budgets are skipped rather than silently reset to zero on
    # every call.
    can_budget = bool(self.tool_context and self.credential_store)
    refresh_key = self._refresh_budget_key() if can_budget else ""
    refreshes = self.tool_context.state.get(refresh_key, 0) if can_budget else 0

    if self.credential_store:
      existing_credential = self.credential_store.get_credential(
          self.auth_scheme, self.auth_credential
      )
      # Checked before the budget so that a credential left with nothing to
      # refresh from still reaches the re-authorization path below, whatever
      # earlier refreshes cost.
      if (
          existing_credential
          and existing_credential.oauth2
          and existing_credential.oauth2.refresh_token
      ):
        if refreshes >= _MAX_RECOVERY_REFRESHES:
          # This credential was already refreshed once and nothing has worked
          # since, so the token endpoint is issuing tokens the API keeps
          # rejecting. Another refresh would spend a round trip, and under
          # rotation another refresh token, to be told the same thing.
          # Re-authorization is deliberately not offered in its place: what
          # produces a valid token the resource still refuses is usually
          # access withdrawn at the resource rather than at the provider, and
          # a new grant from the provider does not restore that.
          logger.warning(
              "Reactive refresh budget exhausted after a 401; reporting the"
              " API error until a call with this credential succeeds."
          )
          return "failed"
        # The server has rejected a token the credential still believes in, so
        # the recorded expiry cannot be trusted to trigger the refresh. A
        # refresh that fails is reported by the helper rather than raised, and
        # recovery falls through to asking the user.
        outcome = await self._refresh_and_store(
            existing_credential, force_expiry=True
        )
        if outcome.new_access_token or outcome.rotated_refresh_token:
          # Spent by any refresh that contacted the token endpoint and produced
          # a new access token or rotated the refresh token. One that failed
          # without moving either token leaves the budget alone, since asking
          # the user is still untried.
          if can_budget:
            self.tool_context.state[refresh_key] = refreshes + 1
        if outcome.new_access_token:
          return "refreshed"

    # Everything below asks the user to re-authorize, which only means
    # something for the interactive OAuth2 flows. A 401 under an apiKey or a
    # static http scheme is a configuration or server-side problem that no
    # amount of user interaction fixes, and the stored credential is a copy of
    # the tool's own static configuration, so evicting it changes nothing.
    # Reporting the API error is the useful answer for those schemes.
    if not self.auth_scheme or self.auth_scheme.type_ not in (
        AuthSchemeType.openIdConnect,
        AuthSchemeType.oauth2,
    ):
      return "failed"

    budget_key = self._reauth_budget_key() if can_budget else ""
    requests = self.tool_context.state.get(budget_key, 0) if can_budget else 0
    if requests >= _MAX_REAUTH_REQUESTS:
      # The user has already re-authorized this credential and the API still
      # rejected it, so asking again would ask for exactly what was just
      # granted. The stored credential is deliberately left in place: evicting
      # it would send the next call down the no-credential path in
      # prepare_auth_credentials, which asks unconditionally and would reopen
      # the loop this budget exists to close.
      logger.warning(
          "Re-authorization budget exhausted after a 401; not asking again"
          " until a call with this credential succeeds."
      )
      return "reauth_limit_reached"

    try:
      self._request_credential()
    except Exception as e:
      # _request_credential validates the scheme and credential and raises if
      # re-authorization cannot be requested (e.g. a missing client_id).
      logger.warning("Failed to request credential after 401: %s", e)
      # The stored credential is deliberately left in place. Evicting it here
      # would drop it with nothing to replace it, and the next call would take
      # the "no credential" path in prepare_auth_credentials, where the same
      # validation failure raises instead of returning an error the model can
      # read.
      return "failed"

    # Spent only once a request has actually been issued, so a scheme that
    # could not ask the user keeps its budget intact.
    if can_budget:
      self.tool_context.state[budget_key] = requests + 1

    # Only now is the stale credential safe to drop: a re-authorization is
    # pending, so the next call has something to replace it with. It does have
    # to go before that call, or _get_existing_credential would hand back the
    # dead token and the new auth response would never be consulted.
    self._evict_credentials()
    return "reauth_requested"

  def _request_credential(self) -> None:
    """Handles the case where an OpenID Connect or OAuth2 authentication request is needed."""
    if self.auth_scheme.type_ in (
        AuthSchemeType.openIdConnect,
        AuthSchemeType.oauth2,
    ):
      if not self.auth_credential or not self.auth_credential.oauth2:
        raise ValueError(
            f"auth_credential is empty for scheme {self.auth_scheme.type_}."
            "Please create AuthCredential using OAuth2Auth."
        )

      if not self.auth_credential.oauth2.client_id:
        raise AuthCredentialMissingError(
            "OAuth2 credentials client_id is missing."
        )

      if not self.auth_credential.oauth2.client_secret:
        raise AuthCredentialMissingError(
            "OAuth2 credentials client_secret is missing."
        )

    self.tool_context.request_credential(self._build_auth_config())
    return None

  def _get_auth_response(self) -> AuthCredential:
    return self.tool_context.get_auth_response(self._build_auth_config())

  def _external_exchange_required(self, credential) -> bool:
    return (
        credential.auth_type
        in (
            AuthCredentialTypes.OAUTH2,
            AuthCredentialTypes.OPEN_ID_CONNECT,
        )
        and not credential.oauth2.access_token
    )

  async def prepare_auth_credentials(
      self,
  ) -> AuthPreparationResult:
    """Prepares authentication credentials, handling exchange and user interaction."""

    # no auth is needed
    if not self.auth_scheme:
      return AuthPreparationResult(state="done")

    # Check for existing credential.
    existing_credential = await self._get_existing_credential()

    credential = existing_credential or self.auth_credential
    # fetch credential from adk framework
    # Some auth scheme like OAuth2 AuthCode & OpenIDConnect may require
    # multistep exchange:
    # client_id , client_secret -> auth_uri -> auth_code -> access_token
    # adk framework supports exchange access_token already
    # for other credential, adk can also get back the credential directly
    if not credential or self._external_exchange_required(credential):
      credential = self._get_auth_response()
      # store fetched credential
      if credential:
        self._store_credential(credential)
      else:
        self._request_credential()
        return AuthPreparationResult(
            state="pending",
            auth_scheme=self.auth_scheme,
            auth_credential=self.auth_credential,
        )

    # here exchangers are doing two different thing:
    # for service account the exchanger is doing actual token exchange
    # while for oauth2 it's actually doing the credential conversion
    # from OAuth2 credential to HTTP credentials for setting credential in
    # http header
    # TODO cleanup the logic:
    # 1. service account token exchanger should happen before we store them in
    #    the token store
    # 2. blow line should only do credential conversion

    exchanged_credential = self._exchange_credential(credential)
    return AuthPreparationResult(
        state="done",
        auth_scheme=self.auth_scheme,
        auth_credential=exchanged_credential,
    )

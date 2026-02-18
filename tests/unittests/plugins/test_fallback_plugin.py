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

from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.fallback_plugin import FallbackPlugin
from google.adk.sessions.session import Session
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session():
    session = Mock(spec=Session)
    session.id = "test-session-id"
    session.app_name = "test-app"
    session.user_id = "test-user"
    return session


@pytest.fixture
def mock_invocation_context(mock_session):
    ctx = Mock(spec=InvocationContext)
    ctx.invocation_id = "test-invocation-id"
    ctx.session = mock_session
    return ctx


@pytest.fixture
def mock_callback_context(mock_invocation_context):
    ctx = Mock(spec=CallbackContext)
    ctx._invocation_context = mock_invocation_context
    return ctx


@pytest.fixture
def default_plugin():
    return FallbackPlugin(
        root_model="gemini-3-flash-preview",
        fallback_model="gemini-2.5-flash",
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestFallbackPluginInitialization:
    """Tests for FallbackPlugin initialization."""

    def test_default_initialization(self):
        """Test plugin initialization with default parameter values."""
        plugin = FallbackPlugin()

        assert plugin.name == "fallback_plugin"
        assert plugin.root_model is None
        assert plugin.fallback_model is None
        assert plugin.error_status == [429, 504]
        assert plugin._fallback_attempts == {}

    def test_custom_initialization(self):
        """Test plugin initialization with custom parameter values."""
        plugin = FallbackPlugin(
            name="my_fallback",
            root_model="gemini-3-flash-preview",
            fallback_model="gemini-2.5-flash",
            error_status=[429, 503, 504],
        )

        assert plugin.name == "my_fallback"
        assert plugin.root_model == "gemini-3-flash-preview"
        assert plugin.fallback_model == "gemini-2.5-flash"
        assert plugin.error_status == [429, 503, 504]
        assert plugin._fallback_attempts == {}

    def test_custom_error_status(self):
        """Test that a custom error status list is stored correctly."""
        plugin = FallbackPlugin(error_status=[500, 503])
        assert plugin.error_status == [500, 503]

    def test_empty_error_status(self):
        """Test initialization with an empty error status list."""
        plugin = FallbackPlugin(error_status=[])
        assert plugin.error_status == []


# ---------------------------------------------------------------------------
# before_model_callback tests
# ---------------------------------------------------------------------------


class TestBeforeModelCallback:
    """Tests for FallbackPlugin.before_model_callback."""

    @pytest.mark.asyncio
    async def test_resets_model_to_root_when_different(
        self, default_plugin, mock_callback_context
    ):
        """Plugin should reset the model back to root_model when it differs."""
        llm_request = LlmRequest(model="some-other-model")

        result = await default_plugin.before_model_callback(
            callback_context=mock_callback_context, llm_request=llm_request
        )

        assert result is None
        assert llm_request.model == "gemini-3-flash-preview"

    @pytest.mark.asyncio
    async def test_does_not_override_model_when_already_root(
        self, default_plugin, mock_callback_context
    ):
        """Plugin should leave the model unchanged when it is already root_model."""
        llm_request = LlmRequest(model="gemini-3-flash-preview")

        result = await default_plugin.before_model_callback(
            callback_context=mock_callback_context, llm_request=llm_request
        )

        assert result is None
        assert llm_request.model == "gemini-3-flash-preview"

    @pytest.mark.asyncio
    async def test_no_root_model_configured_leaves_model_unchanged(
        self, mock_callback_context
    ):
        """When no root_model is configured the request model must not change."""
        plugin = FallbackPlugin(fallback_model="gemini-2.5-flash")
        llm_request = LlmRequest(model="gemini-3-flash-preview")

        result = await plugin.before_model_callback(
            callback_context=mock_callback_context, llm_request=llm_request
        )

        assert result is None
        assert llm_request.model == "gemini-3-flash-preview"

    @pytest.mark.asyncio
    async def test_initializes_fallback_counter_on_first_call(
        self, default_plugin, mock_callback_context
    ):
        """A new context should receive a zero-initialised fallback counter."""
        llm_request = LlmRequest(model="gemini-3-flash-preview")
        context_id = id(mock_callback_context)

        assert context_id not in default_plugin._fallback_attempts

        await default_plugin.before_model_callback(
            callback_context=mock_callback_context, llm_request=llm_request
        )

        assert default_plugin._fallback_attempts[context_id] == 0

    @pytest.mark.asyncio
    async def test_does_not_reset_model_when_fallback_in_progress(
        self, default_plugin, mock_callback_context
    ):
        """When a fallback attempt is in progress the model should not be reset."""
        context_id = id(mock_callback_context)
        default_plugin._fallback_attempts[context_id] = 1  # Simulate active fallback

        llm_request = LlmRequest(model="gemini-3-flash-preview")  # fallback model is set

        await default_plugin.before_model_callback(
            callback_context=mock_callback_context, llm_request=llm_request
        )

        # Model should remain as the fallback model (not reset to root)
        assert llm_request.model == "gemini-3-flash-preview"

    @pytest.mark.asyncio
    async def test_returns_none_to_allow_normal_processing(
        self, default_plugin, mock_callback_context
    ):
        """before_model_callback must return None to continue the chain."""
        llm_request = LlmRequest(model="gemini-3-flash-preview")

        result = await default_plugin.before_model_callback(
            callback_context=mock_callback_context, llm_request=llm_request
        )

        assert result is None


# ---------------------------------------------------------------------------
# after_model_callback tests
# ---------------------------------------------------------------------------


class TestAfterModelCallback:
    """Tests for FallbackPlugin.after_model_callback."""

    @pytest.mark.asyncio
    async def test_no_error_returns_none(
        self, default_plugin, mock_callback_context
    ):
        """A successful response with no error should pass through unchanged."""
        llm_response = LlmResponse(
            content=None, error_code=None, error_message=None
        )

        result = await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert result is None
        assert llm_response.custom_metadata is None

    @pytest.mark.asyncio
    async def test_error_code_429_triggers_fallback_metadata(
        self, default_plugin, mock_callback_context
    ):
        """A 429 error should record fallback metadata on the response."""
        llm_response = LlmResponse(
            error_code="429", error_message="Rate limit exceeded"
        )

        result = await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert result is None
        assert llm_response.custom_metadata is not None
        assert llm_response.custom_metadata["fallback_triggered"] is True
        assert llm_response.custom_metadata["original_model"] == "gemini-3-flash-preview"
        assert llm_response.custom_metadata["fallback_model"] == "gemini-2.5-flash"
        assert llm_response.custom_metadata["error_code"] == "429"

    @pytest.mark.asyncio
    async def test_error_code_504_triggers_fallback_metadata(
        self, default_plugin, mock_callback_context
    ):
        """A 504 error should also trigger fallback metadata."""
        llm_response = LlmResponse(
            error_code="504", error_message="Gateway timeout"
        )

        await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert llm_response.custom_metadata["fallback_triggered"] is True
        assert llm_response.custom_metadata["error_code"] == "504"

    @pytest.mark.asyncio
    async def test_non_configured_error_code_does_not_trigger_fallback(
        self, default_plugin, mock_callback_context
    ):
        """An error code not in error_status should not add fallback metadata."""
        llm_response = LlmResponse(
            error_code="500", error_message="Internal server error"
        )

        await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert llm_response.custom_metadata is None

    @pytest.mark.asyncio
    async def test_fallback_attempt_counter_increments_on_error(
        self, default_plugin, mock_callback_context
    ):
        """Each error response should increment the fallback counter."""
        context_id = id(mock_callback_context)
        default_plugin._fallback_attempts[context_id] = 0

        llm_response = LlmResponse(error_code="429", error_message="Too many requests")

        await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert default_plugin._fallback_attempts[context_id] == 1

    @pytest.mark.asyncio
    async def test_fallback_attempt_counter_increments_multiple_times(
        self, default_plugin, mock_callback_context
    ):
        """Multiple errors on the same context accumulate in the counter."""
        context_id = id(mock_callback_context)
        default_plugin._fallback_attempts[context_id] = 0

        for expected_count in range(1, 4):
            llm_response = LlmResponse(
                error_code="429", error_message="Too many requests"
            )
            await default_plugin.after_model_callback(
                callback_context=mock_callback_context, llm_response=llm_response
            )
            assert default_plugin._fallback_attempts[context_id] == expected_count

    @pytest.mark.asyncio
    async def test_attempt_count_stored_in_metadata(
        self, default_plugin, mock_callback_context
    ):
        """The fallback attempt number should be persisted in custom_metadata."""
        context_id = id(mock_callback_context)
        default_plugin._fallback_attempts[context_id] = 2  # Already attempted twice

        llm_response = LlmResponse(error_code="429", error_message="Rate limit")

        await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert llm_response.custom_metadata["fallback_attempt"] == 3

    @pytest.mark.asyncio
    async def test_no_fallback_model_does_not_set_metadata(
        self, mock_callback_context
    ):
        """When no fallback_model is configured no metadata should be written."""
        plugin = FallbackPlugin(root_model="gemini-3-flash-preview")  # No fallback_model
        llm_response = LlmResponse(error_code="429", error_message="Rate limit")

        await plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert llm_response.custom_metadata is None

    @pytest.mark.asyncio
    async def test_existing_custom_metadata_is_preserved(
        self, default_plugin, mock_callback_context
    ):
        """Fallback metadata must be merged into any pre-existing custom_metadata."""
        llm_response = LlmResponse(
            error_code="429",
            error_message="Rate limit",
            custom_metadata={"my_key": "my_value"},
        )

        await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        # Original key must survive
        assert llm_response.custom_metadata["my_key"] == "my_value"
        # Fallback keys must be added
        assert llm_response.custom_metadata["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_integer_error_codes_match_string_comparison(
        self, mock_callback_context
    ):
        """error_status integers should match string error_code values."""
        plugin = FallbackPlugin(
            root_model="gemini-3-flash-preview",
            fallback_model="gemini-3-flash-preview",
            error_status=[429],
        )
        llm_response = LlmResponse(error_code="429", error_message="Rate limit")

        await plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert llm_response.custom_metadata["fallback_triggered"] is True

    @pytest.mark.asyncio
    async def test_returns_none_to_allow_normal_processing(
        self, default_plugin, mock_callback_context
    ):
        """after_model_callback must return None to continue the chain."""
        llm_response = LlmResponse(error_code=None, error_message=None)

        result = await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_even_on_error(
        self, default_plugin, mock_callback_context
    ):
        """after_model_callback must still return None when an error is handled."""
        llm_response = LlmResponse(error_code="429", error_message="Rate limit")

        result = await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )

        assert result is None


# ---------------------------------------------------------------------------
# Memory management tests
# ---------------------------------------------------------------------------


class TestMemoryManagement:
    """Tests for _fallback_attempts memory management."""

    @pytest.mark.asyncio
    async def test_old_entries_pruned_when_limit_exceeded(self):
        """When more than 100 contexts are tracked the oldest 50 are removed."""
        plugin = FallbackPlugin(
            root_model="gemini-3-flash-preview",
            fallback_model="gemini-3-flash-preview",
        )

        # Pre-populate with 100 fake context IDs
        for i in range(100):
            plugin._fallback_attempts[i] = 0

        assert len(plugin._fallback_attempts) == 100

        # Trigger cleanup via a new context
        new_ctx = Mock(spec=CallbackContext)
        new_ctx_id = id(new_ctx)
        plugin._fallback_attempts[new_ctx_id] = 0

        llm_response = LlmResponse(error_code="429", error_message="Rate limit")

        await plugin.after_model_callback(
            callback_context=new_ctx, llm_response=llm_response
        )

        # After cleanup the dict should have 51 entries
        # (101 entries → remove first 50 → 51 remain)
        assert len(plugin._fallback_attempts) == 51

    @pytest.mark.asyncio
    async def test_no_pruning_when_below_limit(self, default_plugin):
        """No entries should be removed when the dict is below the 100-entry threshold."""
        for i in range(50):
            default_plugin._fallback_attempts[i] = 0

        new_ctx = Mock(spec=CallbackContext)
        llm_response = LlmResponse(error_code=None)

        await default_plugin.after_model_callback(
            callback_context=new_ctx, llm_response=llm_response
        )

        # All 50 pre-populated entries should still be present
        assert len(default_plugin._fallback_attempts) == 50


# ---------------------------------------------------------------------------
# Non-persistent fallback (round-trip) tests
# ---------------------------------------------------------------------------


class TestNonPersistentFallback:
    """Tests that verify the non-persistent (per-request) fallback behaviour."""

    @pytest.mark.asyncio
    async def test_model_resets_to_root_on_second_request(
        self, default_plugin
    ):
        """After a fallback the next fresh request must start with root_model again."""
        ctx1 = Mock(spec=CallbackContext)

        # First call: simulate fallback in progress
        ctx1_id = id(ctx1)
        default_plugin._fallback_attempts[ctx1_id] = 1
        llm_request = LlmRequest(model="gemini-3-flash-preview")

        await default_plugin.before_model_callback(
            callback_context=ctx1, llm_request=llm_request
        )
        # Model should NOT be reset because a fallback is in progress
        assert llm_request.model == "gemini-3-flash-preview"

        # Second call: brand new context (simulates a new request)
        ctx2 = Mock(spec=CallbackContext)
        llm_request2 = LlmRequest(model="gemini-3-flash-preview")

        await default_plugin.before_model_callback(
            callback_context=ctx2, llm_request=llm_request2
        )
        # Fresh context → counter is 0 → model should be reset to root
        assert llm_request2.model == "gemini-3-flash-preview"

    @pytest.mark.asyncio
    async def test_full_round_trip_error_then_new_request(
        self, default_plugin, mock_callback_context
    ):
        """
        Simulate a full lifecycle:
          1. before_model_callback initialises the context.
          2. after_model_callback handles the 429 error.
          3. A new context's before_model_callback resets to root_model.
        """
        # Step 1: before – initialise the context
        llm_request = LlmRequest(model="gemini-3-flash-preview")
        await default_plugin.before_model_callback(
            callback_context=mock_callback_context, llm_request=llm_request
        )
        assert llm_request.model == "gemini-3-flash-preview"

        # Step 2: after – error triggers fallback tracking
        llm_response = LlmResponse(error_code="429", error_message="Rate limit")
        await default_plugin.after_model_callback(
            callback_context=mock_callback_context, llm_response=llm_response
        )
        assert llm_response.custom_metadata["fallback_triggered"] is True

        # Step 3: new context resets cleanly
        new_ctx = Mock(spec=CallbackContext)
        llm_request_new = LlmRequest(model="some-other-model")
        await default_plugin.before_model_callback(
            callback_context=new_ctx, llm_request=llm_request_new
        )
        assert llm_request_new.model == "gemini-3-flash-preview"

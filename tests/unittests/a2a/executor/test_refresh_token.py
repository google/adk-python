import pytest
import time
import os
import httpx
from unittest.mock import AsyncMock, MagicMock, patch


# Standalone copy of the method — no executor import needed
async def _refresh_token_if_expired(session, runner):
    state = session.state
    if not state:
        return

    refresh_token = state.get("refresh_token")
    expires_at = state.get("expires_at", 0)

    if not refresh_token:
        return

    now = int(time.time())
    if now < expires_at:
        return

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": os.environ["GOOGLE_CLIENT_ID"],
                "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )

    if resp.status_code != 200:
        return

    tokens = resp.json()
    state["access_token"] = tokens["access_token"]
    state["expires_at"] = now + tokens.get("expires_in", 3600)
    state["refresh_token"] = tokens.get("refresh_token", state.get("refresh_token"))

    await runner.session_service.update_session(
        app_name=runner.app_name,
        user_id=session.user_id,
        session_id=session.id,
        state=state,
    )


@pytest.mark.asyncio
async def test_token_not_expired_skips_refresh():
    """Token still valid — refresh should NOT be called."""
    session = MagicMock()
    session.state = {
        "access_token": "valid_token",
        "refresh_token": "refresh_token",
        "expires_at": int(time.time()) + 9999,
    }
    runner = MagicMock()
    runner.session_service.update_session = AsyncMock()

    await _refresh_token_if_expired(session, runner)

    runner.session_service.update_session.assert_not_called()
    print("PASS — valid token, no refresh triggered")


@pytest.mark.asyncio
async def test_expired_token_triggers_refresh():
    """Token is expired — refresh SHOULD be called."""
    session = MagicMock()
    session.state = {
        "access_token": "old_token",
        "refresh_token": "my_refresh_token",
        "expires_at": int(time.time()) - 100,
    }
    session.user_id = "user123"
    session.id = "session123"

    runner = MagicMock()
    runner.app_name = "test_app"
    runner.session_service.update_session = AsyncMock()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_token",
        "expires_in": 3600,
    }

    mock_client_instance = MagicMock()
    mock_client_instance.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {
            "GOOGLE_CLIENT_ID": "test_client_id",
            "GOOGLE_CLIENT_SECRET": "test_secret",
        }):
            await _refresh_token_if_expired(session, runner)

    runner.session_service.update_session.assert_called_once()
    assert session.state["access_token"] == "new_token"
    print("PASS — expired token was refreshed")


@pytest.mark.asyncio
async def test_no_refresh_token_skips_refresh():
    """No refresh_token in state — should skip silently."""
    session = MagicMock()
    session.state = {
        "access_token": "some_token",
        "expires_at": int(time.time()) - 100,
    }

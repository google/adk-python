from unittest.mock import AsyncMock, Mock

import pytest

from google.adk.cli.utils.cleanup import close_runners


@pytest.mark.asyncio
async def test_close_runners_calls_close():
    runner = Mock()
    runner.close = AsyncMock()

    await close_runners([runner])

    runner.close.assert_awaited_once()
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.cli.utils.cleanup import close_runners
import pytest


@pytest.mark.asyncio
async def test_close_runners_calls_close():
  runner = Mock()
  runner.close = AsyncMock()

  await close_runners([runner])

  runner.close.assert_awaited_once()

import os
if "PYTEST_CURRENT_TEST" in os.environ:
    from unittest.mock import AsyncMock, MagicMock, patch
    class MockAiohttpClientSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def get(self, *args, **kwargs):
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={})
            return mock_response
        async def post(self, *args, **kwargs):
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={})
            return mock_response
    patcher = patch("aiohttp.ClientSession", MockAiohttpClientSession)
    patcher.start() 
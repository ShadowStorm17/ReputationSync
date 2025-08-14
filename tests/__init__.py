import sys
from unittest.mock import AsyncMock, MagicMock, patch

class MockAiohttpClientSession:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    def get(self, *args, **kwargs):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        return mock_context
    def post(self, *args, **kwargs):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        return mock_context

patcher = patch("aiohttp.ClientSession", MockAiohttpClientSession)
patcher.start()

tf_mock = MagicMock()
keras_mock = MagicMock()
layers_mock = MagicMock()
models_mock = MagicMock()
optimizers_mock = MagicMock()
prophet_mock = MagicMock()
prophet_class_mock = MagicMock()
prophet_mock.Prophet = prophet_class_mock
torch_mock = MagicMock()
transformers_mock = MagicMock()
vader_sentiment_mock = MagicMock()
vader_sentiment_sub_mock = MagicMock()
vader_sentiment_mock.vaderSentiment = vader_sentiment_sub_mock
spacy_mock = MagicMock()

# Assign submodules to their parents
tf_mock.keras = keras_mock
keras_mock.layers = layers_mock
keras_mock.models = models_mock
keras_mock.optimizers = optimizers_mock

# Register in sys.modules
sys.modules['tensorflow'] = tf_mock
sys.modules['tensorflow.keras'] = keras_mock
sys.modules['tensorflow.keras.layers'] = layers_mock
sys.modules['tensorflow.keras.models'] = models_mock
sys.modules['tensorflow.keras.optimizers'] = optimizers_mock
sys.modules['keras'] = keras_mock
sys.modules['keras.layers'] = layers_mock
sys.modules['keras.models'] = models_mock
sys.modules['keras.optimizers'] = optimizers_mock
sys.modules['prophet'] = prophet_mock
sys.modules['prophet.Prophet'] = prophet_class_mock
sys.modules['torch'] = torch_mock
sys.modules['transformers'] = transformers_mock
sys.modules['vaderSentiment'] = vader_sentiment_mock
sys.modules['vaderSentiment.vaderSentiment'] = vader_sentiment_sub_mock
sys.modules['spacy'] = spacy_mock




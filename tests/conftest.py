import sys
from unittest.mock import MagicMock, AsyncMock

import pytest
from typing import AsyncGenerator, Generator

import aiosqlite
import os
import asyncio
import redis.asyncio as redis

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as TestingSessionLocal

from app.core.config import get_settings, settings
from app.core.database import Base
from app.core.security import create_access_token
from app.main import app
from app.services.database import DatabaseService
from app.services.instagram_service import InstagramAPI
from app.services.sentiment_service import SentimentService
from app.core.security import SecurityService
from app.db.session import get_db
from app.models.user import User
from app.db.base_class import Base

from dotenv import load_dotenv
load_dotenv()

# Set testing flag for all tests
settings.TESTING = True

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

# Patch HuggingFace transformers
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.pipeline"] = MagicMock()
sys.modules["transformers.AutoModelForTokenClassification"] = MagicMock()
sys.modules["transformers.AutoTokenizer"] = MagicMock()
sys.modules["transformers.AutoModelForSequenceClassification"] = MagicMock()

# Patch torch
sys.modules["torch"] = MagicMock()

# Patch sklearn
sys.modules["sklearn"] = MagicMock()

# Patch tensorflow
sys.modules["tensorflow"] = MagicMock()

# Patch any other heavy ML modules as needed

# Patch aiohttp.ClientSession to support async context manager in tests


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_db() -> AsyncGenerator:
    """Create a test database."""
    test_db_path = "test.db"
    db = await aiosqlite.connect(test_db_path)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            name TEXT,
            key TEXT,
            user_id INTEGER,
            created_at TEXT,
            revoked INTEGER DEFAULT 0
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            endpoint TEXT,
            response_time INTEGER,
            success INTEGER,
            api_key TEXT,
            FOREIGN KEY(api_key) REFERENCES api_keys(id)
        )
    """)
    await db.commit()
    await db.close()
    yield test_db_path
    # Ensure all connections are closed before deleting
    import time
    for _ in range(10):
        try:
            os.unlink(test_db_path)
            break
        except PermissionError:
            time.sleep(0.1)

@pytest.fixture(scope="session")
async def test_redis() -> AsyncGenerator:
    """Create a test Redis connection."""
    redis_client = redis.Redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        db=1  # Use different DB for testing
    )
    yield redis_client
    await redis_client.flushdb()  # Clean up after tests
    await redis_client.close()

@pytest.fixture
def test_client() -> Generator:
    """Create a test client."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
async def test_api_key(test_db) -> AsyncGenerator:
    """Create a test API key."""
    db_service = DatabaseService()
    db_service.db_path = test_db
    
    key_data = await db_service.create_api_key(
        name="test_key",
        key="test_key_123",
        user_id=1
    )
    yield key_data
    await db_service.revoke_api_key(key_data["id"])

@pytest.fixture
def security_service() -> SecurityService:
    """Create a security service instance."""
    return SecurityService()

@pytest.fixture
def instagram_service(test_redis) -> InstagramAPI:
    """Create an Instagram API instance."""
    service = InstagramAPI()
    service.redis = test_redis
    return service

@pytest.fixture(scope="session")
def db_engine():
    """Create a test database engine."""
    from app.models.user import User  # Ensure User is imported for table creation
    engine = create_engine(settings.TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a test database session."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with a test database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def test_user(db_session):
    """Create a test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password",
        is_active=True,
        is_superuser=False
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture(scope="function")
def test_superuser(db_session):
    """Create a test superuser."""
    user = User(
        username="adminuser",
        email="admin@example.com",
        hashed_password="hashed_password",
        is_active=True,
        is_superuser=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture(scope="function")
def test_user_token(test_user):
    """Create a test user access token."""
    return create_access_token({"sub": test_user.username})

@pytest.fixture(scope="function")
def test_superuser_token(test_superuser):
    """Create a test superuser access token."""
    return create_access_token({"sub": test_superuser.username})

@pytest.fixture(scope="function")
def authorized_client(client, test_user_token):
    """Create an authorized test client."""
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_user_token}"
    }
    return client

@pytest.fixture(scope="function")
def authorized_superuser_client(client, test_superuser_token):
    """Create an authorized test client with superuser privileges."""
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_superuser_token}"
    }
    return client

# --- EARLY GLOBAL PATCHING OF HEAVY ML MODELS FOR TESTS ---
import sys
from unittest.mock import patch, MagicMock

patches = [
    patch("sklearn.ensemble.IsolationForest", autospec=True),
    patch("sklearn.ensemble.RandomForestRegressor", autospec=True),
    patch("sklearn.preprocessing.StandardScaler", autospec=True),
    patch("sklearn.cluster.KMeans", autospec=True),
    patch("sklearn.decomposition.PCA", autospec=True),
    patch("torch.no_grad", MagicMock()),
    patch("torch.softmax", MagicMock(return_value=MagicMock())),
    patch("torch.Tensor", MagicMock()),
    patch("torch.from_numpy", MagicMock()),
]

dummy = MagicMock()
dummy.fit.return_value = dummy
dummy.predict.return_value = [0] * 10
dummy.score_samples.return_value = [0.0] * 10
dummy.transform.return_value = [[0.0] * 3] * 10
dummy.fit_predict.return_value = [1] * 10
dummy.make_future_dataframe.return_value = MagicMock()
dummy.predict_proba.return_value = [[0.5, 0.5]] * 10
dummy.generate.return_value = [[0]]
dummy.decode.return_value = "dummy response"
dummy.return_value = dummy
dummy.get.return_value = dummy
dummy.set.return_value = None
dummy.flushdb.return_value = None
dummy.close.return_value = None
dummy.inc.return_value = None
dummy.observe.return_value = None

patches.append(patch("prophet.Prophet.fit", MagicMock(return_value=None)))
patches.append(patch("prophet.Prophet.predict", MagicMock(return_value=MagicMock(yhat=[0.0]*30))))
patches.append(patch("tensorflow.keras.models.Sequential.fit", MagicMock(return_value=None)))
patches.append(patch("tensorflow.keras.models.Sequential.predict", MagicMock(return_value=[[0.0]]*10)))
patches.append(patch("transformers.pipeline", MagicMock(return_value=dummy)))

if "torch" in sys.modules:
    sys.modules["torch"].no_grad = MagicMock()
    sys.modules["torch"].softmax = MagicMock(return_value=MagicMock())
    sys.modules["torch"].Tensor = MagicMock()
    sys.modules["torch"].from_numpy = MagicMock()

_started_patches = [p.start() for p in patches]

import pytest

def pytest_sessionfinish(session, exitstatus):
    for p in patches:
        p.stop() 

@pytest.fixture(autouse=True)
def reset_metrics_and_monitoring():
    """Reset metrics_manager and monitoring_manager before each test."""
    from prometheus_client import REGISTRY
    collectors = list(REGISTRY._names_to_collectors.values())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    from app.core.metrics import metrics_manager
    from app.core.monitoring import monitoring_manager
    metrics_manager.__init__()
    monitoring_manager.__init__() 
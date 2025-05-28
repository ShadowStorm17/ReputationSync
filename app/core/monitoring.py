from prometheus_client import Counter, Histogram
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from app.core.config import get_settings
import time
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUESTS = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUESTS_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

INSTAGRAM_REQUESTS = Counter(
    'instagram_api_requests_total',
    'Total Instagram API requests',
    ['status']
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'status']
)

def setup_monitoring(app):
    """Configure monitoring for the application."""
    if settings.SENTRY_DSN:
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.ENVIRONMENT,
            traces_sample_rate=1.0
        )
        app.add_middleware(SentryAsgiMiddleware)
        logger.info("Sentry monitoring configured")

    if settings.TRACING_ENABLED:
        # Configure OpenTelemetry
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Configure exporter
        otlp_exporter = OTLPSpanExporter()
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry tracing configured")

class MetricsMiddleware:
    """Middleware to collect request metrics."""
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record request duration
        duration = time.time() - start_time
        
        # Update Prometheus metrics
        REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUESTS_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response

def record_instagram_request(status: str):
    """Record Instagram API request metrics."""
    INSTAGRAM_REQUESTS.labels(status=status).inc()

def record_cache_operation(operation: str, status: str):
    """Record cache operation metrics."""
    CACHE_OPERATIONS.labels(operation=operation, status=status).inc() 
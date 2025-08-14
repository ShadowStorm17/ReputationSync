"""
Monitoring and observability module.
Provides metrics collection, tracing, and health monitoring.
"""

import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
    OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import get_settings
from app.core.metrics import REQUEST_RATE as REQUEST_COUNT, REQUEST_DURATION as REQUEST_LATENCY

settings = get_settings()

class MonitoringManager:
    """Monitoring manager for metrics and tracing."""

    def __init__(self):
        """Initialize monitoring manager."""
        self._setup_tracing()

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        # Create tracer provider
        tracer_provider = TracerProvider()

        # Create OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=f"{settings.OTEL_EXPORTER_OTLP_ENDPOINT}:4317",
            insecure=True
        )

        # Add span processor
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Set tracer provider
        trace.set_tracer_provider(tracer_provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

    async def track_request(
        self, method: str, endpoint: str, status: int, duration: float
    ):
        """Track HTTP request metrics."""
        pass  # Metrics are now initialized in app.core.metrics

    async def track_error(self, error_type: str, endpoint: str):
        """Track error metrics."""
        pass  # Metrics are now initialized in app.core.metrics

    async def track_cache_operation(self, cache_name: str, hit: bool):
        """Track cache operation metrics."""
        pass  # Metrics are now initialized in app.core.metrics

    async def track_db_query(self, query_type: str, duration: float):
        """Track database query metrics."""
        pass  # Metrics are now initialized in app.core.metrics

    async def track_api_response(self, endpoint: str, duration: float):
        """Track API response time metrics."""
        pass  # Metrics are now initialized in app.core.metrics

    async def update_active_users(self, count: int):
        """Update active users metric."""
        pass  # Metrics are now initialized in app.core.metrics

    def get_tracer(self):
        """Get OpenTelemetry tracer."""
        return self.tracer


# Create global monitoring manager instance
monitoring_manager = MonitoringManager()


class MonitoringMiddleware:
    """Middleware for request monitoring."""

    async def __call__(self, request, call_next):
        """Process request through monitoring middleware."""
        start_time = time.time()

        # Get tracer
        tracer = monitoring_manager.get_tracer()

        # Create span
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}"
        ) as span:
            # Add request attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.host", request.headers.get("host", ""))

            try:
                # Process request
                response = await call_next(request)

                # Track metrics
                duration = time.time() - start_time
                await monitoring_manager.track_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code,
                    duration=duration,
                )

                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)

                return response

            except Exception as e:
                # Track error
                await monitoring_manager.track_error(
                    error_type=type(e).__name__, endpoint=request.url.path
                )

                # Add error attributes
                span.set_attribute("error", str(e))
                span.set_status(trace.Status(trace.StatusCode.ERROR))

                raise


# Create global monitoring middleware instance
monitoring_middleware = MonitoringMiddleware()

def record_instagram_request(*args, **kwargs):
    """Stub: Placeholder for Instagram request recording (for tests/imports)."""
    pass

def record_platform_request(*args, **kwargs):
    """Stub: Placeholder for platform request recording (for tests/imports)."""
    pass

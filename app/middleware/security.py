import time

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import logger
from app.core.monitoring import monitoring_manager
from app.core.security_manager import security_manager


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Start timing the request
        start_time = time.time()

        try:
            # Perform security checks
            await security_manager.check_request_security(request)

            # Check rate limits
            endpoint = request.url.path.split("/")[1] if len(request.url.path.split("/")) > 1 else ""
            client_ip = request.client.host if request.client else "unknown"
            client_id = f"{client_ip}:{endpoint}" if endpoint else client_ip
            await security_manager.check_rate_limit(client_id)

            # Process the request
            response = await call_next(request)

            # Add security headers
            response = await security_manager.add_security_headers(response)

            # Track request metrics
            processing_time = time.time() - start_time
            await monitoring_manager.track_request(
                request.url.path,
                request.method,
                response.status_code,
                processing_time,
            )

            # Add timing header
            response.headers["X-Process-Time"] = str(processing_time)

            return response

        except HTTPException:
            # Track failed request
            process_time = time.time() - start_time
            await monitoring_manager.track_error(
                "Security check failed",
                request.url.path,
                request.method,
                process_time,
            )
            raise

        except Exception as e:
            # Track unexpected error
            process_time = time.time() - start_time
            logger.error("Security middleware error: %s", str(e))
            await monitoring_manager.track_error(
                str(e), request.url.path, request.method, process_time
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            )

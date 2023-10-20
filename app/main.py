"""Main FastAPI Application"""
# pylint disable=unused-variable
# pylint disable=unused-argument
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from app.api.routers import model_route
from app.logger.logger import custom_logger

class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging All API request"""

    async def set_body(self, request: Request):
        """Set body."""
        receive_ = await request._receive()

        async def receive():
            """Receive body."""
            return receive_

        request._receive = receive

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Dispatch Logging Middleware"""
        # Log the request
        custom_logger.info("Received request: %s %s", request.method, request.url)

        # Call the next middleware or route handler
        response = await call_next(request)

        # Log the response
        custom_logger.info("Response status code: %s", response.status_code)

        return response


app = FastAPI(
    title="Deepfake Detection"
)

# @app.on_event('startup')
# async def load_ai_model():
#     """Load model at startup"""
#     app.state.model = load_model('app/resources/model_base.h5')

@app.get("/", tags=['Welcome'])
async def hello():
    """Hello"""
    return {"Message": "<3 WELCOME TO OUR PBL6: DEEPFAKE DETECTION <3"}

app.include_router(model_route.router, prefix='/aimodel')
app.add_middleware(LoggingMiddleware)

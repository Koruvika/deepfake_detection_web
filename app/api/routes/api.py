"""API Route"""

from fastapi import APIRouter
from app.api.routes import model_route, upload_route

app = APIRouter()

# Deepfake detection route
app.include_router(
    model_route.router,
    tags=['Deepfake Detection'],
    prefix="/deepfake_detection"
)

# Upload image route
app.include_router(
    upload_route.router,
    tags=['Upload Image'],
)

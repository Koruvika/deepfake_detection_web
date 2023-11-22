"""API Route"""

from fastapi import APIRouter
from app.api.routes import model_route

app = APIRouter()

# Deepfake detection route
app.include_router(
    model_route.router,
    tags=['Deepfake Detection'],
    prefix="/deepfake_detection"
)


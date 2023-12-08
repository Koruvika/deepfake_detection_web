"""Upload image route"""

import os
from httpx import AsyncClient
from io import BytesIO

from fastapi import (
    FastAPI, HTTPException, UploadFile,
    APIRouter
)

from pydantic import BaseModel

router = APIRouter()

def save_image_to_folder(image_bytes, folder_path, file_name):
    """Save image to folder"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "wb") as file:
        file.write(image_bytes)

    return file_path

class ImageURL(BaseModel):
    """ImageURL class"""
    image_url: str

@router.post("/upload_image")
async def upload_image_from_url(image_url: ImageURL):
    """Upload image from URL"""
    try:
        async with AsyncClient() as client:
            # Tải ảnh từ URL
            response = await client.get(image_url.image_url)
            response.raise_for_status()
            image_bytes = response.content

            folder_path = r"app\uploads"
            file_name = "uploaded_image.jpg"
            file_path = save_image_to_folder(image_bytes, folder_path, file_name)

            if os.path.exists(file_path):
                return {"message": "Image uploaded successfully", "file_path": file_path}
            else:
                raise HTTPException(status_code=500, detail="Failed to save the image file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


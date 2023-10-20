"""Router for AI model"""
# pylint: disable=unused-argument

from __future__ import annotations

from io import BytesIO
import cv2
import numpy as np
from PIL import Image

from fastapi import (
    APIRouter, UploadFile, File,
    status, Request, HTTPException, Response
)

from app.ml.base_model import SBI_ONNX, FaceBoxes_ONNX
from app.logger.logger import custom_logger

router = APIRouter(
    tags=['Deepfake Detection Model']
)

face_detector = FaceBoxes_ONNX()
deepfake_detector = SBI_ONNX()

@router.post('/deepfake_detection')
async def deepfake_detection(file: UploadFile = File(...), request: Request = None):
    """Deepfake detection"""
    try:
        # ai_model = request.app.state.model
        image = Image.open(file.file)

        # execute model
        det = face_detector(image)
        det = face_detector.upsize(det, image)
        fixed = face_detector.viz_bbox(image, det)
        fixed_image = Image.fromarray(fixed)
        result = deepfake_detector(fixed)

        # draw test
        result_image = cv2.putText(np.array(fixed_image), f"{result}", (100, 100),
                                   fontScale=1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                   color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        result_image = Image.fromarray(result_image)

        # Convert image into byte PNG
        img_byte_array = BytesIO()
        result_image.save(img_byte_array, format="PNG")
        img_data = img_byte_array.getvalue()

        # Return image with Content-Type là 'image/png'
        return Response(content=img_data, media_type="image/png")
    except Exception as exception:
        custom_logger.error("Detection error: %s", exception)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exception)) from exception

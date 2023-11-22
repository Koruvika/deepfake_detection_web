"""Router for AI model"""
# pylint: disable=unused-argument

from __future__ import annotations

from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import base64
import zlib

from fastapi import (
    APIRouter, UploadFile, File,
    status, Request, HTTPException, Response
)

from app.ml.base_model import SBI_ONNX, FaceBoxes_ONNX
from app.api.services.model_service import ModelService
from app.logger.logger import custom_logger

router = APIRouter()

face_detector = FaceBoxes_ONNX()
deepfake_detector = SBI_ONNX()
model_service = ModelService(face_detector, deepfake_detector)

@router.post('/image')
async def image_deepfake_detection(input_image: UploadFile = File(...), request: Request = None):
    """Deepfake detection for image"""
    try:
        # ai_model = request.app.state.model
        image = Image.open(input_image.file)

        # Call AI Model Service
        img_data = model_service.image_deepfake_detection(image)

        # Return image with Content-Type là 'image/png'
        return Response(content=img_data, media_type="image/png")
    except Exception as exception:
        custom_logger.error("Image detection API error: %s", exception)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exception)) from exception

@router.post('/video')
async def video_deepfake_detection(video: UploadFile = File(...), request: Request = None):
    """Deepfake detection for video"""
    try:
        # if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        #     return {"error": "Invalid video file format"}

        # Save the video file to a temporary location
        video_path = f"app/resources/uploads/{video.filename}"
        with open(video_path, "wb") as f:
            f.write(video.file.read())

        result = process_video(video_path)

        # Call AI Model Service
        # output = model_service.video_deepfake_detection(image)

        # Return list of deepfake percentage of each frames
        return result
    except Exception as exception:
        custom_logger.error("Video detection API error: %s", exception)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=str(exception)) from exception
def process_video(video_path: str):
    """Process video"""
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Could not open video"}

    frames = []
    frame_count = 0
    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_counts, 10).astype(np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame here (e.g., apply filters, analyze, etc.)
        if frame_count in indices:
            frames.append(frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    # Deepfake detection
    try:
        percentages = []
        img_datas = []
        max_frame = 0
        for idx, frame in enumerate(frames):
            pil_image = Image.fromarray(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
            percentage = model_service.image_deepfake_detection(pil_image, get_percentage=True)
            percentages.append(percentage)
            # img_datas.append(img_data)
            # if percentage >= percentages[-1]:
            #     max_frame = idx
        # print(type(img_datas[max_frame]))
        # compressed_data = zlib.compress(img_datas[max_frame])
        # return {"max percentage": f"{max(percentages):.5f}%"}
        return {"percentages": percentages}

    except Exception as exception:
        custom_logger.error("Video detection API error: %s", exception)
        return {"error": "Video detection API error"}


    # frame_data = [cv2.imencode(".jpg", frame)[1].tobytes() for frame in frames]
    # pil_image = Image.fromarray(cv2.cvtColor(frames[20], cv2.COLOR_BGR2RGB))
    #
    # # Convert the Pillow Image to a byte array in PNG format
    # img_byte_array = BytesIO()
    # pil_image.save(img_byte_array, format="PNG")
    # img_data = img_byte_array.getvalue()
    #
    # return Response(content=img_data, media_type="image/png")
    # return {
    #     "message": "Video processing complete",
    #     "frames_count": len(frames)
    # }


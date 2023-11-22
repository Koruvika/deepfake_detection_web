"""AI Model service"""
# pylint: disable=unused-import

from io import BytesIO
import cv2
import numpy as np
from PIL import Image

from app.logger.logger import custom_logger

class ModelService:
    """AI Model Service"""
    def __init__(self, face_detector, deepfake_detector):
        self.face_detector = face_detector
        self.deepfake_detector = deepfake_detector

    def image_deepfake_detection(self, image, get_percentage=False):
        """Image Deepfake and Face Detection"""
        # execute model
        try:
            det = self.face_detector(image)
            det = self.face_detector.upsize(det, image)
            fixed = self.face_detector.viz_bbox(image, det)
            fixed_image = Image.fromarray(fixed)
            result, percentage = self.deepfake_detector(fixed)

            # draw test
            result_image = cv2.putText(np.array(fixed_image), f"{result}", (100, 100),
                                       fontScale=1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                       color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            result_image = Image.fromarray(result_image)

            # Convert image into byte PNG
            img_byte_array = BytesIO()
            result_image.save(img_byte_array, format="PNG")
            img_data = img_byte_array.getvalue()

            if get_percentage:
                return percentage*100
            return img_data
        except Exception as exception:
            custom_logger.error("Detection error: %s", exception)
            raise exception

    def video_deepfake_detection(self, frames):
        """Video Deepfake and Face Detection"""
        # self.image_deepfake_detection()
        pass

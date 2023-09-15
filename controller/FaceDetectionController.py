from model import FaceBoxes_ONNX


class FaceDetectionController:
    def __init__(self, main_model: FaceBoxes_ONNX):
        super().__init__()
        self.main_model = main_model


import cv2
import numpy as np
import onnxruntime

from model.Settings import Settings
import gc

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SBI_ONNX:
    def __init__(self):
        self.settings = Settings("./assets/configs/deepfake_detection_sbi.yaml")
        self.session = onnxruntime.InferenceSession(self.settings.configs["onnx_path"],
                                                    None,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def release(self):
        del self.session

    def __call__(self, img_):
        img = np.array(img_)

        if len(img.shape) == 2:
            img = np.concatenate((img, img, img), axis=2)
        im_height, im_width, im_channel = img.shape
        if im_channel == 1:
            img = np.concatenate((img, img, img), axis=2)
        elif im_channel == 4:
            img = img[..., :3]

        # processing data
        img = cv2.resize(img, dsize=(self.settings.configs["WIDTH"], self.settings.configs["HEIGHT"]))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.
        img = img[np.newaxis, ...]
        ort_inputs = {self.session.get_inputs()[0].name: img}
        # forward
        out = self.session.run(None, ort_inputs)[0]
        out = softmax(out)

        res = np.argmax(out)
        if res == 0:
            return "Pristine"
        else:
            return "Deepfake"

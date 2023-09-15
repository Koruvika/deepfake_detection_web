import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from model import FaceBoxes_ONNX, SBI_ONNX

class StreamlitApp:
    def __init__(self):
        self.face_detector = FaceBoxes_ONNX()
        self.deepfake_detector = SBI_ONNX()
        self.MAX_FILE_SIZE = 5 * 1024 * 1024
        self.init_frontend()

    def init_frontend(self):
        st.set_page_config(layout="wide", page_title="Deepfake Detector")

        st.write("## Detect if your media has been modified")
        st.write(
            ":dog: Try uploading an image. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/Koruvika/deepfake_detection) on GitHub :grin:"
        )
        st.sidebar.write("## Upload and download :gear:")

        self.col1, self.col2 = st.columns(2)
        self.my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    def run(self):
        if self.my_upload is not None:
            if self.my_upload.size > self.MAX_FILE_SIZE:
                st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
            else:
                self.fix_image(upload=self.my_upload)
        else:
            self.fix_image("./assets/images/face_image.jpg")

    def convert_image(self, img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        return byte_im


    def fix_image(self, upload):
        image = Image.open(upload)
        self.col1.write("Original Image :camera:")
        self.col1.image(image)

        det = self.face_detector(image)
        det = self.face_detector.upsize(det, image)
        fixed = self.face_detector.viz_bbox(image, det)
        fixed = Image.fromarray(fixed)
        result = self.deepfake_detector(fixed)

        # draw test
        result = cv2.putText(np.array(fixed), f"{result}", (100, 100), fontScale=1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                             color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        result = Image.fromarray(result)

        self.col2.write("Fixed Image :wrench:")
        self.col2.image(result)
        st.sidebar.markdown("\n")
        st.sidebar.download_button("Download fixed image", self.convert_image(result), "fixed.png", "image/png")


if __name__ == '__main__':
    app = StreamlitApp()
    app.run()
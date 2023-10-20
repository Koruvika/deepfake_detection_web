"""Run Streamlit Application"""
# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=unused-import

import streamlit as st
from PIL import Image
from io import BytesIO
import requests

class StreamlitApp:
    """Streamlit Application"""
    def __init__(self):
        self.column_1, self.column_2 = None, None
        self.my_upload = None
        self.result_image = None
        self.MAX_FILE_SIZE = 5 * 1024 * 1024
        self.init_frontend()

    def init_frontend(self):
        """Initialize Front End"""
        st.set_page_config(layout="wide", page_title="Deepfake Detector")

        st.write("## Detect if your media has been modified")
        st.write(
            ":dog: Try uploading an image. Full quality images can be downloaded from the sidebar. "
            "This code is open source and available [here](https://github.com/Koruvika/deepfake_detection) "
            "on GitHub :grin:"
        )
        st.sidebar.write("## Upload and download :gear:")

        self.column_1, self.column_2 = st.columns(2)
        self.my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    def run(self):
        """Run Streamlit application"""
        if self.my_upload is not None:
            if self.my_upload.size > self.MAX_FILE_SIZE:
                st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
            else:
                self.result_image = self.call_fastapi_api(self.my_upload.getvalue())
                self.fix_image(upload_image=self.my_upload,
                               result_image=self.result_image)
        else:
            init_path = "app/assets/images/pristine/face_image.jpg"
            with open(init_path, "rb") as image_file:
                image_data = image_file.read()
            self.result_image = self.call_fastapi_api(BytesIO(image_data))
            self.fix_image(upload_image=init_path,
                           result_image=self.result_image)

    def convert_image(self, img):
        """Convert image"""
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_img = buf.getvalue()
        return byte_img

    def call_fastapi_api(self, image_upload):
        """Call API to process image"""

        url = "http://127.0.0.1:8000/aimodel/deepfake_detection"
        files = {'file': image_upload}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            result_image_bytes = response.content
            result_image = Image.open(BytesIO(result_image_bytes))
            return result_image
        else:
            st.error("Error calling Deepfake Detection API!")
            return None

    def fix_image(self, upload_image, result_image):
        """Process image"""
        # Input image
        origin_image = Image.open(upload_image)
        self.column_1.write("Original Image :camera:")
        self.column_1.image(origin_image)

        # Result image
        self.column_2.write("Fixed Image :wrench:")
        self.column_2.image(result_image)
        st.sidebar.markdown("\n")
        st.sidebar.download_button("Download fixed image", self.convert_image(result_image), "fixed.png", "image/png")


if __name__ == '__main__':
    app = StreamlitApp()
    app.run()

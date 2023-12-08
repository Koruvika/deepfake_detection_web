"""Run Streamlit Application"""
# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=unused-import

import streamlit as st
from PIL import Image
from io import BytesIO
import os
import requests
import pandas as pd
import plotly.express as px
import json

class StreamlitApp:
    """Streamlit Application"""
    def __init__(self):
        self.column_1, self.column_2 = None, None
        self.video_path = None
        self.my_upload = None
        self.result_image = None
        self.result_video = None
        self.MAX_FILE_SIZE = 5 * 1024 * 1024
        self.import_input = None
        self.import_button = None
        self.result_upload_image = None
        self.init_frontend()

    def init_frontend(self):
        """Initialize Front End"""
        im = Image.open("app/resources/icon/logo.png")
        st.set_page_config(
            layout="wide",
            page_title="Deepfake Detector",
            page_icon=im,
        )

        st.write("## Detect if your media has been modified")
        st.write(
            ":dog: Try uploading an image. Full quality images can be downloaded from the sidebar. "
            "This code is open source and available [here](https://github.com/Koruvika/deepfake_detection) "
            "on GitHub :grin:"
        )
        st.sidebar.write("## Upload and download :gear:")

        self.column_1, self.column_2 = st.columns(2)
        self.my_upload = st.sidebar.file_uploader("Upload an image or video", type=["png", "jpg", "jpeg", "mp4"])
        self.import_input = st.sidebar.text_input("Import Image URL", "")
        self.import_button = st.sidebar.button("Upload")

    def run_with_import(self):
        """Import button"""
        if self.import_button:
            if self.import_input:
                with st.spinner("Processing..."):
                    if any(ext in self.import_input.lower() for ext in [".png", ".jpg", ".jpeg"]):
                        self.result_upload_image = self.image_upload_api_call(self.import_input)
                        image_path = self.result_upload_image["file_path"]
                        format_image_path = os.path.normpath(image_path)

                        with open(format_image_path, "rb") as image_file:
                            image_data = image_file.read()
                        self.result_image = self.image_api_call(BytesIO(image_data))
                        self.fix_image(upload_image=format_image_path,
                                       result_image=self.result_image)
                    else:
                        st.error("Unsupported file format. Please provide a valid image\
                         (.png, .jpg, .jpeg) or video (.mp4) URL.")

    def run(self):
        """Run Streamlit application"""
        if self.my_upload is not None:
            with st.spinner("Processing..."):
                if self.my_upload.type.startswith("image"):
                    if self.my_upload.size > self.MAX_FILE_SIZE:
                        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
                    else:
                        self.result_image = self.image_api_call(self.my_upload.getvalue())
                        self.fix_image(upload_image=self.my_upload,
                                       result_image=self.result_image)
                elif self.my_upload.type.startswith("video"):
                    self.result_video = self.video_api_call(self.my_upload.getvalue())
                    self.video_path = self.save_uploaded_video(self.my_upload)
                    json_result = self.result_video.decode('utf-8', errors='ignore')
                    result_dict = json.loads(json_result)
                    result_percentages = result_dict['percentages']
                    self.draw_bar_chart(result_percentages)
        elif self.import_button:  # Check import button
            self.run_with_import()
        else:
            init_path = "app/ml/assets/images/pristine/face_image.jpg"
            with open(init_path, "rb") as image_file:
                image_data = image_file.read()
            self.result_image = self.image_api_call(BytesIO(image_data))
            self.fix_image(upload_image=init_path,
                           result_image=self.result_image)

    def convert_image(self, img):
        """Convert image"""
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_img = buf.getvalue()
        return byte_img

    def image_upload_api_call(self, url):
        """Upload image API call"""
        api_url = f"http://127.0.0.1:8000/pbl6/upload_image"

        upload_image = {'image_url': url}
        response = requests.post(api_url, json=upload_image)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error("Error calling upload image API!")
            return None

    def image_api_call(self, image_upload):
        """Call API to process image"""

        # url = "https://publicly-rapid-robin.ngrok-free.app/pbl6/deepfake_detection/image"
        url = "http://127.0.0.1:8000/pbl6/deepfake_detection/image"
        files = {'input_image': image_upload}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            result_image_bytes = response.content
            result_image = Image.open(BytesIO(result_image_bytes))
            return result_image
        else:
            st.error("Error calling Deepfake Detection API!")
            return None

    def video_api_call(self, video_upload):
        """Call API to process video"""

        url = "http://127.0.0.1:8000/pbl6/deepfake_detection/video"
        files = {'video': video_upload}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.content
            return result
        else:
            st.error("Error calling Deepfake Detection API!")
            return None

    def save_uploaded_video(self, uploaded_file):
        """Save uploaded video"""
        with open(f'app/resources/uploads/{uploaded_file.name}', "wb") as f:
            f.write(uploaded_file.getvalue())
        return f'app/resources/uploads/{uploaded_file.name}'

    def display_video(self, video_path):
        """Display video"""
        video_bytes = open(video_path, "rb").read()
        self.column_1.video(video_bytes, format="video/mp4")

    def draw_bar_chart(self, result_video):
        """Draw bar chart"""
        list_frames = []
        for i in range(len(result_video)):
            list_frames.append(f"Frame {i+1}")

        chart = pd.DataFrame({'Frames': list_frames,
                              'Deepfake Percentage': result_video})

        # self.column_2.bar_chart(chart.set_index('Frames')['Deepfake Percentage'])
        chart = chart.set_index('Frames').loc[list_frames].reset_index()

        # Make bar chart
        fig = px.bar(chart, x='Frames', y='Deepfake Percentage')
        fig.update_traces(marker_color=['red' if val > 50 else 'blue' for val in chart['Deepfake Percentage']])

        self.display_video(self.video_path)

        # Display bar chart in column_2
        self.column_2.write("Bar Chart :bar_chart:")
        self.column_2.plotly_chart(fig, use_container_width=True)

    def fix_image(self, upload_image, result_image):
        """Process image"""
        # Input image
        origin_image = Image.open(upload_image)
        self.column_1.write("Original Image :camera:")
        resized_origin_image = origin_image.resize((440, 580))
        self.column_1.image(resized_origin_image)

        # Result image
        self.column_2.write("Detected Image :wrench:")
        resized_result_image = result_image.resize((440, 580))
        self.column_2.image(resized_result_image)
        st.sidebar.markdown("\n")
        st.sidebar.download_button("Download detected image",
                                   self.convert_image(resized_result_image),
                                   "fixed.png", "image/png")


if __name__ == '__main__':
    app = StreamlitApp()
    app.run()

import streamlit as st
from PIL import Image
from io import BytesIO
from utils import bucket_name, s3_client
import uuid

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def upload_to_s3(image, bucket_name, file_name):
    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=image)
        return True
    except Exception as e:
        st.error(f"An error occurred while uploading to S3: {str(e)}")
        return False

def fix_image(upload, col1, col2):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    # fixed = remove(image)
    fixed = image
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)

    # Upload the fixed image to S3
    s3_file_name = str(uuid.uuid4())
    print(s3_file_name)
    if upload_to_s3(convert_image(fixed), bucket_name, s3_file_name):
        st.sidebar.markdown(f"Download the fixed image [here](https://{bucket_name}.s3.amazonaws.com/{s3_file_name})")


def main():
    st.set_page_config(layout="wide", page_title="Deepfake Detector")

    st.write("## Detect if your media has been modified")
    st.write(
        ":dog: Try uploading an image. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/Koruvika/deepfake_detection) on GitHub :grin:"
    )
    st.sidebar.write("## Upload and download :gear:")

    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    col1, col2 = st.columns(2)
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
            st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            fix_image(upload=my_upload, col1=col1, col2=col2)
    else:
        fix_image("./assets/images/face_image.jpg", col1=col1, col2=col2)


if __name__ == '__main__':
    main()
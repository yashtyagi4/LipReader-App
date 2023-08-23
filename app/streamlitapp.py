import streamlit as st
import os
from io import BytesIO
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
from PIL import Image
import base64

# Set the page width to a desired value
st.set_page_config(layout="wide")

def image_to_base64(img: Image.Image) -> str:
    """Converts a PIL image to base64 encoded string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load the Sheldon logo image
image = Image.open('SheldonLogo.png')

# Convert the PIL image to a base64 encoded string
encoded_image = image_to_base64(image)

# Display the centered image using HTML
st.markdown(
    f"""
    <div style="text-align:center">
        <img src="data:image/png;base64,{encoded_image}" width="600">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; background-color: #f2f2f2; border-radius: 10px; padding: 20px; margin: 20px 0;'>
        <h3>Meet Sheldon, an AI model that can read lips and decode sentences. </h3>
        <h5>Share a silent video, and let Sheldon showcase its mesmerizing skills!</h5>
    </div>
    """,
    unsafe_allow_html=True
)


# Generating a list of options or videos
options = os.listdir(os.path.join('..','data','s1'))

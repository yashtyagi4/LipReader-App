import streamlit as st
import os
from io import BytesIO
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
from PIL import Image
import base64

# ---- SET PAGE CONFIGURATION ----
st.set_page_config(layout="wide")

# ---- STYLING ----
st.markdown(
    """
    <style>
        body, .streamlit, .stApp, .stApp .stMarkdown blockquote {
            background-color: #ffffff !important;
        }
        .st-b6 {
            font-size: 20px !important;
            font-weight: bold !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- FUNCTIONS ----
def image_to_base64(img: Image.Image) -> str:
    """Converts a PIL image to base64 encoded string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ---- DISPLAY LOGO ----
image = Image.open('app/SheldonLogo.png')
encoded_image = image_to_base64(image)
st.markdown(
    f"""
    <div style="text-align:center">
        <img src="data:image/png;base64,{encoded_image}" width="400">
    </div>
    """,
    unsafe_allow_html=True
)

# ---- DESCRIPTION ----
st.markdown(
    """
    <div style='text-align: center; background-color: #ffffff; border-radius: 10px; padding: 20px; margin: 20px 0;'>
        <h4 style="color: #000000; font-family: 'Georgia', serif;">An AI model that can read lips and decode sentences. </h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- SELECT VIDEO ----
st.markdown(
    """
    <div style='text-align: center; background-color: #ffffff;'>
        <h6 style="color: #000000; font-family: 'Georgia', serif;">Pick any video below and let Sheldon showcase its mesmerizing skills!</h6>
    </div>
    """,
    unsafe_allow_html=True
)
options = os.listdir(os.path.join('data', 's1'))  # Generating a list of options or videos
selected_video = st.selectbox('', options, index=0, format_func=lambda x: f"🎬 {x}")  # Add prefix for clarity and style
st.text('')
st.text('')
st.text('')

# ---- DISPLAY VIDEO AND PREDICTION ----
col1, col2 = st.columns(2)
if options:
    with col1:
        st.markdown("<h6 style='text-align: center; color: #000000;'>Selected Video</h6>", unsafe_allow_html=True)
        file_path = os.path.join('data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')  

        # Rendering Video
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.markdown("<h6 style='text-align: center; color: #000000;'>What Sheldon sees</h6>", unsafe_allow_html=True)

        # Loading Video
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        video = tf.squeeze(video, axis=-1)
        video_np = (video.numpy() * 255).astype('uint8')
        imageio.mimsave('animation.gif', video_np, duration=50)
        st.image('animation.gif', use_column_width=True)

        # Displaying Model
        st.text('')
        st.text('')
        st.markdown("<div style= 'text-align: center; color: #000000; font-weight: bold;'>Sheldon's Prediction</h6>" , unsafe_allow_html=True)  # Wrapping with div to center align
        with st.spinner('Loading...'):
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.markdown("</div>", unsafe_allow_html=True)  # End of div for center alignment

        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.markdown(
            f"""
            <div style='background-color: #ffe0e0; border-radius: 10px; text-align: center; font-weight: bolder;'>
                <p style="font-family: 'Georgia', serif; font-size: 23px; color: #000; font-weight: bold;">{converted_prediction}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


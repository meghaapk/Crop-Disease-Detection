import streamlit as st
import tensorflow as tf
import numpy as np
from tempfile import NamedTemporaryFile
from PIL import Image   

st.set_page_config(page_title="Crop Disease Identification", layout="wide", page_icon="🌱")
st.header("Crop Disease Identification")
# st.markdown("<h1 style='text-align: center'>Crop Disease Identification</h1>", unsafe_allow_html=True)
image=Image.open('Images/logo.png')
col1, col2, col3 = st.columns([3, 3, 1])
col2.image(image, width=175)
# st.image(image)
st.write("Created by [Kiran](https://github.com/kirandr04) [Megha](https://github.com/meghaapk) [Nidhisha](https://github.com/NidhishaRao).")
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Ever grew a plant and wanted to know if they were growing healthy?</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">You can now identify if your plant has any disease just by scanning your plant leaves.This app predicts the disease of the crop.</p>', unsafe_allow_html=True)

# @st.cache()
def load_model():
    if crop == "Coffee":
        return tf.keras.models.load_model('models/coffee.h5'),["Coffee Miner Disease","Coffee Rust Disease"]
    elif crop == "Pepper":
        return tf.keras.models.load_model('models/pepper.h5'),["Pepper bell Bacterial spot Disease", "Pepperbell healthy"]
    elif crop == "Potato":
        return tf.keras.models.load_model('models/potatoes.h5'),["Early Blight Disease", "Late Blight Disease", "Healthy"]
    else:
        return tf.keras.models.load_model('models/tomatoes.h5'),['Tomato Early Blight Disease', 'Tomato Late Blight Disease', 'Tomato healthy']

def predict(image):
    my_img = tf.keras.utils.load_img(image, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(my_img)
    img_array = tf.expand_dims(img_array, 0)
    model,class_names = load_model()
    pred = model.predict(img_array)
    pred = np.argmax(pred[0])
    return pred,class_names

if __name__ == "__main__":
    st.sidebar.title("Give an appropriate sidebar heading")
    crop = st.sidebar.radio("Select the crop", ("Coffee", "Pepper", "Potato", "Tomato"))
    st.write("##### You selected: {}".format(crop))
    option = st.sidebar.selectbox("Select an option", ["Upload Image", "Capture Image"])
    if option == "Upload Image":
        image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    elif option == "Capture Image":
        image = st.sidebar.camera_input("Upload an image")

    temp_file = NamedTemporaryFile(delete=False)
    if btn := st.sidebar.button("Predict"):
        if image is not None:
            st.image(image)
            with st.spinner("Analyzing..."):
                temp_file.write(image.getvalue())
                pred,class_names = predict(temp_file.name)
                st.balloons()
                st.success("##### The prediction is {}".format(class_names[pred]))
        else:
            st.error("Please upload an image")

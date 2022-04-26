import streamlit as st
import tensorflow as tf
import numpy as np
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Crop Disease Identification", layout="wide", page_icon="🌱")
st.header("Crop Disease Identification")
st.write("Created by [Maggy](https://github.com/meghaapk).")
st.subheader("This app predicts the disease of the crop. Add a proper description here.")
class_names = ["Early Blight", "Late Blight", "Healthy"]

# @st.cache()
def load_model():
    if crop == "Coffee":
        return tf.keras.models.load_model('models/coffee.h5')
    elif crop == "Pepper":
        return tf.keras.models.load_model('models/pepper.h5')
    elif crop == "Potato":
        return tf.keras.models.load_model('models/potatoes.h5')
    else:
        return tf.keras.models.load_model('models/tomatoes.h5')

def predict(image):
    my_img = tf.keras.utils.load_img(image, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(my_img)
    img_array = tf.expand_dims(img_array, 0)
    model = load_model()
    pred = model.predict(img_array)
    pred = np.argmax(pred[0])
    return pred

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
                pred = predict(temp_file.name)
                st.balloons()
                st.success("##### The predicted disease is {}".format(class_names[pred]))
        else:
            st.error("Please upload an image")

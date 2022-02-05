import streamlit as st
import tensorflow as tf
import numpy as np
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Crop Disease Identification", layout="wide", page_icon="🌱")
st.header("Crop Disease Identification")
st.write("Created by [Megha](https://github.com/meghaapk).")
st.subheader("This app predicts the disease of the crop")
class_names = ["Early Blight", "Late Blight", "Healthy"]

# @st.cache()
# def load_model():
#     model = tf.keras.models.load_model('potatoes.h5')
#     return model

def predict(image):
    # my_img = Image.open(image)
    my_img = tf.keras.utils.load_img(image, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(my_img)
    img_array = tf.expand_dims(img_array, 0)
    model = tf.keras.models.load_model('potatoes.h5')
    pred = model.predict(img_array)
    pred = np.argmax(pred[0])
    return pred

if __name__ == "__main__":
    #Ask option to upload image or capture image
    option = st.sidebar.selectbox("Select an option", ["Upload Image", "Capture Image"])
    if option == "Upload Image":
        image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    elif option == "Capture Image":
        image = st.sidebar.camera_input("Upload an image")
    # image = st.sidebar.camera_input("Upload an image")
    # image = st.sidebar.file_uploader("Upload a file", type=["jpg", "png", "jpeg"])
    temp_file = NamedTemporaryFile(delete=False)
    # if image is not None:
    #         temp_file.write(image.getvalue())
    btn = st.sidebar.button("Predict")

    if btn:
        if image is not None:
            st.image(image)
            temp_file.write(image.getvalue())
            pred = predict(temp_file.name)
            st.success("The predicted disease is {}".format(class_names[pred]))
        else:
            st.error("Please upload an image")
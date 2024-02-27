import streamlit as st
from functions import Functions

st.set_page_config(page_title="Classifier", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

st.header("Fabric Pattern Classifier", divider="grey")
st.write("""
This application showcases the implementation of four Convolutional Neural Network (CNN) models trained using transfer learning techniques, developed as part of a master's thesis project. The models are designed to classify fabric patterns with high accuracy and efficiency.
The project aims to leverage pre-trained CNN models and fine-tune them on a custom dataset of fabric patterns, enabling the models to accurately identify various types of patterns with minimal computational resources.
Through this demonstration, users can experience the effectiveness of transfer learning in the context of fabric pattern classification, witnessing the models' capability to discern intricate patterns and make informed predictions.
For further details on the methodology, training process, and evaluation metrics employed in the thesis, please refer to the accompanying documentation or contact the author.
""")

st.subheader("Demo", divider="grey")
st.write("Labels: Polka-dotted, Paisley, Plain, Striped, Chequered, Animal-pattern, and Zigzagged")
with st.expander("Click this to see sample images with labels"):
    st.image("sample_images.png")



model_choice = st.radio("Select a model to be used:",
         ["EfficientNet", "Resnet50", "MobileNet V2", "Inception V3"],
         horizontal=True)

img_file_buffer = st.camera_input("Picture a pattern:")

if img_file_buffer is not None and model_choice is not None:
    selected_model = Functions.load_model(model_choice)
    with st.spinner('Classifying'):
        predicted_label, confidence_level = Functions.preprocess_and_predict_image(img_file_buffer, selected_model, image_size=(224, 224))
    if confidence_level <= 0.95:
        st.error("The model is not confident in classifying the pattern.")
    else:
        st.success(f"The captured image is **{predicted_label.upper()}**")
        st.write(f"Confidence Level: **{round(confidence_level * 100, 2)}%**")



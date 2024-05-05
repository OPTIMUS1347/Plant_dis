import os
import json
from PIL import Image

import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu



page_bg_img="""
<style>
[data-testid="stAppViewContainer"] 
{
background-image:url("https://images.pexels.com/photos/807598/pexels-photo-807598.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2
");
background-size:cover;
}
</style>

"""
st.markdown(page_bg_img, unsafe_allow_html=True)

selected=option_menu(
        menu_title="MAIN MENU",
        options=["User Login","Detection","Pre-Caution","Crop Info","Community","Pest Info"],
        icons=["person-circle","binoculars","emoji-astonished","info-square","people","bug-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
              "container":{"padding":"0!important","background-color":"grey"},
              "icon":{"color":"orange","font-size":"25px"},
              "nav-link":{
                    "font-size":"25px",
                    "text-align":"left",
                    "margin":"0px",
                    "--hover-color":"#eee",
              },  
        
        "nav-link-selected":{"background-color":"green"},    
        },
     )

if selected == "User Login":
        st.title(f"you have selected Admin")
if selected == "Detection":
        st.title(f"you have selected detection")
        working_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
        # Load the pre-trained model
        model = tf.keras.models.load_model(model_path)

        # loading the class names
        class_indices = json.load(open(f"{working_dir}/class_indices.json"))

        # Function to Load and Preprocess the Image using Pillow
        def load_and_preprocess_image(image_path, target_size=(224, 224)):
                # Load the image
                img = Image.open(image_path)
                # Resize the image
                img = img.resize(target_size)
                # Convert the image to a numpy array
                img_array = np.array(img)
                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)
                #Scale the image values to [0, 1]
                img_array = img_array.astype('float32') / 255.
                return img_array


        # Function to Predict the Class of an Image
        def predict_image_class(model, image_path, class_indices):
                preprocessed_img = load_and_preprocess_image(image_path)
                predictions = model.predict(preprocessed_img)
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                predicted_class_name = class_indices[str(predicted_class_index)]
                return predicted_class_name

        # Streamlit App
        st.title('Plant Disease Classifier')

        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
                image = Image.open(uploaded_image)
                col1,col2 = st.columns(2)

                with col1:
                        resized_img = image.resize((150, 150))
                        st.image(resized_img)
                with col2:
                        if st.button('Classify'):
                                # Preprocess the uploaded image and predict the class
                                prediction = predict_image_class(model, uploaded_image, class_indices)
                                st.success(f'Prediction: {str(prediction)}')
if selected == "Pre-Caution":
        st.title(f"you have selected Pre-Caution")
if selected == "Crop Info":
        st.title(f"you have selected Crop-Info")
if selected == "Community":
        st.title(f"you have selected Community")
if selected == "Pest-Info":
        st.title(f"you have selected Pest-Info")

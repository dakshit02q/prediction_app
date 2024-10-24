import streamlit as st
import pickle 
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
#from tensorflow import img_to_array
from PIL import Image
from tensorflow import keras


def get_clean_data():
    data = pd.read_csv("data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis = 1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    print(data.head())
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ] 

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label = label,
            min_value = float(data[key].min()),
            max_value = float(data[key].max()),
            value = float(data[key].mean())
        )
    
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()

    x = data.drop(['diagnosis'], axis = 1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = x[key].max()
        min_val = x[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r = [input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']],
          theta = categories,
          fill = 'toself',
          name = 'Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open(r'model.pkl', 'rb'))
    scaler = pickle.load(open(r'scaler.pkl', 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1,-1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)

    st.subheader("Prediction")
    st.write("The prediction is:")
    if prediction[0] == 0: 
        st.write("Benign")
    else:
        st.write("Malignant")
    

    st.write("Probability of being benign: ", model.predict_proba(input_scaled)[0][0])
    st.write("probability of being malignant: ", model.predict_proba(input_scaled)[0][1])
    st.write("This app is only toabe used by professionals with the reports in hand, do not use this as the standard.")

def cnn_predict(input_xray):


    if input_xray is not None:
        
        # Display the image with its filename
        xray_image = Image.open(input_xray)
        xray_image = xray_image.resize((64,64))
        img_array = np.array(xray_image)
        img_array = np.expand_dims(img_array, axis = 0)
        
        img_array = img_array / 255.0
        cnn_model = tf.keras.models.load_model('cnn_model.h5')
        prediction_cnn = cnn_model.predict(img_array)

        if prediction_cnn[0][0] < 0.5:
            st.write("There are tumours present")
        else:
            st.write("The tumour is absent")

        st.write(f"the probability of tumour being absent: {prediction_cnn[0][0]}")
        st.write(f"the probability of tumour being present: {prediction_cnn[0][1]}")

        
def display_image(input_xray):
    if input_xray is not None:
        bytes_data = input_xray.read()

        st.image(bytes_data, caption = "Your X-Ray")



def main():
    st.set_page_config(
        page_title = 'cancer predictor',
        page_icon = ':doctor',
        layout = 'wide',
        initial_sidebar_state = 'expanded')
    

    
    input_data = add_sidebar()
    #st.write(input_data)

    with st.container():
        st.title('Cancer Predictor')
        
    col1 , col2 = st.columns([4,1])
    
    with col1: 
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2: 
        add_predictions(input_data)

    with st.container():
        st.title("X-ray analyser")

    st.text("Note: This is not a definite determiner of the disease, it is trained on previous patterns.")
    xray = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"] )


    if st.button(label = "Analyse"):
        display_image(xray)
        st.subheader("Likeliness of a tumour:")
        cnn_predict(xray)
        
    

if __name__ == '__main__':
    main()


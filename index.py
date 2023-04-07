import streamlit as st # To create interface
import nibabel as nib # To read images
import numpy as np # Operations
import matplotlib.pyplot as plt # Plots

st.title("Procesamiento Digital de Imágenes")
# st.write("Escribe el nombre de la imagen a procesar")

# Get image
image_input = st.text_input("Ingresa el nombre de la imagen")
image_data = nib.load(f'./images/1/{image_input}.gz')
image = image_data.get_fdata()

segmentation_options = ['Thresholding', 'Region Growing', 'K-means']
selected_segmentation_option = st.radio('Selecciona una técnica de segmentación', segmentation_options)

# Create three columns for the inputs
col1, col2, col3 = st.columns(3)

# Add a numeric input field to each column
with col1:
    input1 = st.number_input("Input 1", value=0, step=1)
with col2:
    input2 = st.number_input("Input 2", value=0, step=1)
with col3:
    input3 = st.number_input("Input 3", value=0, step=1)

if selected_segmentation_option == 'Thresholding':

    st.write('You selected Option 1')
elif selected_segmentation_option == 'Region Growing':
    st.write('You selected Option 2')
else:
    st.write('You selected Option 3')

print(image)

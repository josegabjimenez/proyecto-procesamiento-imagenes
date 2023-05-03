import streamlit as st # To create interface
import nibabel as nib # To read images
import numpy as np # Operations
import matplotlib.pyplot as plt # Plots

# Algorithms
from algorithms.segmentation import thresholding, region_growing, clustering


import os

st.title("Procesamiento Digital de Imágenes")

# # Create a file uploader widget
# uploaded_file = st.file_uploader("Selecciona una imagen")

image = st.session_state["image"]

# If a file was uploaded
if image is not None:

    st.markdown("## Visualización de la imagen")

    # Create two columns for the axis inputs
    col1, col2 = st.columns(2)

    # Add a dropdown to select an axis and a slider to select an index
    with col1:
        filenames = ["Eje X", "Eje Y", "Eje Z"]
        axis_selected = st.selectbox("Selecciona un eje", filenames)

        if axis_selected == "Eje X":
            axis_shape = image.shape[0]

        if axis_selected == "Eje Y":
            axis_shape = image.shape[1]

        if axis_selected == "Eje Z":
            axis_shape = image.shape[2]


    with col2:
        axis_value = st.slider(label="Posición", min_value=0, max_value=(axis_shape-1), step=1, value=1)
        # axis_value = st.slider(label="Posición", 0, axis_shape, step=1 )
    
    # Axis adjusment
    axisX = slice(None)
    axisY = slice(None)
    axisZ = slice(None)

    if axis_selected == "Eje X":
        axisX = axis_value
        axisY = slice(None)
        axisZ = slice(None)
    if axis_selected == "Eje Y":
        axisY = axis_value
        axisX = slice(None)
        axisZ = slice(None)
    if axis_selected == "Eje Z":
        axisZ = axis_value
        axisX = slice(None)
        axisY = slice(None)

    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(image[axisX, axisY, axisZ])
    st.pyplot(fig)

    # Segmentation
    st.markdown("## Segmentación")

    segmentation_options = ['Thresholding', 'Region Growing', 'Clustering']
    selected_segmentation_option = st.radio('Selecciona una técnica de segmentación', segmentation_options)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if selected_segmentation_option == 'Thresholding':
        tol = st.number_input("Selecciona una tolerancia::", 0, None, 1, 1)
        tau = st.number_input("Selecciona un TAU:", 0, None, 20, 1)

    if selected_segmentation_option == 'Clustering':
        k = st.number_input("Selecciona número de grupos", 0, None, 3, 1)

    # Create segmentation button
    segmentation_button_clicked = st.button("Crear segmentación")

    # Algorithms

    if selected_segmentation_option == 'Thresholding' and segmentation_button_clicked:
        print("TAU:", tau)
        print("tol:", tol)
        # Create the plot using imshow
        image_segmentated = thresholding(image, tol, tau)

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_segmentated[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)
        
    elif selected_segmentation_option == 'Region Growing' and segmentation_button_clicked:
        # Create the plot using imshow
        image_segmentated = region_growing(image)

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_segmentated[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)

    elif selected_segmentation_option == 'Clustering' and segmentation_button_clicked:
        # Create the plot using imshow
        image_segmentated = clustering(image, k)

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_segmentated[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)

    # print(image)

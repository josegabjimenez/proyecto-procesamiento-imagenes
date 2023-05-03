import streamlit as st # To create interface
import nibabel as nib # To read images
import numpy as np # Operations
import matplotlib.pyplot as plt # Plots
import os

st.title("Preprocesamiento")

# Create a file uploader widget
uploaded_file = st.file_uploader("Selecciona una imagen")

# If a file was uploaded
if uploaded_file is not None:
    # Get the contents of the file as a byte string
    file_contents = uploaded_file.getvalue()

    # # Set the file extension based on the uploaded file type
    file_extension = os.path.splitext(uploaded_file.name)[1]

    if (file_extension==".gz"):
        file_extension = ".nii.gz"
    else: 
        file_extension = ".nii"

    # Save the byte string to a temporary file with the correct extension
    with open("./temp_images/temp_file" + file_extension, "wb") as f:
        f.write(file_contents)

    # Get the path of the temporary file
    path = os.path.abspath("./temp_images/temp_file" + file_extension)

    # Load the NIfTI image using nibabel
    image_data = nib.load(path)
    image = image_data.get_fdata()

    st.session_state["image"] = image   

    # Image was uploaded
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

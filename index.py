import streamlit as st # To create interface
import nibabel as nib # To read images
import numpy as np # Operations
import matplotlib.pyplot as plt # Plots
import os

st.title("Procesamiento Digital de Imágenes")

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
    with open("temp_file" + file_extension, "wb") as f:
        f.write(file_contents)

    # Get the path of the temporary file
    path = os.path.abspath("temp_file" + file_extension)

    # Load the NIfTI image using nibabel
    image_data = nib.load(path)
    image = image_data.get_fdata()

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
        print(axis_shape)
        axis_value = st.slider(label="Posición", min_value=0, max_value=axis_shape, step=1, value=1)
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

    # Create a container
    container = st.container()

    segmentation_options = ['Thresholding', 'Region Growing', 'Clustering']
    selected_segmentation_option = st.radio('Selecciona una técnica de segmentación', segmentation_options)

    # Create segmentation button
    segmentation_button_clicked = st.button("Crear segmentación")

    # Algorithms
    def thresholding(image, tol = 1, tau = 20):

        while True:
            # print(tau)

            segmentation = image >= tau
            mBG = image[np.multiply(image > 10, segmentation == 0)].mean()
            mFG = image[np.multiply(image > 10, segmentation == 1)].mean()

            tau_post = 0.5 * (mBG + mFG)

            if np.abs(tau - tau_post) < tol:
                break
            else:
                tau = tau_post

        return segmentation

    def clustering(image, tol = 1, tau = 150):
        k1 = np.amin(image)
        k2 = np.mean(image)
        k3 = np.amax(image)
        # print(k1, k2, k3)

        for i in range(0,3):
            d1 = np.abs(k1 - image)
            d2 = np.abs(k2 - image)
            d3 = np.abs(k3 - image)

            segmentation = np.zeros_like(image)
            segmentation[np.multiply(d1 < d2, d1 < d3)] = 0
            segmentation[np.multiply(d2 < d1, d2 < d3)] = 1
            segmentation[np.multiply(d3 < d1, d3 < d2)] = 2

            k1 = image[segmentation == 0].mean()
            k2 = image[segmentation == 1].mean()
            k3 = image[segmentation == 2].mean()
        
        return segmentation

    if selected_segmentation_option == 'Thresholding' and segmentation_button_clicked:
        # Create the plot using imshow
        image_segmentated = thresholding(image)

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_segmentated[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)
        
    elif selected_segmentation_option == 'Region Growing' and segmentation_button_clicked:
        # Create the plot using imshow
        image_segmentated = clustering(image)

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_segmentated[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)

    elif selected_segmentation_option == 'Clustering' and segmentation_button_clicked:
        # Create the plot using imshow
        image_segmentated = clustering(image)

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_segmentated[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)

    # print(image)

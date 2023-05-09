import streamlit as st  # To create interface
import nibabel as nib  # To read images
import numpy as np  # Operations
import matplotlib.pyplot as plt  # Plots
import os

# Standarization algorithms
from algorithms.standarization import rescaling, z_score, white_stripe

# Global variables

# Axis adjusment
if "axis_selected" not in st.session_state:
    st.session_state["axis_selected"] = "Eje X"

if "axisX" not in st.session_state:
    st.session_state["axisX"] = slice(None)

if "axisY" not in st.session_state:
    st.session_state["axisY"] = slice(None)

if "axisZ" not in st.session_state:
    st.session_state["axisZ"] = slice(None)

image = st.session_state["image"]


st.title("Preprocesamiento")

uploaded_file = st.file_uploader("Selecciona una imagen")

# If a file was uploaded
if uploaded_file is not None:
    # Get the contents of the file as a byte string
    file_contents = uploaded_file.getvalue()

    # # Set the file extension based on the uploaded file type
    file_extension = os.path.splitext(uploaded_file.name)[1]

    if file_extension == ".gz":
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
    st.session_state["axisX"] = slice(None)
    st.session_state["axisY"] = slice(None)
    st.session_state["axisZ"] = slice(None)
    st.session_state["axis_selected"] = "Eje X"

# If a file was uploaded
if image is not None:
    # Image was uploaded
    st.markdown("## Visualización de la imagen")

    # Create two columns for the axis inputs
    col1, col2 = st.columns(2)

    # Add a dropdown to select an axis and a slider to select an index
    with col1:
        filenames = ["Eje X", "Eje Y", "Eje Z"]

        axis_selected = st.session_state.get("axis_selected", "Eje X")
        axis_selected = st.selectbox(
            "Selecciona un eje", filenames, index=filenames.index(axis_selected)
        )

        if axis_selected == "Eje X":
            axis_shape = image.shape[0]
            st.session_state["axis_selected"] = "Eje X"
            default_value = st.session_state["axisX"]

        if axis_selected == "Eje Y":
            axis_shape = image.shape[1]
            st.session_state["axis_selected"] = "Eje Y"
            default_value = st.session_state["axisY"]

        if axis_selected == "Eje Z":
            axis_shape = image.shape[2]
            st.session_state["axis_selected"] = "Eje Z"
            default_value = st.session_state["axisZ"]

    with col2:
        if type(default_value) != int:
            default_value = 0

        axis_value = st.slider(
            label="Posición",
            min_value=0,
            max_value=(axis_shape - 1),
            step=1,
            value=default_value,
        )
        # axis_value = st.slider(label="Posición", 0, axis_shape, step=1 )

    axisX = st.session_state["axisX"]
    axisY = st.session_state["axisY"]
    axisZ = st.session_state["axisZ"]

    if axis_selected == "Eje X":
        axisX = axis_value
        axisY = slice(None)
        axisZ = slice(None)
        st.session_state["axisX"] = axis_value
        st.session_state["axisY"] = slice(None)
        st.session_state["axisZ"] = slice(None)
    if axis_selected == "Eje Y":
        axisY = axis_value
        axisX = slice(None)
        axisZ = slice(None)
        st.session_state["axisY"] = axis_value
        st.session_state["axisX"] = slice(None)
        st.session_state["axisZ"] = slice(None)
    if axis_selected == "Eje Z":
        axisZ = axis_value
        axisX = slice(None)
        axisY = slice(None)
        st.session_state["axisZ"] = axis_value
        st.session_state["axisX"] = slice(None)
        st.session_state["axisY"] = slice(None)

    # Plot image
    fig, ax = plt.subplots()
    ax.set_xlim([0, image.shape[0]])
    ax.set_ylim([0, image.shape[1]])
    ax.imshow(image[axisX, axisY, axisZ])
    st.pyplot(fig)

    # ------------------------------------------
    # Standarization section
    st.markdown("## Estandarización")

    standarization_options = [
        "Rescaling",
        "Z-score",
        "White Stripe",
        "Histogram Matching",
    ]
    selected_standarization_option = st.radio(
        "Selecciona una técnica de estandarización", standarization_options
    )
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )

    # Create standardization button
    standardization_button_clicked = st.button("Generar estandarización")

    # Algorithms
    if selected_standarization_option == "Rescaling" and standardization_button_clicked:
        # Apply algorithm
        image_standardized = rescaling(image)

        # Set new image to state
        st.session_state["image"] = image_standardized

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_standardized[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)

    elif selected_standarization_option == "Z-score" and standardization_button_clicked:
        # Apply algorithm
        image_standardized = z_score(image)

        # Set new image to state
        st.session_state["image"] = image_standardized

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_standardized[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)

    elif (
        selected_standarization_option == "White Stripe"
        and standardization_button_clicked
    ):
        # Apply algorithm
        image_standardized = white_stripe(image)

        # Set new image to state
        st.session_state["image"] = image_standardized

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_standardized[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)

    elif (
        selected_standarization_option == "Histogram Matching"
        and standardization_button_clicked
    ):
        # Apply algorithm
        image_standardized = white_stripe(image)

        # Set new image to state
        st.session_state["image"] = image_standardized

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image_standardized[axisX, axisY, axisZ])

        # Display the plot using Streamlit
        st.pyplot(fig)

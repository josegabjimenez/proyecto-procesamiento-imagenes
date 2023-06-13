import streamlit as st  # To create interface
import nibabel as nib  # To read images
import numpy as np  # Operations
import matplotlib.pyplot as plt  # Plots
import os

# Standarization algorithms
from algorithms.standarization import rescaling, z_score, white_stripe, histogram_matching

# Denoise algorithms
from algorithms.denoise import mean_filter, median_filter, edge_filter

# Registration algorithms
from algorithms.registration import registration

# Global variables

# Axis adjusment
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "images" not in st.session_state:
    st.session_state["images"] = []

if "image_standardized" not in st.session_state:
    st.session_state["image_standardized"] = []

if "image_denoised" not in st.session_state:
    st.session_state["image_denoised"] = []

if "image_registered" not in st.session_state:
    st.session_state["image_registered"] = []

if "image_segmented" not in st.session_state:
    st.session_state["image_segmented"] = []

if "image_border_detected" not in st.session_state:
    st.session_state["image_border_detected"] = []

if "selected_image_index" not in st.session_state:
    st.session_state["selected_image_index"] = 0

if "axis_selected" not in st.session_state:
    st.session_state["axis_selected"] = "Eje X"

if "axisX" not in st.session_state:
    st.session_state["axisX"] = slice(None)

if "axisY" not in st.session_state:
    st.session_state["axisY"] = slice(None)

if "axisZ" not in st.session_state:
    st.session_state["axisZ"] = slice(None)

images = st.session_state["images"]

st.title("Preprocesamiento")

uploaded_file = st.file_uploader("Sube una o varias imágenes", accept_multiple_files=True)

# Verify if a file was uploaded before
if st.session_state["uploaded_files"] != []:
    uploaded_file = st.session_state["uploaded_files"]

# If a file was uploaded
if uploaded_file != []:

    # If there is no image in the state, create an empty array
    if st.session_state["images"] == []:
        st.session_state["images"] = [None] * len(uploaded_file)
        st.session_state["image_standardized"] = [None] * len(uploaded_file)
        st.session_state["image_denoised"] = [None] * len(uploaded_file)
        st.session_state["image_registered"] = [None] * len(uploaded_file)
        st.session_state["image_segmented"] = [None] * len(uploaded_file)
        st.session_state["image_border_detected"] = [None] * len(uploaded_file)


    st.session_state["uploaded_files"] = uploaded_file
    image_options = {
        i: uploaded_file[i].name for i in range(0, len(uploaded_file))
    }
    selected_image_index = st.radio(
        "Selecciona la imagen que vas a usar", options=image_options.keys(), horizontal=True, format_func=lambda x: image_options[x].split(".")[0], index=st.session_state["selected_image_index"]
    )
    st.session_state["selected_image_index"] = selected_image_index
    st.write(selected_image_index)
    reset_button_clicked = st.button("Resetear imágenes")

    # For each image, load it and save it to the state
    for i in range(0, len(uploaded_file)):
        # Verify if the image is already loaded 
        if st.session_state["images"][i] is None or reset_button_clicked:
            # Get the contents of the file as a byte string
            file_contents = uploaded_file[i].getvalue()

            # # Set the file extension based on the uploaded file type
            file_extension = os.path.splitext(uploaded_file[i].name)[1]

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
            imageToLoad = image_data.get_fdata()

            st.session_state["images"][i] = imageToLoad

            st.session_state["image_standardized"][i] = None
            st.session_state["image_denoised"][i] = None
            st.session_state["axisX"] = slice(None)
            st.session_state["axisY"] = slice(None)
            st.session_state["axisZ"] = slice(None)
            st.session_state["axis_selected"] = "Eje X"

            #update images
            images = st.session_state["images"]

# If a file was uploaded
if images != []:
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
            axis_shape = images[selected_image_index].shape[0]
            st.session_state["axis_selected"] = "Eje X"
            default_value = st.session_state["axisX"]

        if axis_selected == "Eje Y":
            axis_shape = images[selected_image_index].shape[1]
            st.session_state["axis_selected"] = "Eje Y"
            default_value = st.session_state["axisY"]

        if axis_selected == "Eje Z":
            axis_shape = images[selected_image_index].shape[2]
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
    ax.set_xlim([0, images[selected_image_index].shape[0]])
    ax.set_ylim([0, images[selected_image_index].shape[1]])
    ax.imshow(images[selected_image_index][axisX, axisY, axisZ], cmap="gray")
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
        image_standardized = rescaling(images[selected_image_index])

        # Set new image to state
        st.session_state["images"][selected_image_index] = image_standardized
        st.session_state["image_standardized"][selected_image_index] = image_standardized

    elif selected_standarization_option == "Z-score" and standardization_button_clicked:
        # Apply algorithm
        image_standardized = z_score(images[selected_image_index])

        # Set new image to state
        st.session_state["images"][selected_image_index] = image_standardized
        st.session_state["image_standardized"][selected_image_index] = image_standardized

    elif (
        selected_standarization_option == "White Stripe"
        and standardization_button_clicked
    ):
        # Apply algorithm
        image_standardized = white_stripe(images[selected_image_index])

        # Set new image to state
        st.session_state["images"][selected_image_index] = image_standardized
        st.session_state["image_standardized"][selected_image_index] = image_standardized

    elif (
        selected_standarization_option == "Histogram Matching"
        and standardization_button_clicked
    ):
        # Apply algorithm
        image_standardized = histogram_matching(images[selected_image_index])

        # Set new image to state
        st.session_state["images"][selected_image_index] = image_standardized
        st.session_state["image_standardized"][selected_image_index] = image_standardized

    # Plot Standardized image if exists
    image_standardized = st.session_state["image_standardized"][selected_image_index]
    if image_standardized is not None:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Show image in the first subplot
        ax1.set_xlim([0, images[selected_image_index].shape[0]])
        ax1.set_ylim([0, images[selected_image_index].shape[1]])
        ax1.imshow(image_standardized[axisX, axisY, axisZ], cmap="gray")
        ax1.set_title('Imagen estandarizada')

        # Show intensities histogram in the second subplot
        ax2.hist(image_standardized.flatten(), bins=50)
        ax2.set_title('Histograma de intensidades')

        # Display the plot using Streamlit
        st.pyplot(fig1)

    # ------------------------------------------
    # Denoise section
    st.markdown("## Reducción de ruido")

    denoise_options = [
        "Mean Filter",
        "Median Filter",
        "Edge Filter",
    ]
    selected_denoise_option = st.radio(
        "Selecciona una técnica de reducción de ruido", denoise_options
    )
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )

    if selected_denoise_option == "Edge Filter":
        tol = st.number_input("Tolerancia:", 0.0, None, 50.0, 1.0)

    # Create denoise button
    denoise_button_clicked = st.button("Generar reducción de ruido")

    # Algorithms
    if selected_denoise_option == "Mean Filter" and denoise_button_clicked:
        # Apply algorithm
        image_denoised = mean_filter(images[selected_image_index])

        # Set new image to state
        st.session_state["images"][selected_image_index] = image_denoised
        st.session_state["image_denoised"][selected_image_index] = image_denoised

    elif selected_denoise_option == "Median Filter" and denoise_button_clicked:
        # Apply algorithm
        image_denoised = median_filter(images[selected_image_index])

        # Set new image to state
        st.session_state["images"][selected_image_index] = image_denoised
        st.session_state["image_denoised"][selected_image_index] = image_denoised

    elif selected_denoise_option == "Edge Filter" and denoise_button_clicked:
        # Apply algorithm
        image_denoised = edge_filter(images[selected_image_index], tol)

        # Set new image to state
        st.session_state["images"][selected_image_index] = image_denoised
        st.session_state["image_denoised"][selected_image_index] = image_denoised

    # Plot Denoised image if exists
    image_denoised_plot = st.session_state["image_denoised"][selected_image_index]
    if image_denoised_plot is not None:
        # Plot image
        fig2, ax2 = plt.subplots()
        ax2.set_xlim([0, images[selected_image_index].shape[0]])
        ax2.set_ylim([0, images[selected_image_index].shape[1]])
        ax2.imshow(image_denoised_plot[axisX, axisY, axisZ], cmap="gray")

        # Display the plot using Streamlit
        st.pyplot(fig2)


    # ------------------------------------------
    # Registration section
    st.markdown("## Registro de imágenes")

    # Create two columns for the axis inputs
    col1, col2 = st.columns(2)

    # Add a dropdown to select an axis and a slider to select an index
    with col1:

        fixed_image_selected = st.selectbox(
            "Selecciona la imagen fija", options=image_options.keys(), format_func=lambda x: image_options[x].split(".")[0], index=0
        )



    with col2:
        moving_image_selected = st.selectbox(
            "Selecciona la imagen móvil", options=image_options.keys(), format_func=lambda x: image_options[x].split(".")[0], index=0
        )
    
    # Create registration button
    registration_button_clicked = st.button("Generar registro")

    # Algorithms
    if registration_button_clicked:
        # Apply algorithm
        image_registered = registration(images[fixed_image_selected], images[moving_image_selected])

        # Set new image to state
        st.session_state["images"][selected_image_index] = image_registered
        st.session_state["image_registered"][selected_image_index] = image_registered
    
    # Plot Registered image if exists
    image_registered_plot = st.session_state["image_registered"][selected_image_index]
    if image_registered_plot is not None:
        # Plot image
        fig3, ax3 = plt.subplots()
        ax3.set_xlim([0, images[selected_image_index].shape[0]])
        ax3.set_ylim([0, images[selected_image_index].shape[1]])
        ax3.imshow(image_registered_plot[axisX, axisY, axisZ], cmap="gray")

        # Display the plot using Streamlit
        st.pyplot(fig3)
import streamlit as st  # To create interface
import nibabel as nib  # To read images
import numpy as np  # Operations
import matplotlib.pyplot as plt  # Plots
import os
import io
import tempfile

from algorithms.registration import count_volume
from algorithms.utils import remove_brain

if "images" not in st.session_state:
    st.warning("No ha seleccionado una imagen")

if "final_image" not in st.session_state:
    st.session_state["final_image"] = None

st.title("Procesamiento Digital de Imágenes")

images = st.session_state["images"]
selected_image_index = st.session_state["selected_image_index"]
uploaded_file = st.session_state["uploaded_files"]

# If a file was uploaded
if images != []:
    st.markdown("## Visualización de la imagen")

    image_options = {i: uploaded_file[i].name for i in range(0, len(uploaded_file))}
    selected_image_index = st.radio(
        "Selecciona la imagen que vas a usar",
        options=image_options.keys(),
        horizontal=True,
        format_func=lambda x: image_options[x].split(".")[0],
        index=st.session_state["selected_image_index"],
    )
    st.session_state["selected_image_index"] = selected_image_index

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

    # Axis adjusment
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

    st.write(uploaded_file[0])
    # # Save each image as a NIfTI file
    for i in range(0, len(uploaded_file)):
        imageUploaded = nib.load(os.path.join("images/1", uploaded_file[i].name))
        affine = imageUploaded.affine
        # Create a nibabel image object from the image data
        image = nib.Nifti1Image(images[i].astype(np.float32), affine)
        # Save the image as a NIfTI file
        output_path = os.path.join("temp_images", uploaded_file[i].name)
        nib.save(image, output_path)

    # Read the .nii image file
    with open(
        os.path.join("temp_images", uploaded_file[selected_image_index].name), "rb"
    ) as file:
        nii_data = file.read()

    # Set the Streamlit download button
    st.download_button(
        label="Download NIfTI Image",
        data=nii_data,
        file_name=uploaded_file[selected_image_index].name,
    )

    # Remove brain
    remove_brain_button = st.button("Quitar el craneo")

    if remove_brain_button:
        final_image = remove_brain()
        st.session_state["final_image"] = final_image

    # If final image exist, plot it
    final_image = st.session_state["final_image"]
    if final_image is not None:
        # Plot image
        fig, ax = plt.subplots()
        ax.set_xlim([0, final_image.shape[0]])
        ax.set_ylim([0, final_image.shape[1]])
        ax.imshow(final_image[axisX, axisY, axisZ], cmap="gray")
        st.pyplot(fig)

    # Generate volumes
    generate_volume_button = st.button("Generar volumen")

    if generate_volume_button:
        # Plot image
        volumes = count_volume("FLAIR_skull_lesion.nii.gz")

        for i in range(0, len(volumes)):
            st.write(
                f"LABEL #{i} - Cantidad de voxeles: {volumes[i][0]} - volumen en mm3: {volumes[i][1].round(2)}"
            )

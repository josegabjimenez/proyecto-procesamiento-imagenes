import streamlit as st  # To create interface
import nibabel as nib  # To read images
import numpy as np  # Operations
import matplotlib.pyplot as plt  # Plots
import io


# Algorithms
from algorithms.segmentation import (
    thresholding,
    region_growing,
    clustering,
    k_means,
    gmm,
    GMM,
)
from algorithms.border_detection import finite_differences
from algorithms.registration import save_image

# from streamlit.uploaded_file_manager import UploadedFile


import os

# Global Variables
if "axisX" not in st.session_state:
    st.session_state["axisX"] = slice(None)

if "axisY" not in st.session_state:
    st.session_state["axisY"] = slice(None)

if "axisZ" not in st.session_state:
    st.session_state["axisZ"] = slice(None)

if "images" not in st.session_state:
    st.warning("No ha seleccionado una imagen")


st.title("Procesamiento Digital de Imágenes")

images = st.session_state["images"]
selected_image_index = st.session_state["selected_image_index"]
uploaded_file = st.session_state["uploaded_files"]

# If a file was uploaded
if images != []:
    # Select image to use
    image_options = {i: uploaded_file[i].name for i in range(0, len(uploaded_file))}
    selected_image_index = st.radio(
        "Selecciona la imagen que vas a usar",
        options=image_options.keys(),
        horizontal=True,
        format_func=lambda x: image_options[x].split(".")[0],
        index=st.session_state["selected_image_index"],
    )
    st.session_state["selected_image_index"] = selected_image_index
    st.write(selected_image_index)

    # Segmentation --------------------------------------------------
    st.markdown("## Segmentación")

    segmentation_options = ["Thresholding", "Region Growing", "Clustering", "GMM"]
    selected_segmentation_option = st.radio(
        "Selecciona una técnica de segmentación", segmentation_options
    )
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )

    if selected_segmentation_option == "Thresholding":
        tol = st.number_input("Selecciona una tolerancia:", 0.0, None, 1.0)
        tau = st.number_input("Selecciona un TAU:", 0.0, None, 20.0, 1.0)

    if selected_segmentation_option == "Region Growing":
        origin_x = st.number_input("Origin X:", 0, None, 100, 1)
        origin_y = st.number_input("Origin Y:", 0, None, 100, 1)
        origin_z = st.number_input("Origin Z:", 0, None, 20, 1)
        tol = st.number_input("Tolerancia:", 0, None, 50, 1)

    if selected_segmentation_option == "Clustering":
        k = st.number_input("Selecciona número de grupos", 0, None, 3, 1)
        iterations = st.number_input("Selecciona número de iteraciones", 0, None, 15, 1)

    if selected_segmentation_option == "GMM":
        k = st.number_input("Selecciona número de grupos", 0, None, 3, 1)
        # iterations = st.number_input("Selecciona número de iteraciones", 0, None, 3, 1)
    # Create segmentation button
    segmentation_button_clicked = st.button("Crear segmentación")

    # Algorithms

    if selected_segmentation_option == "Thresholding" and segmentation_button_clicked:
        # Apply algorithm
        image_segmentated = thresholding(images[selected_image_index], tol, tau)

        # Set new image to state
        # st.session_state["images"][selected_image_index] = image_segmentated
        st.session_state["image_segmented"][selected_image_index] = {
            "name": "segmented_" + uploaded_file[selected_image_index].name,
            "image": image_segmentated,
        }

    elif (
        selected_segmentation_option == "Region Growing" and segmentation_button_clicked
    ):
        # Apply algorithm
        image_segmentated = region_growing(images[selected_image_index])

        # Set new image to state
        # st.session_state["images"][selected_image_index] = image_segmentated
        st.session_state["image_segmented"][selected_image_index] = {
            "name": "segmented_" + uploaded_file[selected_image_index].name,
            "image": image_segmentated,
        }

    elif selected_segmentation_option == "Clustering" and segmentation_button_clicked:
        # Apply algorithm
        # image_segmentated = clustering(image, k, iterations)
        image_segmentated = k_means(images[selected_image_index], k, iterations)

        # Set new image to state
        # st.session_state["images"][selected_image_index] = image_segmentated
        st.session_state["image_segmented"][selected_image_index] = {
            "name": "segmented_" + uploaded_file[selected_image_index].name,
            "image": image_segmentated,
        }

    elif selected_segmentation_option == "GMM" and segmentation_button_clicked:
        # Apply algorithm
        # image_segmentated = GMM(image)
        image_segmentated = gmm(images[selected_image_index], k)

        # Set new image to state
        # st.session_state["images"][selected_image_index] = image_segmentated
        st.session_state["image_segmented"][selected_image_index] = {
            "name": "segmented_" + uploaded_file[selected_image_index].name,
            "image": image_segmentated,
        }

    # Plot segmented image if exists
    image_segmented = st.session_state["image_segmented"][selected_image_index]
    if image_segmented is not None:
        image_segmented = image_segmented["image"]

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
                axis_shape = image_segmented.shape[0]
                st.session_state["axis_selected"] = "Eje X"
                default_value = st.session_state["axisX"]

            if axis_selected == "Eje Y":
                axis_shape = image_segmented.shape[1]
                st.session_state["axis_selected"] = "Eje Y"
                default_value = st.session_state["axisY"]

            if axis_selected == "Eje Z":
                axis_shape = image_segmented.shape[2]
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
        ax.set_xlim([0, image_segmented.shape[0]])
        ax.set_ylim([0, image_segmented.shape[1]])
        ax.imshow(image_segmented[axisX, axisY, axisZ], cmap="gray")
        # Display the plot using Streamlit
        st.pyplot(fig)

        # Save segmented image to temp files
        save_image(
            image_segmented,
            "segmented_" + uploaded_file[selected_image_index].name,
            uploaded_file[selected_image_index].name,
        )

    # ------------------------------------------
    # Border detection section
    st.markdown("## Detección de bordes")

    border_detection_options = [
        "Diferencias finitas",
    ]
    selected_border_detection_option = st.radio(
        "Selecciona una técnica de detección de bordes", border_detection_options
    )
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )

    # Create border detection button
    border_detection_button_clicked = st.button("Generar detección de bordes")

    # Algorithms
    if (
        selected_border_detection_option == "Diferencias finitas"
        and border_detection_button_clicked
    ):
        # Apply algorithm
        image_border_detected = finite_differences(images[selected_image_index])

        # Set new image to state
        # st.session_state["image"] = image_border_detected
        st.session_state["image_border_detected"][
            selected_image_index
        ] = image_border_detected

    # Plot Border detected image if exists
    image_border_detected = st.session_state["image_border_detected"][
        selected_image_index
    ]
    if image_border_detected is not None:
        # Plot image
        fig2, ax2 = plt.subplots()
        ax2.set_xlim([0, images[selected_image_index].shape[0]])
        ax2.set_ylim([0, images[selected_image_index].shape[1]])
        ax2.imshow(image_border_detected[axisX, axisY, axisZ], cmap="gray")

        save_image(image_border_detected, uploaded_file[selected_image_index].name)

        # Display the plot using Streamlit
        st.pyplot(fig2)

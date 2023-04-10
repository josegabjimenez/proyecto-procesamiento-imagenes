import streamlit as st # To create interface
import nibabel as nib # To read images
import numpy as np # Operations
import matplotlib.pyplot as plt # Plots

st.title("Procesamiento Digital de Imágenes")

filenames = ["T1.nii", "IR.nii", "FLAIR.nii"]

# Get image
image_input = st.selectbox("Selecciona una imagen", filenames)
image_data = nib.load(f'./images/1/{image_input}.gz')
image = image_data.get_fdata()

# Create three columns for the inputs
col1, col2, col3 = st.columns(3)

# Add a numeric input field to each column
with col1:
    axisX = st.number_input("Eje X", value=-1, step=1)
with col2:
    axisY = st.number_input("Eje Y", value=-1, step=1)
with col3:
    axisZ = st.number_input("Eje Z", value=0, step=1)

if sum([axisX <= -1, axisY <= -1, axisZ <= -1]) != 2:
    st.warning("Dos ejes deben tener exáctamente el valor de -1")

if axisX == -1:
    axisX = slice(None)
if axisY == -1:
    axisY = slice(None)
if axisZ == -1:
    axisZ = slice(None)

segmentation_options = ['Thresholding', 'Region Growing', 'K-means']
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


if selected_segmentation_option == 'Thresholding' and segmentation_button_clicked:
    # Create the plot using imshow
    print("AXIS",axisX, axisY, axisZ)
    image_segmentated = thresholding(image)

    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(image_segmentated[axisX, axisY, axisZ])

    # Display the plot using Streamlit
    st.pyplot(fig)
    
elif selected_segmentation_option == 'Region Growing':
    st.write('You selected Option 2')
else:
    st.write('You selected Option 3')

# print(image)

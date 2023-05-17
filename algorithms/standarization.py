from scipy.signal import find_peaks
import numpy as np
import nibabel as nib  # To read images
import os


# Rescaling standarization algorithm
def rescaling(image):
    # Get reference image to standardise original image
    path = os.path.abspath("./images/1/FLAIR.nii.gz")
    reference_image = nib.load(path).get_fdata()

    # Get min and max value of reference image
    min_value = reference_image.min()
    max_value = reference_image.max()

    image_rescaled = (image - min_value) / (max_value - min_value)

    return image_rescaled


# Z Score standardization algorithm
def z_score(image):
    # Get reference image to standardise original image
    path = os.path.abspath("./images/1/FLAIR.nii.gz")
    reference_image = nib.load(path).get_fdata()

    mean_value = reference_image[reference_image > 10].mean()
    standard_deviation_value = reference_image[reference_image > 10].std()

    if np.std(image) == 0:
        image_rescaled = image
    else:
        # image_standardized = (image - np.mean(image)) / np.std(image)
        image_rescaled = (image - mean_value) / (standard_deviation_value)

    return image_rescaled


# White stripe standardization algorithm
def white_stripe(image):
    # Get reference image to standardise original image
    path = os.path.abspath("./images/1/FLAIR.nii.gz")
    reference_image = nib.load(path).get_fdata()

    # Create histogram
    hist, bin_edges = np.histogram(reference_image.flatten(), bins=100)

    # Find all the histogram peaks
    peaks, _ = find_peaks(hist, height=100)
    peaks_values = bin_edges[peaks]

    # Rescaled image with the second peak (White matter)
    image_rescaled = image / peaks_values[1]

    return image_rescaled

# def histogram_matching(image_data):
#     ## Load the original image data
#     data_orig = image_data
#     # Load the target image data
#     path = os.path.abspath("./images/1/FLAIR.nii.gz")
#     data_target = nib.load(path).get_fdata()

#     # Flatten the data arrays into 1D arrays
#     flat_orig = data_orig.flatten()
#     flat_target = data_target.flatten()

#     # Calculate the cumulative histograms for the original and target images
#     hist_orig, bins = np.histogram(flat_orig, bins=256, range=(0, 255), density=True)
#     hist_orig_cumulative = hist_orig.cumsum()
#     hist_target, _ = np.histogram(flat_target, bins=256, range=(0, 255), density=True)
#     hist_target_cumulative = hist_target.cumsum()

#     # Map the values of the original image to the values of the target image
#     lut = np.interp(hist_orig_cumulative, hist_target_cumulative, bins[:-1])

#     # Apply the mapping to the original image data
#     data_matched = np.interp(data_orig, bins[:-1], lut)

#     return data_matched


def histogram_matching(transform_data,k=3):
    path = os.path.abspath("./images/1/FLAIR.nii.gz")
    reference_data = nib.load(path).get_fdata()

    # Reshape the data arrays to 1D arrays
    reference_flat = reference_data.flatten()
    transform_flat = transform_data.flatten()


    reference_landmarks = np.percentile(reference_flat, np.linspace(0, 100, k))
    transform_landmarks = np.percentile(transform_flat, np.linspace(0, 100, k))

    piecewise_func = np.interp(transform_flat, transform_landmarks, reference_landmarks)


    transformed_data = piecewise_func.reshape(transform_data.shape)

    return transformed_data
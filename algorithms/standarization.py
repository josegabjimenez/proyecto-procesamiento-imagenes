from scipy.signal import find_peaks
import numpy as np


# Rescaling standarization algorithm
def rescaling(image):
    min_value = image.min()
    max_value = image.max()

    image_rescaled = (image - min_value) / (max_value - min_value)

    return image_rescaled


# Z Score standardization algorithm
def z_score(image):
    mean_value = image[image > 10].mean()
    standard_deviation_value = image[image > 10].std()

    image_rescaled = (image - mean_value) / (standard_deviation_value)

    return image_rescaled


# White stripe standardization algorithm
def white_stripe(image):
    # Create histogram
    hist, bin_edges = np.histogram(image.flatten(), bins=100)

    # Find all the histogram peaks
    peaks, _ = find_peaks(hist, height=100)
    peaks_values = bin_edges[peaks]

    # Rescaled image with the second peak (White matter)
    image_rescaled = image / peaks_values[1]

    return image_rescaled

    # # Mostrar el histograma con los picos identificados
    # plt.axvline(peaks_values[1], color="r", linestyle="--")
    # plt.hist(image.flatten(), bins=100)
    # plt.plot(bin_edges[picos], hist[picos], "x")
    # plt.show()

# Rescaling standarization algorithm
def rescaling(image):
    min_value = image.min()
    max_value = image.max()

    image_rescaled = (image - min_value) / (max_value - min_value)

    return image_rescaled


def z_score(image):
    mean_value = image[image > 10].mean()
    standard_deviation_value = image[image > 10].std()

    image_rescaled = (image - mean_value) / (standard_deviation_value)

    return image_rescaled

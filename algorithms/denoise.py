import numpy as np


def mean_filter(image):
    filtered_image = np.zeros_like(image)
    for x in range(1, image.shape[0] - 2):
        for y in range(1, image.shape[1] - 2):
            for z in range(1, image.shape[2] - 2):
                voxel_of_interest = image[x, y, z]

                avg = 0
                for dx in range(-1, 1):
                    for dy in range(-1, 1):
                        for dz in range(-1, 1):
                            avg = avg + image[x + dx, y + dy, z + dz]

                filtered_image[x + 1, y + 1, z + 1] = avg / 27
    return filtered_image


def median_filter(image):
    # Median Filter
    filtered_image = np.zeros_like(image)
    for x in range(1, image.shape[0] - 2):
        for y in range(1, image.shape[1] - 2):
            for z in range(1, image.shape[2] - 2):
                neightbours = []
                for dx in range(-1, 1):
                    for dy in range(-1, 1):
                        for dz in range(-1, 1):
                            neightbours.append(image[x + dx, y + dy, z + dz])

                median = np.median(neightbours)
                filtered_image[x + 1, y + 1, z + 1] = median
    return filtered_image

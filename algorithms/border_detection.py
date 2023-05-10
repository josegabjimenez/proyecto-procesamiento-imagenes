import numpy as np


def finite_differences(image):
    dfdx = np.zeros_like(image)
    dfdy = np.zeros_like(image)
    dfdz = np.zeros_like(image)
    for x in range(1, image.shape[0] - 2):
        for y in range(1, image.shape[1] - 2):
            for z in range(1, image.shape[2] - 2):
                dfdx[x, y, z] = image[x + 1, y, z] - image[x - 1, y, z]
                dfdy[x, y, z] = image[x, y + 1, z] - image[x, y - 1, z]
                dfdz[x, y, z] = image[x, y, z + 1] - image[x, y, z - 1]

    magnitude = np.sqrt(np.power(dfdx, 2) + np.power(dfdy, 2) + np.power(dfdz, 2))

    return magnitude

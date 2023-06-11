import numpy as np


# Mean filter denoise
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


# Median filter denoise
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


# Median Filter with noise detection using finite differences (Edge filter)
def edge_filter(image, tol=0.1, tau=0.5):
    filtered_image = np.zeros_like(image)

    for x in range(1, image.shape[0] - 2):
        for y in range(1, image.shape[1] - 2):
            for z in range(1, image.shape[2] - 2):
                # Compute the derivatives in x, y, and z directions
                dx = image[x + 1, y, z] - image[x - 1, y, z]
                dy = image[x, y + 1, z] - image[x, y - 1, z]
                dz = image[x, y, z + 1] - image[x, y, z - 1]

                # Compute the magnitude of the gradient
                magnitude = np.sqrt(dx * dx + dy * dy + dz * dz)

                # Update the threshold using ISODATA algorithm
                segmentation = image >= tau

                # Background
                mBG = image[segmentation == False]
                if len(mBG) > 0:
                    mBG = np.nan_to_num(mBG, nan=0)
                    mBG = mBG.mean()
                else:
                    mBG = 0

                # Foreground
                mFG = image[segmentation]
                if len(mFG) > 0:
                    mFG = np.nan_to_num(mFG, nan=0)
                    mFG = mFG.mean()
                else:
                    mFG = 0

                # Update tau
                tau_post = 0.5 * (mBG + mFG)

                # Check if accepts the tolerance, if not, continue iterating
                if np.abs(tau - tau_post) < tol:
                    tau = tau
                else:
                    tau = tau_post

                # print(tol)

                # Compute the threshold using a fraction of the standard deviation
                # threshold = 3 * std

                # If the magnitude is below the threshold, apply median filter
                if magnitude < tau:
                    neighbours = []
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                neighbours.append(image[x + dx, y + dy, z + dz])
                    median = np.median(neighbours)
                    filtered_image[x, y, z] = median
                else:
                    filtered_image[x, y, z] = image[x, y, z]
    return filtered_image

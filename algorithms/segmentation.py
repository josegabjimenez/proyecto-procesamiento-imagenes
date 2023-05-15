import numpy as np

# Thresholding - Umbralización
def thresholding(image, tol = 1, tau = 20):

    while True:
        segmentation = image >= tau
        # Background
        mBG = image[np.multiply(image > 0.01, segmentation == 0)].mean()
        # Foreground
        mFG = image[np.multiply(image > 0.01, segmentation == 1)].mean()

        # Update tau
        tau_post = 0.5 * (mBG + mFG)

        # Check if accepts the tolerance, if not, continue iterating
        if np.abs(tau - tau_post) < tol:
            break
        else:
            tau = tau_post

    return segmentation

# Clustering - Kmeans
def clustering(image, k=3):
    # Initialize the centroids
    centroids = np.linspace(np.amin(image), np.amax(image), num=k)
    
    for i in range(3):
        # Compute the distances from each point to each centroid
        distances = np.abs(image[..., np.newaxis] - centroids)
        
        # Assign each point to the closest centroid
        segmentation = np.argmin(distances, axis=-1)
        
        # Update the centroids
        for group in range(k):
            centroids[group] = image[segmentation == group].mean()
    
    return segmentation



# Region Growing
def region_growing(image, origin_x = 100, origin_y = 100, origin_z = 20):
    # origin_x = 100
    # origin_y = 100
    # origin_z = 1
    x = 1
    y = 1
    z = 1
    valor_medio_cluster = image[origin_x, origin_y, origin_z]
    tol = 50
    segmentation = np.zeros_like(image)
    point = [origin_x,origin_y]

    tail = [point]
    evaluated = image == True

    while True:
        punto = tail.pop(0)

        print(len(tail))
        
        for dx in [-x, 0, x] :
            for dy in [-y, 0, y] :
                nuevoPunto = [punto[0]+dx, punto[1]+dy]
                if((nuevoPunto[0] < 230) and ((nuevoPunto[0]) > 0) and (nuevoPunto[1] < 230) and ((nuevoPunto[1]) > 0) ):
                    if (not evaluated[nuevoPunto[0], nuevoPunto[1],origin_z]):
                        if np.abs(valor_medio_cluster - image[nuevoPunto[0], nuevoPunto[1], origin_z]) < tol :
                            segmentation[nuevoPunto[0], nuevoPunto[1], origin_z] = 1
                            tail.append([nuevoPunto[0], nuevoPunto[1]])
                            evaluated[nuevoPunto[0], nuevoPunto[1], origin_z] = True
                            evaluated[punto[0], punto[1], origin_z] = True
                        else :
                            segmentation[nuevoPunto[0], nuevoPunto[1], origin_z] = 0
                            tail.append([nuevoPunto[0], nuevoPunto[1]])
                            evaluated[nuevoPunto[0], nuevoPunto[1], origin_z] = True
                            evaluated[punto[0], punto[1], origin_z] = True


        valor_medio_cluster = image[segmentation == 1].mean()

        if len(tail) == 0:
            break
    return segmentation



def gaussian(x, mu, sigma):
    """
    Computes the probability density function of a Gaussian distribution.

    :param x: Input data.
    :param mu: Mean of the Gaussian distribution.
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: Probability density function values for the input data.
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def gmm(image_data, k, num_iterations=100, threshold=0.01):
    """
    Segments an image into multiple classes using a Gaussian Mixture Model.

    :param image_data: Input image data.
    :param k: Number of classes to segment the image into.
    :param num_iterations: Maximum number of iterations to run the algorithm for (default: 100).
    :param threshold: Convergence threshold for the algorithm (default: 0.01).
    :return: Segmented image data.
    """
    # Initialize parameters
    num_voxels = np.prod(image_data.shape)
    mu = np.linspace(image_data.min(), image_data.max(), k)
    sigma = np.ones(k) * (image_data.max() - image_data.min()) / (2 * k)
    p = np.ones(k) / k
    q = np.zeros((num_voxels, k))

    # Run the algorithm
    for i in range(num_iterations):
        # Calculate responsibilities
        for k in range(k):
            q[:, k] = p[k] * gaussian(image_data.flatten(), mu[k], sigma[k])
        q = q / np.sum(q, axis=1)[:, np.newaxis]

        # Update parameters
        n = np.sum(q, axis=0)
        p = n / num_voxels
        mu = np.sum(q * image_data.flatten()[:, np.newaxis], axis=0) / n
        sigma = np.sqrt(np.sum(q * (image_data.flatten()[:, np.newaxis] - mu) ** 2, axis=0) / n)

        # Check for convergence
        if np.max(np.abs(p - q.sum(axis=0) / num_voxels)) < threshold:
            break

    # Generate segmentation
    segmentation = np.argmax(q, axis=1)
    segmentation = segmentation.reshape(image_data.shape)

    return segmentation
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

def k_means(image, ks,iteracion):
        
    # Inicialización de valores k
    k_values = np.linspace(np.amin(image), np.amax(image), ks)
    iteracion=10
    for i in range(iteracion):
        d_values = [np.abs(k - image) for k in k_values]
        segmentationr = np.argmin(d_values, axis=0)

        for k_idx in range(ks):
            k_values[k_idx] = np.mean(image[segmentationr == k_idx])

    return segmentationr

def clustering(image, k=3, iterations=3):
    # Initialize the centroids
    centroids = np.linspace(np.amin(image), np.amax(image), num=k)
    
    for i in range(iterations):
        # Compute the distances from each point to each centroid
        distances = np.abs(image[..., np.newaxis] - centroids)
        
        # Assign each point to the closest centroid
        segmentation = np.argmin(distances, axis=-1)
        
        # Update the centroids
        for group in range(k):
            centroids[group] = image[segmentation == group].mean()
    
    return segmentation



# Region Growing
def region_growing(image, x=100, y=100, z=20, tol=50):
    segmentation = np.zeros_like(image)
    if segmentation[x,y,z] == 1:
        return
    valor_medio_cluster = image[x,y,z]
    segmentation[x,y,z] = 1
    vecinos = [(x, y, z)]
    while vecinos:
        x, y, z = vecinos.pop()
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    #vecino
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if nx >= 0 and nx < image.shape[0] and \
                        ny >= 0 and ny < image.shape[1] and \
                        nz >= 0 and nz < image.shape[2]:
                        if np.abs(valor_medio_cluster - image[nx,ny,nz]) < tol and \
                            segmentation[nx,ny,nz] == 0:
                            segmentation[nx,ny,nz] = 1
                            vecinos.append((nx, ny, nz))
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


def GMM(image):
    # Each component has a weight (wi), a mean (mui), and a standard deviation (sdi)
    w1 = 1/3
    w2 = 1/3
    w3 = 1/3
    mu1 = 0
    sd1 = 50
    mu2 = 100
    sd2 = 50
    mu3 = 150
    sd3 = 50

    segmentation = np.zeros_like(image)
    for iter in range(1, 5) :

        # Compute likelihood of belonging to a cluster
        p1 = 1/np.sqrt(2*np.pi*sd1**2) * np.exp(-0.5*np.power(image - mu1, 2) / sd1**2)
        p2 = 1/np.sqrt(2*np.pi*sd2**2) * np.exp(-0.5*np.power(image - mu2, 2) / sd2**2)
        p3 = 1/np.sqrt(2*np.pi*sd3**2) * np.exp(-0.5*np.power(image - mu3, 2) / sd3**2)

        # Normalise probability
        r1 = np.divide(w1 * p1, w1 * p1 + w2 * p2 + w3 * p3)
        r2 = np.divide(w2 * p2, w1 * p1 + w2 * p2 + w3 * p3) 
        r3 = np.divide(w3 * p3, w1 * p1 + w2 * p2 + w3 * p3) 

        # Update parameters
        w1 = r1.mean()
        w2 = r2.mean()
        w3 = r3.mean()
        mu1 = np.multiply(r1, image).sum() / r1.sum()
        sd1 = np.sqrt(np.multiply(r1, np.power(image - mu1, 2)).sum() / r1.sum())
        mu2 = np.multiply(r2, image).sum() / r2.sum()
        sd2 = np.sqrt(np.multiply(r2, np.power(image - mu2, 2)).sum() / r2.sum())
        mu3 = np.multiply(r3, image).sum() / r3.sum()
        sd3 = np.sqrt(np.multiply(r3, np.power(image - mu3, 2)).sum() / r3.sum())

    # Perform segmentation
    segmentation[np.multiply(r1 > r2, r1 > r3)] = 0
    segmentation[np.multiply(r2 > r1, r2 > r3)] = 1
    segmentation[np.multiply(r3 > r1, r3 > r2)] = 2

    return segmentation
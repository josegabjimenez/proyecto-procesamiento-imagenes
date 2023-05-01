import numpy as np

# Thresholding - Umbralización
def thresholding(image, tol = 1, tau = 20):

    while True:
        segmentation = image >= tau
        # Background
        mBG = image[np.multiply(image > 10, segmentation == 0)].mean()
        # Foreground
        mFG = image[np.multiply(image > 10, segmentation == 1)].mean()

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
def region_growing(image):
    origin_x = 100
    origin_y = 100
    origin_z = 1
    x = 1
    y = 1
    z = 1
    valor_medio_cluster = image[origin_x, origin_y, 20]
    tol = 50
    segmentation = np.zeros_like(image)
    itera = 1
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
                    if (not evaluated[nuevoPunto[0], nuevoPunto[1],20]):
                        if np.abs(valor_medio_cluster - image[nuevoPunto[0], nuevoPunto[1], 20]) < tol :
                            segmentation[nuevoPunto[0], nuevoPunto[1], 20] = 1
                            tail.append([nuevoPunto[0], nuevoPunto[1]])
                            evaluated[nuevoPunto[0], nuevoPunto[1], 20] = True
                            evaluated[punto[0], punto[1], 20] = True
                        else :
                            segmentation[nuevoPunto[0], nuevoPunto[1], 20] = 0
                            tail.append([nuevoPunto[0], nuevoPunto[1]])
                            evaluated[nuevoPunto[0], nuevoPunto[1], 20] = True
                            evaluated[punto[0], punto[1], 20] = True


        valor_medio_cluster = image[segmentation == 1].mean()

        if len(tail) == 0:
            break
    return segmentation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


#--------------------------------------------------- 1 K-Means Clustering-----------------------------------------------------
def kmeans(data, K, thresh, n_iter, n_attempts):
    """
    Cluster data in K clusters using the K-Means algorithm
    :param data: numpy.array(float), the input data array with N (#data) x D (#feature_dimensions) dimensions
    :param K: int, number of clusters
    :param thresh: float, convergence threshold
    :param n_iter: int, #iterations of the K-Means algorithm
    :param n_attempts: int, #attempts to run the K-Means algorithm
    :return:
    compactness: float, the sum of squared distance from each point to their corresponding centers
    labels: numpy.array(int), the label array with Nx1 dimensions, where it denotes the corresponding cluster of
    each data point
    centers : numpy.array(float), a KxD array with the final centroids
    """

    assert data.ndim == 2 # Data should be a 2D array.
    assert K > 0 and thresh > 0 and n_iter > 0 and n_attempts > 0 # K, thresh, n_iter, and n_attempts must be positive.
    assert K <= data.shape[0] # K should be less than or equal to the number of data points.

    best_compactness = float('inf')
    best_labels = None
    best_centers = None

    for attempt in range(n_attempts):
        # Step 1: Randomly initialize K centroids
        initial_centroids = data[np.random.choice(data.shape[0], K, replace=False)]
        centers = initial_centroids.copy()

        for iteration in range(n_iter):
            # Step 2: Assign labels based on closest center
            distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Step 3: Calculate new centroids
            new_centers = np.array(
                [data[labels == k].mean(axis=0) if len(data[labels == k]) > 0 else centers[k] for k in range(K)])

            # Step 4: Check for convergence
            if np.linalg.norm(new_centers - centers) < thresh:
                break
            centers = new_centers

        # Step 5: Calculate compactness
        compactness = sum(np.min(distances ** 2, axis=1))

        # Keep the best attempt based on compactness
        if compactness < best_compactness:
            best_compactness = compactness
            best_labels = labels
            best_centers = centers

    compactness = best_compactness
    labels = best_labels
    centers = best_centers

    return compactness, labels, centers


def plot_clustered_image(image_shape, labels, centers):

    centers = np.uint8(centers)
    result_image = centers[labels.flatten()]
    result_image = result_image.reshape(image_shape)
    plt.imshow(result_image)
    plt.axis('off')


# Load and plot image
image = cv2.imread("data/home.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Original Image")
plt.show(block=False)
plt.pause(0.1)

# Prepare image data
rgb_vectors = np.reshape(image_rgb, (-1, 3)).astype(np.float32)

height, width, _ = image_rgb.shape
# Create i (row indices) and j (column indices) grids
i_indices, j_indices = np.indices((height, width))
# Flatten the i, j coordinates and RGB values, and stack them
i_flat = i_indices.flatten()        # Flattened row indices
j_flat = j_indices.flatten()        # Flattened column indices
# Stack i, j, r, g, b into a single array
pixel_coordinates_rgb = np.column_stack((i_flat, j_flat, rgb_vectors)).astype(np.float32)


# Define parameters
K, thresh, n_iter, n_attempts = 4, 1.0, 10, 10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n_iter, thresh)

# Run custom K-Means on (r, g, b) features
compactness_rgb, labels_rgb, centers_rgb = kmeans(rgb_vectors, K, thresh, n_iter, n_attempts)
# Run OpenCV K-Means on (r, g, b) features
compactness_opencv_rgb, labels_opencv_rgb, centers_opencv_rgb = cv2.kmeans(rgb_vectors, K, None, criteria, n_attempts, cv2.KMEANS_RANDOM_CENTERS)[0:3]

# Run custom K-Means on (i, j, r, g, b) features
compactness_xy_rgb, labels_xy_rgb, centers_xy_rgb = kmeans(pixel_coordinates_rgb, K, thresh, n_iter, n_attempts)
# Run OpenCV K-Means on (i, j, r, g, b) features
compactness_opencv_xy_rgb, labels_opencv_xy_rgb, centers_opencv_xy_rgb = cv2.kmeans(pixel_coordinates_rgb, K, None, criteria, n_attempts, cv2.KMEANS_RANDOM_CENTERS)[0:3]


# Plot results
plt.figure(figsize=(10, 5))
# Plot my K-Means on (r, g, b)
plt.subplot(1, 2, 1)
plt.title("My K-Means (r, g, b)")
plot_clustered_image(image.shape, labels_rgb, centers_rgb)
# Plot OpenCV K-Means on (r, g, b)
plt.subplot(1, 2, 2)
plt.title("OpenCV K-Means (r, g, b)")
plot_clustered_image(image.shape, labels_opencv_rgb, centers_opencv_rgb)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

plt.figure(figsize=(10, 5))
# Plot my K-Means on (i, j, r, g, b)
plt.subplot(1, 2, 1)
plt.title("My K-Means (i, j, r, g, b)")
plot_clustered_image(image.shape, labels_xy_rgb, centers_xy_rgb[:, 2:])# Use only RGB columns
# Plot OpenCV K-Means on (i, j, r, g, b)
plt.subplot(1, 2, 2)
plt.title("OpenCV K-Means (i, j, r, g, b)")
plot_clustered_image(image.shape, labels_opencv_xy_rgb, centers_opencv_xy_rgb[:, 2:])# Use only RGB columns
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)
#-----------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------2 Efficient Graph-Based Image Segmentation-----------------------------------------------------
# Load and plot image
image = cv2.imread("data/eiffel_tower.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")
plt.show(block=False)
plt.pause(0.1)

# Apply Gaussian filter
blurred_image = cv2.GaussianBlur(image, (3, 3), 0.8)

def nn_graph(input_image, k):
    """
    Create a graph based on the k-nearest neighbors of each pixel in the (i,j,r,g,b) feature space.
    Edge weights are calculated as the Euclidean distance of the node's features
    and its corresponding neighbors.
    :param input_image: numpy.array(uint8), input image of HxWx3 dimensions
    :param k: int, nearest neighbors for each node
    :return:
    graph: tuple(V: numpy.array(int), E: <graph connectivity representation>), the NN-graph where
    V is the set of pixel-nodes of (W*H)x2 dimensions and E is a representation of the graph's
    undirected edges along with their corresponding weight
    """

    assert len(input_image.shape) == 3 # Input image must have 3 dimensions (HxWx3)
    assert isinstance(k, int) and k > 0 # k must be a non-zero positive integer

    H, W, C = input_image.shape

    # Create the set of pixel-nodes V
    V = np.array([[i, j] for i in range(H) for j in range(W)], dtype=int)

    # Convert the input image into a (W*H, 5) feature matrix
    features = np.zeros((H * W, 5), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            r, g, b = input_image[i, j]
            features[i * W + j] = [i, j, r, g, b]



    nbrs = NearestNeighbors().fit(features)
    adj_matrix_1 = nbrs.kneighbors_graph(features, n_neighbors=k + 1, mode='distance')

    # Get rows and cols of edges - non zero elements
    rows, cols = adj_matrix_1.nonzero()

    indices = np.where(rows <= cols)
    adj_matrix_1[rows[indices], cols[indices]] = 0

    E = adj_matrix_1

    E.setdiag(0)  # Remove diagonal (self-loops)
    E.eliminate_zeros()  # Clean up any explicit zeros

    print(E)


    return V, E




def segmentation(G, k, min_size):
    """
    Segment the image based on the Efficient Graph-Based Image Segmentation algorithm.
    :param G: tuple(V, E), the input graph
    :param k: int, sets the threshold k/|C|
    :param min_size: int, minimum size of clusters
    :return:
    clusters: numpy.array(int), a |V|x1 array where it denotes the cluster for each node v of the graph
    """

    assert isinstance(k, int) and k > 0  # k must be a non-zero positive integer
    assert isinstance(min_size, int) and min_size > 0  # min_size must be a non-zero positive integer

    V, E = G
    num_nodes = V.shape[0]


    # Step 1: Sort edges by ascending weight
    rows, cols = E.nonzero()  # Get the row and column indices of non-zero elements
    weights = E.data  # Get the corresponding weights of the edges
    edges = list(zip(rows, cols, weights))  # (vi, vj, weight)
    edge_list = sorted(edges, key=lambda x: x[2])  # Sort by weight


    # Step 2: Initialize each vertex in its own segment
    segmentation1 = np.arange(num_nodes)  # Initially, each node is its own cluster
    cluster_internal_diff = np.zeros(num_nodes)  # Store internal differences for clusters
    cluster_size = np.ones(num_nodes)  # Store sizes of clusters


    def find_segment(segmentation, node):
        # Find the root/representative of a segment
        if segmentation[node] != node:
            segmentation[node] = find_segment(segmentation, segmentation[node])
        return segmentation[node]

    def merge_segments(ci, cj, weight, segmentation1, cluster_internal_diff, cluster_size):
        # Merge clusters ci and cj

        root1 = find_segment(segmentation1, ci)
        root2 = find_segment(segmentation1, cj)

        if root1 != root2:
        # Merge the smaller cluster into the larger cluster
            if cluster_size[root1] < cluster_size[root2]:
                root1, root2 = root2, root1  # Ensure root1 is the larger root
            segmentation1[root2] = root1  # Make root1 the root of root2
            cluster_size[root1] += cluster_size[root2]  # Update cluster size
            cluster_size[root2] = 0

        cluster_internal_diff[root1] = max(cluster_internal_diff[root1], cluster_internal_diff[root2], weight)  # Update internal difference for the new root


    def compute_MInt(ci, cj, cluster_internal_diff, cluster_size, k):
        # Compute the minimum internal difference
        tau_ci = k / cluster_size[ci]
        tau_cj = k / cluster_size[cj]
        min_int = min(cluster_internal_diff[ci] + tau_ci, cluster_internal_diff[cj] + tau_cj)
        return min_int

    # Step 3: Process edges


    for i, (vi, vj, weight) in enumerate(edge_list):

        ci = find_segment(segmentation1, vi)
        cj = find_segment(segmentation1, vj)
        if ci != cj:
            if weight <= compute_MInt(ci, cj, cluster_internal_diff, cluster_size, k):
                merge_segments(ci, cj, weight, segmentation1, cluster_internal_diff, cluster_size)


    # Step 4: Post-processing step
    for vi, vj, weight in edge_list:
        ci = find_segment(segmentation1, vi)
        cj = find_segment(segmentation1, vj)
        if ci != cj:
            if cluster_size[ci] < min_size or cluster_size[cj] < min_size:
                merge_segments(ci, cj, weight, segmentation1, cluster_internal_diff, cluster_size)




    # Final segmentation: Reassign unique cluster IDs
    unique_segments = np.unique([find_segment(segmentation1, v) for v in range(num_nodes)])
    cluster_map = {seg_id: idx for idx, seg_id in enumerate(unique_segments)}  # Map old IDs to new IDs
    final_clusters = np.array([cluster_map[find_segment(segmentation1, v)] for v in range(num_nodes)])

    clusters = final_clusters
    return clusters




def visualize_segments(input_image, clusters):

    H, W, _ = input_image.shape
    num_clusters = clusters.max() + 1  # Number of unique clusters

    # Generate distinct colors for each cluster using HSV and convert to RGB
    hsv_colors = np.linspace(0, 1, num_clusters, endpoint=False)  # Generate evenly spaced hues
    rgb_colors = plt.cm.hsv(hsv_colors)[:, :3]  # Convert HSV to RGB (drop alpha channel)
    rgb_colors = (rgb_colors * 255).astype(np.uint8)  # Normalize to [0, 255]

    # Create a segmented image by mapping clusters to their corresponding RGB color
    segmented_image = np.zeros((H, W, 3), dtype=np.uint8)  # Create a blank RGB image

    # Map each pixel to its cluster's color
    for i in range(H):
        for j in range(W):
            cluster_id = clusters[i * W + j]  # Get the cluster ID of the current pixel
            segmented_image[i, j] = rgb_colors[cluster_id]  # Assign the corresponding RGB color

    # Plot the segmented image
    plt.figure()
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.title("Segmented Image")
    plt.show()


# Run NN-Graph
V, E = nn_graph(blurred_image, 10)

# Run segmentation
k, min_size = 550, 300
final_clusters = segmentation((V, E), k, min_size)

# Visualize the results
visualize_segments(image, final_clusters)

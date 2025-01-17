import cv2
import numpy as np
import matplotlib.pyplot as plt
from libs.helper import drawlines
from libs.helper import plot_3d_points


#--------------------------------------------- 1 Fundamental Matrix Linear System ----------------------------------------------
def fundamental_matrix_linear_system(pts1, pts2):
    """
        Create linear equations for estimating the fundamental matrix in matrix form
    :param pts1: numpy.array(float), an array Nx2 that holds the source image points
    :param pts2: numpy.array(float), an array Nx2 that holds the destination image points
    :return:
        A: numpy.array(float), an array Nx8 that holds the left side coefficients of the linear equations
        b: numpy.array(float), an array Nx1 that holds the right side coefficients of the linear equations
    """
    # Ensure inputs are valid
    assert isinstance(pts1, np.ndarray) and isinstance(pts2, np.ndarray) # Inputs must be numpy arrays
    assert pts1.shape == pts2.shape # Input arrays must have the same shape
    assert pts1.shape[1] == 2 # Input arrays must have two columns (x, y coordinates)
    assert pts1.shape[0] >= 8 # There must be at least 8 point correspondences

    N = pts1.shape[0]  # Number of correspondences

    # Extract x, y coordinates
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    # Construct matrix A
    A = np.zeros((N, 8))
    A[:, 0] = x1 * x2
    A[:, 1] = y1 * x2
    A[:, 2] = x2
    A[:, 3] = x1 * y2
    A[:, 4] = y1 * y2
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1

    # Construct vector b
    b = -np.ones((N, 1))  # Since f33 is set to 1, move it to the right side

    return A, b
#----------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------- 2 ComputeEpipolar Lines ----------------------------------------------
def compute_correspond_epilines(points, which_image, F):
    """
        For points in an image of a stereo pair, computes the corresponding epilines in the other image.
    :param points: numpy.array(float), an array Nx2 that holds the image points
    :param which_image: int, index of the image (1 or 2) that contains the points
    :param F: numpy.array(float), fundamental matrix between the stereo pair
    :return:
        epilines: numpy.array(float): an array Nx3 that holds the coefficients of the corresponding epipolar lines
    """
    # Input validation
    assert isinstance(points, np.ndarray) # Points must be a numpy array
    assert points.ndim == 2 and points.shape[1] == 2 # Points must be an Nx2 array
    assert which_image in {1, 2} # which_image must be either 1 or 2
    assert isinstance(F, np.ndarray) and F.shape == (3, 3) # F must be a 3x3 fundamental matrix

    # Add a homogeneous coordinate (z=1) to the points
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    # Compute epilines based on which image the points belong to
    if which_image == 1:
        epilines = np.dot(F, points_h.T).T  # Compute lines in image 2 for points in image 1
    else:
        epilines = np.dot(F.T, points_h.T).T  # Compute lines in image 1 for points in image 2

    # Normalize the epilines so that a^2 + b^2 = 1
    norms = np.sqrt(epilines[:, 0] ** 2 + epilines[:, 1] ** 2).reshape(-1, 1)
    epilines = epilines / norms

    return epilines
#----------------------------------------------------------------------------------------------------------------------------



#---------------------------------------------  4 Normalize Points ----------------------------------------------
def points_normalization(pts1, pts2):
    """
    Normalize points so that each coordinate system is located at the centroid of the image points and
    the mean square distance of the transformed image points from the origin should be 2 pixels.
    :param pts1: numpy.array(float), an Nx2 array that holds the source image points
    :param pts2: numpy.array(float), an Nx2 array that holds the destination image points
    :return:
    pts1_normalized: numpy.array(float), an Nx2 array with the transformed source image points
    pts2_normalized: numpy.array(float), an Nx2 array with the transformed destination image points
    M1: numpy.array(float), an 3x3 array- transformation for source image
    M2: numpy.array(float), an 3x3 array- transformation for destination image
    """
    assert pts1.ndim == 2 and pts1.shape[1] == 2 # pts1 must be Nx2
    assert pts2.ndim == 2 and pts2.shape[1] == 2 # pts2 must be Nx2
    assert pts1.shape == pts2.shape # pts1 and pts2 must have the same dimensions

    def normalize_points(pts):
        N = pts.shape[0]

        # Step 1: Compute the centroid
        centroid = np.mean(pts, axis=0)

        # Step 2: Translate points to have the centroid at the origin
        translated_pts = pts - centroid

        # Step 3: Compute the mean square distance from the origin
        mean_sq_dist = np.mean(np.sum(translated_pts**2, axis=1))

        # Step 4: Compute the scaling factor
        scale = np.sqrt(2 / mean_sq_dist)

        # Step 5: Create the transformation matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])

        # Step 6: Normalize the points
        # Convert to homogeneous coordinates, apply transformation, and return to Cartesian coordinates
        homogeneous_pts = np.hstack((pts, np.ones((N, 1))))  # Nx3
        normalized_homogeneous_pts = (T @ homogeneous_pts.T).T  # Apply transformation
        normalized_pts = normalized_homogeneous_pts[:, :2] / normalized_homogeneous_pts[:, 2:]

        return normalized_pts, T

    # Normalize both sets of points
    pts1_normalized, M1 = normalize_points(pts1)
    pts2_normalized, M2 = normalize_points(pts2)

    return pts1_normalized, pts2_normalized, M1, M2
#----------------------------------------------------------------------------------------------------------------------------








#---------------------------------------------  3 Estimate Fundamental Matrix ----------------------------------------------
# Load images
image1 = cv2.imread(f'data/image1.png', cv2.IMREAD_COLOR)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

image2 = cv2.imread(f'data/image2.png', cv2.IMREAD_COLOR)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


gray_image_1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
gray_image_2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

images = []
images.append(image1)
images.append(image2)

gray_images = []
gray_images.append(gray_image_1)
gray_images.append(gray_image_2)

sift = cv2.SIFT_create()
keypoints, descriptors = [], []
for img in gray_images:
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)



#Nearest neighbor matching with ratio test
def match_features(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])
    return good_matches



# Display input images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
fig.suptitle("Input Images", fontsize=16)
ax = axes[0]  # First col
ax.imshow(image1)
ax.set_title("Image 1")
ax.axis('off')
ax = axes[1]  # Second col
ax.imshow(image2)
ax.set_title("Image 2")
ax.axis('off')
plt.show()



plt.figure(figsize=(10, 5))
plt.suptitle("Correspondences After Ratio Test")
# Find and plot good matches for images
good_matches_all = []
for i in range(len(images) - 1):
    good_matches = match_features(descriptors[i], descriptors[i + 1])
    good_matches_all.append(good_matches)

    img_matches = np.empty((max(images[i].shape[0], images[i+1].shape[0]), images[i].shape[1] + images[i+1].shape[1], 3),dtype=np.uint8)
    cv2.drawMatchesKnn(images[i], keypoints[i], images[i+1], keypoints[i+1], outImg=img_matches, matches1to2=good_matches, flags=2)

    # Display the img_matches
    plt.imshow(img_matches)
    plt.axis('off')

plt.show()


def ransac(src_points, dst_points, ransac_reproj_threshold=0.5, max_iters=5000, inlier_ratio=0.9, normalize = False):
    """
     Calculate the set of inlier correspondences w.r.t. fundamental matrix, using the RANSAC method.
     :param src_points: numpy.array(float), an Nx2 array that holds the coordinates of the points in the
     source image
     :param dst_points: numpy.array(float), an Nx2 array that holds the coordinates of the points in the
     destination image
     :param ransac_reproj_threshold: float, maximum allowed reprojection error to treat a point-epiline pair
     as an inlier
     :param max_iters: int, the maximum number of RANSAC iterations
     :param inlier_ratio: float, ratio of inliers w.r.t. total number of correspondences
     :return:
     F: numpy.array(float), the estimated fundamental matrix using linear least-squares
     mask: numpy.array(uint8), mask that denotes the inlier correspondences
     """
    # Input validation
    assert src_points.shape == dst_points.shape # Source and destination points must have the same dimensions
    assert src_points.ndim == 2 and src_points.shape[1] == 2 # Points must be Nx2 arrays
    assert ransac_reproj_threshold >= 0 # Threshold must be non-negative
    assert isinstance(max_iters, int) and max_iters > 0 # max_iters must be a positive non-zero integer
    assert 0 <= inlier_ratio <= 1 # inlier_ratio must be in range [0, 1]

    n_points = src_points.shape[0]
    best_inliers = 0
    best_F = None
    best_mask = None

    # Optional normalization
    if normalize:
        src_points, dst_points, M1, M2 = points_normalization(src_points, dst_points)

    for _ in range(max_iters):
        try:
            # Randomly select 8 correspondences to estimate the fundamental matrix
            indices = np.random.choice(n_points, 8, replace=False)
            src_sample = src_points[indices]
            dst_sample = dst_points[indices]

            # Use the fundamental_matrix_linear_system function to construct the linear system
            A, b = fundamental_matrix_linear_system(src_sample, dst_sample)

            # Solve the linear system for F
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                F = np.append(x, 1).reshape(3, 3)  # Rebuild the fundamental matrix
            except np.linalg.LinAlgError as e:
                if "Singular matrix" in str(e):
                    continue
                else:
                    raise e

            if normalize:
                # De-normalize the fundamental matrix
                F = M2.T @ F @ M1

            # Compute epilines for all src_points
            epilines = compute_correspond_epilines(src_points, which_image=1, F=F)

            # Calculate distances from dst_points to their corresponding epilines
            distances = np.abs(epilines[:, 0] * dst_points[:, 0] + epilines[:, 1] * dst_points[:, 1] + epilines[:, 2]) / \
                        np.sqrt(epilines[:, 0] ** 2 + epilines[:, 1] ** 2)

            # Determine inliers
            inlier_mask = distances <= ransac_reproj_threshold
            inlier_count = np.sum(inlier_mask)

            # Update the best model if the current one has more inliers
            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_F = F
                best_mask = inlier_mask

            # Stop early if the inlier ratio is sufficient
            if inlier_count >= inlier_ratio * n_points:
                break
        except Exception as e:
            continue

    # Refine F using all inliers
    if best_mask is not None:
        inlier_src_points = src_points[best_mask]
        inlier_dst_points = dst_points[best_mask]
        A, b = fundamental_matrix_linear_system(inlier_src_points, inlier_dst_points)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        best_F = np.append(x, 1).reshape(3, 3)

        if normalize:
            # De-normalize the refined fundamental matrix
            best_F = M2.T @ best_F @ M1

    F = best_F
    mask = best_mask

    return F, mask



src_pts = np.float32([keypoints[0][m[0].queryIdx].pt for m in good_matches_all[0]])
dst_pts = np.float32([keypoints[1][m[0].trainIdx].pt for m in good_matches_all[0]])

# Use custom RANSAC function
F_custom, mask_custom = ransac(src_pts, dst_pts, ransac_reproj_threshold=0.5, max_iters=5000, inlier_ratio=0.9, normalize=False)

# Use OpenCV's RANSAC implementation
F_cv, mask_cv = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransacReprojThreshold=0.5)


# To store src and dst inliers
src_inliers = []
dst_inliers = []
src_inliers_cv = []
dst_inliers_cv = []

# Draw inliers after RANSAC
fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column
fig.suptitle("Inliers After RANSAC: Custom vs OpenCV", fontsize=16)

for i in range(1):  # One pair of images
    # Custom RANSAC inliers
    img_inliers_custom = np.empty(
        (max(images[i].shape[0], images[i + 1].shape[0]), images[i].shape[1] + images[i + 1].shape[1], 3),
        dtype=np.uint8
    )
    good = np.array(good_matches_all[i])
    inliers = good[np.where(np.squeeze(mask_custom) == 1)[0]]  # Mask for custom RANSAC

    # Extract src_inliers and dst_inliers
    for match in inliers:
        src_inliers.append(keypoints[i][match[0].queryIdx].pt)
        dst_inliers.append(keypoints[i + 1][match[0].trainIdx].pt)
    src_inliers = np.array(src_inliers)
    dst_inliers = np.array(dst_inliers)





    cv2.drawMatchesKnn(images[i], keypoints[i], images[i + 1], keypoints[i + 1], matches1to2=inliers,
                       outImg=img_inliers_custom, flags=2)
    ax = axes[0]  # First row
    ax.imshow(img_inliers_custom)
    ax.set_title(f"Custom RANSAC Inliers (Image {i + 1} to Image {i + 2})")
    ax.axis('off')

    # OpenCV RANSAC inliers
    img_inliers_opencv = np.empty(
        (max(images[i].shape[0], images[i + 1].shape[0]), images[i].shape[1] + images[i + 1].shape[1], 3),
        dtype=np.uint8
    )
    inliers = good[np.where(np.squeeze(mask_cv) == 1)[0]]  # Mask for OpenCV RANSAC

    # Extract src_inliers and dst_inliers
    for match in inliers:
        src_inliers_cv.append(keypoints[i][match[0].queryIdx].pt)
        dst_inliers_cv.append(keypoints[i + 1][match[0].trainIdx].pt)
    src_inliers_cv = np.array(src_inliers_cv)
    dst_inliers_cv = np.array(dst_inliers_cv)




    cv2.drawMatchesKnn(images[i], keypoints[i], images[i + 1], keypoints[i + 1], matches1to2=inliers,
                       outImg=img_inliers_opencv, flags=2)
    ax = axes[1]  # Second row
    ax.imshow(img_inliers_opencv)
    ax.set_title(f"OpenCV RANSAC Inliers (Image {i + 1} to Image {i + 2})")
    ax.axis('off')

plt.show()



# Convert the inlier points to integer coordinates
src_inliers = np.array(src_inliers, dtype=np.int32)
dst_inliers = np.array(dst_inliers, dtype=np.int32)
src_inliers_cv = np.array(src_inliers_cv, dtype=np.int32)
dst_inliers_cv = np.array(dst_inliers_cv, dtype=np.int32)


# Plot epipolar lines and corresponding points
fig, axes = plt.subplots(2, 2, figsize=(15, 15))  # 2×2 grid
fig.suptitle("Epipolar Lines and Corresponding Points", fontsize=16)

# Custom RANSAC - Epipolar Lines on Image 2 with Points from Image 1
lines_img2_custom = compute_correspond_epilines(points=src_inliers, which_image=1, F=F_custom)
img1, img2 = drawlines(images[1], images[0], lines_img2_custom, src_inliers, dst_inliers)
axes[0, 0].set_title("Image 1 corresponding points (Custom)")
axes[0, 0].imshow(img2)
axes[0, 0].axis('off')
axes[0, 1].set_title("Image 2 epilines (Custom)")
axes[0, 1].imshow(img1)
axes[0, 1].axis('off')

# OpenCV RANSAC - Epipolar Lines on Image 2 with Points from Image 1
lines_img2_opencv = compute_correspond_epilines(points=src_inliers_cv, which_image=1, F=F_cv)
img1, img2 = drawlines(images[1], images[0], lines_img2_opencv, src_inliers_cv, dst_inliers_cv)
axes[1, 0].set_title("Image 1 corresponding points (OpenCV)")
axes[1, 0].imshow(img2)
axes[1, 0].axis('off')
axes[1, 1].set_title("Image 2 epilines OpenCV")
axes[1, 1].imshow(img1)
axes[1, 1].axis('off')

plt.axis('off')
plt.show()



# Plot epipolar lines and corresponding points
fig, axes = plt.subplots(2, 2, figsize=(15, 15))  # 2×2 grid
fig.suptitle("Epipolar Lines and Corresponding Points", fontsize=16)

# Custom RANSAC - Epipolar Lines on Image 1 with Points from Image 2
lines_img1_custom = compute_correspond_epilines(points=dst_inliers, which_image=2, F=F_custom)
img1, img2 = drawlines(images[0], images[1], lines_img1_custom, dst_inliers, src_inliers)
axes[0, 0].set_title("Image 1 epilines (Custom)")
axes[0, 0].imshow(img1)
axes[0, 0].axis('off')
axes[0, 1].set_title("Image 2 corresponding points (Custom)")
axes[0, 1].imshow(img2)
axes[0, 1].axis('off')

# OpenCV RANSAC - Epipolar Lines on Image 1 with Points from Image 2
lines_img1_opencv = compute_correspond_epilines(points=dst_inliers_cv, which_image=2, F=F_cv)
img1, img2 = drawlines(images[0], images[1], lines_img1_opencv, dst_inliers_cv, src_inliers_cv)
axes[1, 0].set_title("Image 1 epilines (OpenCV)")
axes[1, 0].imshow(img1)
axes[1, 0].axis('off')
axes[1, 1].set_title("Image 2 corresponding points (OpenCV)")
axes[1, 1].imshow(img2)
axes[1, 1].axis('off')

plt.axis('off')
plt.show()
#----------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------- 5 2D Points Triangulation ----------------------------------------------
# Load camera intrinsic matrices
data_intrinsics = np.load("data/intrinsics.npz")
K1 = data_intrinsics['K1']
K2 = data_intrinsics['K2']

# Load good correspondences
data_correspondences = np.load("data/good_correspondences.npz")
pts1 = data_correspondences['pts1']  # Points in image 1
pts2 = data_correspondences['pts2']  # Points in image 2

# Compute essential matrices
E_custom = K2.T @ F_custom @ K1  # From your fundamental matrix
E_opencv = K2.T @ F_cv @ K1  # From OpenCV's fundamental matrix

# Decompose essential matrices
R1_custom, R2_custom, t_custom = cv2.decomposeEssentialMat(E_custom)
R1_opencv, R2_opencv, t_opencv = cv2.decomposeEssentialMat(E_opencv)

# Four possible extrinsics for camera 2
extrinsics_custom = [
    (R1_custom, t_custom),
    (R1_custom, -t_custom),
    (R2_custom, t_custom),
    (R2_custom, -t_custom),
]

extrinsics_opencv = [
    (R1_opencv, t_opencv),
    (R1_opencv, -t_opencv),
    (R2_opencv, t_opencv),
    (R2_opencv, -t_opencv),
]


# Projection matrix for camera 1
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))

# Projection matrices for camera 2 (four possibilities)
P2_custom_candidates = [K2 @ np.hstack((R, t.reshape(-1, 1))) for R, t in extrinsics_custom]
P2_opencv_candidates = [K2 @ np.hstack((R, t.reshape(-1, 1))) for R, t in extrinsics_opencv]


def triangulation(P1, pts1, P2, pts2):
    """
    Triangulate pairs of 2D points in the images to a set of 3D points
    :param P1: numpy.array(float), an array 3x4 that holds the projection matrix of camera 1
    :param pts1: numpy.array(float), an array Nx2 that holds the 2D points on image 1
    :param P2: numpy.array(float), an array 3x4 that holds the projection matrix of camera 2
    :param pts2: numpy.array(float), an array Nx2 that holds the 2D points on image 2
    :return:
    pts3d: numpy.array(float), an array Nx3 that holds the reconstructed 3D points
    """

    assert pts1.shape == pts2.shape # Point sets must have the same dimensions
    assert pts1.ndim == 2 and pts1.shape[1] == 2 # Points must be Nx2 arrays
    assert P1.shape == (3, 4) and P2.shape == (3, 4) # Projection matrices must be 3x4

    pts3d = []

    for p1, p2 in zip(pts1, pts2):
        A = np.array([
            p1[0] * P1[2] - P1[0],
            p1[1] * P1[2] - P1[1],
            p2[0] * P2[2] - P2[0],
            p2[1] * P2[2] - P2[1]
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1]
        X = X[:3] / X[3]  # Convert from homogeneous to Cartesian coordinates
        pts3d.append(X)

    return np.array(pts3d)


def points_in_front(pts3d, P):
    #Check if 3D points lie in front of the camera.

    homogenous_pts = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    projected = P @ homogenous_pts.T
    return np.all(projected[2, :] > 0)

# Find correct extrinsic matrix
for P2 in P2_custom_candidates:
    pts3d_custom = triangulation(P1, pts1, P2, pts2)
    if points_in_front(pts3d_custom, P1) and points_in_front(pts3d_custom, P2):
        best_pts3d_custom = pts3d_custom
        break

for P2 in P2_opencv_candidates:
    pts3d_opencv = triangulation(P1, pts1, P2, pts2)
    if points_in_front(pts3d_opencv, P1) and points_in_front(pts3d_opencv, P2):
        best_pts3d_opencv = pts3d_opencv
        break


# Plot 3D points for both implementations
plot_3d_points(best_pts3d_custom, fig_title="Custom Implementation")
plot_3d_points(best_pts3d_opencv, fig_title="OpenCV Implementation")
#----------------------------------------------------------------------------------------------------------------------------
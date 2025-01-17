import numpy as np
import matplotlib.pyplot as plt
import cv2




#---------------------------------------------------1 Harris Corner Detector-----------------------------------------------------
images = []
images_gray = []
for i in range(1, 7):
    image = cv2.imread(f'data/corners/corner_{i}.png', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image)
    images_gray.append(image_gray)


plt.figure(figsize=(10, 5))
for i, img in enumerate(images):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.suptitle('Images')
plt.show(block=False)
plt.pause(0.1)


def detect_corners(input_image, max_corners=0, quality_level=0.01, min_distance=10, block_size=5, k=0.05):
    """
    Detect corners using Harris Corner Detector
    :param input_image: numpy.array(uint8 or float), input 8-bit or foating-point 32-bit, single-channel
    image
    :param max_corners: int, maximum number of corners to return, if 0 then return all
    :param quality_level: float, parameter characterizing the minimal accepted quality of image corners
    :param min_distance: float, minimum possible Euclidean distance between the returned corners
    :param block_size: int, size of an average block for computing a derivative covariation matrix
    over each pixel neighborhood.
    :param k: float, free parameter of the Harris detector
    :return:
    corners: numpy.array(uint8)), corner coordinates for each input image
    """

    assert isinstance(max_corners, int) and max_corners >= 0, "max_corners must be a non-negative integer"
    assert 0 <= quality_level <= 1, "quality_level must be in [0, 1]"
    assert min_distance >= 0, "min_distance must be a non-negative number"
    assert isinstance(block_size,int) and block_size > 0 and block_size % 2 == 1, "block_size must be a positive odd integer"
    assert 0.04 <= k <= 0.06, "k must be in [0.04, 0.06]"

    # Pad image
    input_image = cv2.copyMakeBorder(src=input_image, top=block_size // 2, bottom=block_size // 2, left=block_size // 2, right=block_size // 2,borderType=cv2.BORDER_REPLICATE)

    #Compute gradients using Sobel
    Ix = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)

    Iy = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)


    #Compute products of gradients
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    half_block = block_size // 2
    height, width = input_image.shape

    # Initialize the response matrix R with zeros
    R = np.zeros_like(input_image, dtype=np.float64)

    #Constructing M matrix for each pixel
    for y in range(half_block, height - half_block):
        for x in range(half_block, width - half_block):
            # Extract the neighborhood window
            window_Ixx = Ixx[y - half_block: y + half_block + 1, x - half_block: x + half_block + 1]
            window_Ixy = Ixy[y - half_block: y + half_block + 1, x - half_block: x + half_block + 1]
            window_Iyy = Iyy[y - half_block: y + half_block + 1, x - half_block: x + half_block + 1]

            # Sum of products to form M
            Sxx = np.sum(window_Ixx)
            Sxy = np.sum(window_Ixy)
            Syy = np.sum(window_Iyy)


            # Covariance matrix M for this pixel
            M = np.array([[Sxx, Sxy], [Sxy, Syy]])

            # Calculate R = det(M) - k * (trace(M)^2) and store it
            det_M = np.linalg.det(M)
            trace_M = np.trace(M)
            R[y, x] = det_M - k * (trace_M ** 2)






    # Identify candidate corners by thresholding the R-score
    threshold = quality_level * R.max()
    corners = np.argwhere(R > threshold)

    # Sort corners by R-score in descending order
    corners = sorted(corners, key=lambda x: R[x[0], x[1]], reverse=True)



    # Apply minimum distance filtering using KDTree
    if min_distance > 0:
        from scipy.spatial import KDTree
        kdtree = KDTree(corners)
        selected_corners = []
        for i, corner in enumerate(corners):
            if not any([np.linalg.norm(corner - np.array(other)) < min_distance for other in selected_corners]):
                selected_corners.append(corner)
        corners = selected_corners

    # Limit to max_corners if specified
    if max_corners > 0:
        corners = corners[:max_corners]

    # Adjust corners to remove padding effect
    padding_offset = block_size // 2
    corners = np.array([[y - padding_offset, x - padding_offset] for y, x in corners])

    return np.array(corners)



quality_level, max_corners, min_distance, block_size, k = 0.01, 0, 10.0, 5, 0.05


# Initialize two separate figures for two 2x3 grids
fig1 = plt.figure(figsize=(10, 5))
fig2 = plt.figure(figsize=(10, 5))

# Loop through the images to plot the custom and OpenCV corners on separate grids
for i in range(0, len(images)):
    # Detect corners using custom implementation
    corners_custom = detect_corners(images_gray[i], max_corners, quality_level, min_distance, block_size, k)

    # Detect corners using OpenCV's goodFeaturesToTrack
    corners_cv = cv2.goodFeaturesToTrack(
        image=images_gray[i], maxCorners=max_corners, qualityLevel=quality_level,
        minDistance=min_distance, blockSize=block_size, useHarrisDetector=True, k=k
    )

    # Draw corners on copies of the original images
    img_custom = np.copy(images[i])
    img_cv = np.copy(images[i])

    # Draw custom corners in green
    for corner in corners_custom:
        cv2.circle(img_custom, (corner[1], corner[0]), 5, (0, 255, 0), -1)

    # Draw OpenCV corners in red
    if corners_cv is not None:
        for corner in corners_cv:
            x, y = int(corner[0][0]), int(corner[0][1])
            cv2.circle(img_cv, (x, y), 5, (0, 0, 255), -1)

    # Add to first figure (Custom Detector)
    plt.figure(fig1)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_custom)
    plt.title(f"Custom Detector {i+1}")

    # Add to second figure (OpenCV Detector)
    plt.figure(fig2)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_cv)
    plt.title(f"OpenCV Detector {i+1}")

# Adjust layout for both figures and display
plt.figure(fig1)
plt.tight_layout()


plt.figure(fig2)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

#-----------------------------------------------------------------------------------------------------------------------------------






#---------------------------------------------------2 ImageStitching------------------------------------------------------------------

#Load and display images
images = []
for i in range(1, 6):
    image = cv2.imread(f'data/panoramas/pano_{i}.jpg', cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

plt.figure("Images", figsize=(15, 5))
for i in range(0, 5):
    plt.subplot(1, 5, 1+i)
    plt.axis("off")
    plt.imshow(images[i], cmap='gray')
plt.suptitle('Images')
plt.show(block=False)
plt.pause(0.1)



#Convert to grayscale and detect SIFT features
gray_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]

sift = cv2.SIFT_create()
keypoints, descriptors = [], []
for img in gray_images:
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

# Plot keypoints for each image
fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
for ax, img, kp in zip(axes, images, keypoints):
    img_keypoints = np.empty_like(img)
    cv2.drawKeypoints(img, kp, img_keypoints)
    ax.imshow(img_keypoints)
    ax.axis('off')
plt.suptitle('Sift featutes')
plt.show(block=False)
plt.pause(0.1)



#Nearest neighbor matching with ratio test
def match_features(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append([m])
    return good_matches


# 2x2 grid for the matches visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle("Good Correspondences")

# Find and plot good matches for adjacent image pairs
good_matches_all = []
for i in range(len(images) - 1):
    good_matches = match_features(descriptors[i], descriptors[i + 1])
    good_matches_all.append(good_matches)

    img_matches = np.empty((max(images[i].shape[0], images[i+1].shape[0]), images[i].shape[1] + images[i+1].shape[1], 3),dtype=np.uint8)
    cv2.drawMatchesKnn(images[i], keypoints[i], images[i+1], keypoints[i+1], outImg=img_matches, matches1to2=good_matches, flags=2)

    ax = axes[i // 2, i % 2]  # Row and column for subplot

    # Display the img_matches in the respective subplot
    ax.imshow(img_matches)
    ax.set_title(f"Good Matches between Image {i + 1} and Image {i + 2}")
    ax.axis('off')

plt.show(block=False)
plt.pause(0.1)

def ransac(src_points, dst_points, ransac_reproj_threshold, max_iters, inlier_ratio=0.8):
    """
    Estimate the homography transformation using the RANSAC algorithm,while
    identifying inlier correspondences.
    :param src_points: numpy.array(float), coordinates of points in the source image
    :param dst_points: numpy.array(float), coordinates of points in the destination
    image
    :param ransac_reproj_threshold:float, maximum reprojection error allowed to
    classify a point pair as an inlier
    :param max_iters: int, maximum number of RANSAC iterations
    :param inlier_ratio: float,the desired ratio of inliers to total correspondences
    return:
    H:numpy.array(float), the estimated homography matrix using linear
    least-squares
    mask: numpy.array(uint8),mask indicating the inlier correspondences
    """

    assert src_points.shape == dst_points.shape, "src and dst points must have the same dimensions"
    assert ransac_reproj_threshold >= 0, "Threshold must be non-negative"
    assert isinstance(max_iters, int) and max_iters > 0, "max_iters must be a positive non-zero integer"
    assert 0 <= inlier_ratio <= 1, "inlier_ratio must be in range [0, 1]"

    n_points = src_points.shape[0]
    best_inliers = 0
    best_H = None
    best_mask = None

    for i in range(max_iters):

        # Randomly select 4 points to compute the homography
        indices = np.random.choice(n_points, 4, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        # Compute the homography matrix H using the 4-point samples
        H = cv2.getPerspectiveTransform(src_sample.astype(np.float32), dst_sample.astype(np.float32))


        # Create an array to hold inlier status for each point
        inliers_current = np.zeros(n_points, dtype=np.uint8)
        inlier_count = 0

        # Transform all source points using the current homography
        homogeneous_src_points = np.hstack((src_points, np.ones((n_points, 1))))  # Convert to homogeneous coordinates


        # Transform all source points using the current homography
        transformed_points = np.dot(homogeneous_src_points, H.T)  # Apply homography
        transformed_points /= transformed_points[:, 2][:, np.newaxis]  # Normalize by the third coordinate

        # Calculate distances between transformed source points and corresponding destination points
        distances = np.linalg.norm(dst_points - transformed_points[:, :2], axis=1)

        # Determine inliers based on reprojection threshold
        inliers_current = (distances <= ransac_reproj_threshold).astype(np.uint8)
        inlier_count = np.sum(inliers_current)

        # Update the best homography and mask if current model is better
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_H = H
            best_mask = inliers_current.copy()

        # Check if we have sufficient inliers to stop early
        if inlier_count >= inlier_ratio * n_points:
            break



    # Recompute H using only inliers, with least-squares method
    inlier_src_points = src_points[best_mask == 1]
    inlier_dst_points = dst_points[best_mask == 1]

    # Construct the final homography using least-squares (pseudo-inverse) for inliers
    A = []
    B = []
    for (src, dst) in zip(inlier_src_points, inlier_dst_points):
        x, y = src
        u, v = dst
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B.append(u)
        B.append(v)
    A = np.array(A)
    B = np.array(B)

    # Solve for homography using the pseudo-inverse
    h = np.dot(np.linalg.pinv(A), B)
    H = np.append(h, 1).reshape(3, 3)


    mask = best_mask

    print("H: ", H)
    print("mask: ", mask)

    return H, mask





# Use RANSAC to find homography for each pair of images (Custom and OpenCV)
ransac_reprojection_threshold, max_iters = 1.0, 1000
mask_custom = []
homographies_custom = []
mask_cv = []
homographies_cv = []
for i in range(len(images) - 1):
    src_pts = np.float32([keypoints[i][m[0].queryIdx].pt for m in good_matches_all[i]])
    dst_pts = np.float32([keypoints[i + 1][m[0].trainIdx].pt for m in good_matches_all[i]])

    cv_H, cv_mask = cv2.findHomography(srcPoints= src_pts, dstPoints = dst_pts, method = cv2.RANSAC,ransacReprojThreshold = ransac_reprojection_threshold, maxIters = max_iters)
    homographies_cv.append(cv_H)
    mask_cv.append(cv_mask)

    H, mask = ransac(src_pts, dst_pts, ransac_reproj_threshold = ransac_reprojection_threshold, max_iters = max_iters)
    homographies_custom.append(H)
    mask_custom.append(mask)


#Draw custom inliers
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle("My Inliers (Removed Outliers)")
for i in range (0, 4):
    img_inliers=np.empty((max(images[i].shape[0],images[i+1].shape[0]), images[i].shape[1]+ images[i+1].shape[1],3),dtype=np.uint8)
    good=np.array(good_matches_all[i])
    inliers= good[np.where(np.squeeze(mask_custom[i])==1)[0]]
    cv2.drawMatchesKnn(images[i],keypoints[i],images[i+1],keypoints[i+1],outImg=img_inliers,matches1to2=inliers,flags=2)
    ax = axes[i // 2, i % 2]

    # Display the img_matches in the respective subplot
    ax.imshow(img_inliers)
    ax.set_title(f"Inliers between Image {i + 1} and Image {i + 2}")
    ax.axis('off')
plt.show(block=False)
plt.pause(0.1)


#Draw OpenCV inliers
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle("OpenCV Inliers (Removed Outliers)")
for i in range (0, 4):
    img_inliers=np.empty((max(images[i].shape[0],images[i+1].shape[0]), images[i].shape[1]+ images[i+1].shape[1],3),dtype=np.uint8)
    good=np.array(good_matches_all[i])
    inliers= good[np.where(np.squeeze(mask_cv[i])==1)[0]]
    cv2.drawMatchesKnn(images[i],keypoints[i],images[i+1],keypoints[i+1],outImg=img_inliers,matches1to2=inliers,flags=2)
    ax = axes[i // 2, i % 2]  # Row and column for subplot

    # Display the img_matches in the respective subplot
    ax.imshow(img_inliers)
    ax.set_title(f"Inliers between Image {i + 1} and Image {i + 2}")
    ax.axis('off')
plt.show(block=False)
plt.pause(0.1)


def create_panorama(images, homographies):

    def stitch_and_blend(image1, image2, H):
        #Stitch images
        panorama_height =np.maximum(image1.shape[0],image2.shape[0])
        panorama_width =image1.shape[1] +image2.shape[1]
        panorama=np.zeros((panorama_height,panorama_width,3),dtype=np.uint8)
        panorama[0:image1.shape[0],0:image1.shape[1]]=image1
        warped_img=cv2.warpPerspective(image2,H,(panorama_width,panorama_height), flags=cv2.WARP_INVERSE_MAP)
        #Blending
        temp_panorama=np.round(0.5 * panorama+0.5*warped_img).astype(np.uint8)
        temp_panorama[warped_img==[0, 0,0]]=panorama[warped_img == [0,0,0]]
        temp_panorama[panorama== [0, 0,0]]=warped_img[panorama ==[0,0,0]]
        panorama=temp_panorama.copy()

        return panorama

    pano1 = stitch_and_blend(images[3], images[4], homographies[3])
    pano2 = stitch_and_blend(images[2], pano1, homographies[2])
    pano3 = stitch_and_blend(images[1], pano2, homographies[1])
    pano = stitch_and_blend(images[0], pano3, homographies[0])

    return pano


# Create panoramas
panorama_custom = create_panorama(images, homographies_custom)
panorama_cv = create_panorama(images, homographies_cv)

# Display both panoramas in a 2x1 grid
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.imshow(panorama_custom)
plt.title("My Panorama")
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(panorama_cv)
plt.title("OpenCV Panorama")
plt.axis('off')

plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------







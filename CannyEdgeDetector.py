import numpy as np
import matplotlib.pyplot as plt
import cv2


image = cv2.imread(filename="building.jpg", flags=cv2.IMREAD_COLOR)



#--------------------------------------------------1 2D Convolution----------------------------------------------------------------

# def convolution_2D(arr, kernel, border_type):
#     """
#     Calculate the 2D convolution kernel*arr
#     :param arr: numpy.array(float), input array
#     :param kernel: numpy.array(float), convolution kernel of nxn size (only odd dimensions are allowed)
#     :param border_type: int, padding method (OpenCV)
#     :return:
#     conv_arr: numpy.array(float), convolution output
#     """
#
#     # Check if the kernel is of square odd size
#     assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1, "Array must be square, and have an odd size."
#
#     # Flip the kernel
#     kernel = np.fliplr(kernel)  # Flip array in the left/right direction
#     kernel = np.flipud(kernel)  # Flip array in the up/down direction
#
#     # Initializing conv_arr
#     conv_arr = np.zeros_like(arr)
#
#     #Padding input array according to kernel size
#     pad_size = kernel.shape[0] // 2
#     arr_pad = cv2.copyMakeBorder(src=arr, top=pad_size, bottom=pad_size, left=pad_size, right=pad_size, borderType=border_type)
#
#     if arr.ndim == 3:
#         for c in range(arr.shape[2]): # Channels
#             for i in range(pad_size, arr_pad.shape[0]-pad_size): # Rows
#                 for j in range(pad_size, arr_pad.shape[1]-pad_size): # Columns
#                     conv_sum=0
#                     for l in range(-(kernel.shape[0]//2), kernel.shape[0]//2 + 1): # Kernel Rows
#                         for m in range(-(kernel.shape[1]//2), kernel.shape[1]//2 + 1): # Kernel Columns
#                             conv_sum+= arr_pad[i+l, j+m, c] * kernel[l+kernel.shape[0]//2, m+kernel.shape[1]//2]
#
#                     conv_arr[i-pad_size, j-pad_size, c] = conv_sum
#
#     elif arr.ndim == 2:
#         for i in range(pad_size, arr_pad.shape[0] - pad_size):  # Rows
#             for j in range(pad_size, arr_pad.shape[1] - pad_size):  # Columns
#                 conv_sum = 0
#                 for l in range(-(kernel.shape[0] // 2), kernel.shape[0] // 2 + 1):  # Kernel Rows
#                     for m in range(-(kernel.shape[1] // 2), kernel.shape[1] // 2 + 1):  # Kernel Columns
#                         conv_sum += arr_pad[i + l, j + m] * kernel[
#                             l + kernel.shape[0] // 2, m + kernel.shape[1] // 2]
#
#                 conv_arr[i - pad_size, j - pad_size] = conv_sum
#
#
#     return conv_arr
#----------------------------------------------------------------------------------------------------------------------------------------------------------






#--------------------------------------------------6 Extra Credits: “Flattened” Convolution----------------------------------------------------------------
def convolution_2D(arr, kernel, border_type):
    """
    Calculate the 2D convolution kernel*arr
    :param arr: numpy.array(float), input array
    :param kernel: numpy.array(float), convolution kernel of nxn size (only odd dimensions are allowed)
    :param border_type: int, padding method (OpenCV)
    :return:
    conv_arr: numpy.array(float), convolution output
    """

    # Check if the kernel is of square odd size
    assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1, "Array must be square, and have an odd size."

    # Flip the kernel
    kernel = np.fliplr(kernel)  # Flip array in the left/right direction
    kernel = np.flipud(kernel)  # Flip array in the up/down direction

    # Initializing conv_arr
    conv_arr = np.zeros_like(arr)

    #Padding input array according to kernel size
    pad_size = kernel.shape[0] // 2
    arr_pad = cv2.copyMakeBorder(src=arr, top=pad_size, bottom=pad_size, left=pad_size, right=pad_size, borderType=border_type)

    # Flatten the kernel into a 1D vector
    kernel_flattened = kernel.flatten()

    if arr.ndim == 2:  # For 2D grayscale images
        # Get image dimensions after padding
        img_h, img_w = arr_pad.shape
        k_h, k_w = kernel.shape

        # Calculate output size
        output_h = img_h - k_h + 1
        output_w = img_w - k_w + 1

        # Create a matrix to hold the flattened image patches
        arr_col = np.zeros((output_h * output_w, k_h * k_w))

        # Fill the columns with flattened patches of the image
        col = 0
        for i in range(output_h):
            for j in range(output_w):
                patch = arr_pad[i:i + k_h, j:j + k_w].flatten()
                arr_col[col, :] = patch
                col += 1

        # Perform matrix multiplication between flattened image patches and kernel
        result = arr_col.dot(kernel_flattened)

        # Reshape the result back to the output shape
        output_shape = (arr.shape[0], arr.shape[1])
        conv_arr = result.reshape(output_shape)

    elif arr.ndim == 3:  # For RGB images or multi-channel
        conv_arr = np.zeros_like(arr)
        for c in range(arr.shape[2]):  # Loop over channels
            arr_channel = arr[:, :, c]

            # Pad the individual channel
            arr_pad_channel = cv2.copyMakeBorder(src=arr_channel, top=pad_size, bottom=pad_size, left=pad_size,
                                                 right=pad_size, borderType=border_type)

            # Get dimensions for this channel
            img_h, img_w = arr_pad_channel.shape
            k_h, k_w = kernel.shape

            # Calculate output size
            output_h = img_h - k_h + 1
            output_w = img_w - k_w + 1

            # Create matrix to hold the flattened image patches (im2col) for this channel
            arr_col = np.zeros((output_h * output_w, k_h * k_w))

            # Fill the columns with flattened patches of the image for this channel
            col = 0
            for i in range(output_h):
                for j in range(output_w):
                    patch = arr_pad_channel[i:i + k_h, j:j + k_w].flatten()
                    arr_col[col, :] = patch
                    col += 1

            # Perform matrix multiplication between flattened patches and kernel
            result = arr_col.dot(kernel_flattened)

            # Reshape the result back to the output shape for the current channel
            output_shape = (arr_channel.shape[0], arr_channel.shape[1])
            conv_arr[:, :, c] = result.reshape(output_shape)


    return conv_arr
#----------------------------------------------------------------------------------------------------------------------------------------------------------







#--------------------------------------------------2 Noise Reduction----------------------------------------------------------------

def gaussian_kernel_2D(ksize, sigma):
    """
    Calculate a 2D Gaussian kernel
    :param ksize: int, size of 2d kernel, always needs to be an odd number
    :param sigma: float, standard deviation of gaussian
    :return:
    kernel: numpy.array(float), ksize x ksize gaussian kernel with mean=0
    """

    # Check if the input size is an odd positive non-zero number
    assert ksize % 2 == 1 and ksize > 0, "Input size must be an odd positive non-zero number."

    # Check if sigma is a non-zero positive number
    assert sigma > 0, "Sigma must be a non-zero positive number"


    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2

    for x in range(ksize):
        for y in range(ksize):
            # Calculate the distance from the center
            x_dist = x - center
            y_dist = y - center
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x_dist ** 2 + y_dist ** 2) / (2 * sigma ** 2))

    # Normalize the kernel to ensure the sum equals 1
    kernel /= np.sum(kernel)
    return kernel





# Define parameters
ksize = 5  # Kernel size (must be odd)
sigma = 1.0  # Standard deviation

# Generate the Gaussian kernel
kernel = gaussian_kernel_2D(ksize, sigma)

# Convolve the image with the Gaussian kernel
blurred_image = convolution_2D(image, kernel, 1)

# Plot the original and blurred images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Blurred Image")
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show(block=False)
plt.pause(0.1)

#--------------------------------------------------3 Image Gradient----------------------------------------------------------------

def sobel_x(arr):
    """
    Calculate the 1st order partial derivatives along x-axis
    :param arr: numpy.array(float), input image
    :return:
    dx: numpy.array(float), output partial derivative
    """

    sobel_kernel_x = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])

    # Convolve the image with Sobel kernel
    dx = convolution_2D(arr, sobel_kernel_x, 1)

    # Normalize the output to range (-1, +1)
    dx = cv2.normalize(dx, None, -1, 1, cv2.NORM_MINMAX)


    return dx


def sobel_y(arr):
    """
    Calculate 1st the order partial derivatives along y-axis
    :param arr: numpy.array(float), input image
    :return:
    dy: numpy.array(float), output partial derivatives
    """

    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

    # Convolve the image with Sobel kernel
    dy = convolution_2D(arr, sobel_kernel_y, 1)

    # Normalize the output to range (-1, +1)
    dy = cv2.normalize(dy, None, -1, 1, cv2.NORM_MINMAX)


    return dy



# Convert to grayscale
BlurredImgGray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY).astype(np.float32)



dx = sobel_x(BlurredImgGray)
dy = sobel_y(BlurredImgGray)

# Plot Ix and Iy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Sobel X (Ix)")
plt.imshow(dx, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sobel Y (Iy)")
plt.imshow(dy, cmap='gray')
plt.axis('off')

plt.show(block=False)
plt.pause(0.1)

# Calculate magnitude
magnitude = np.sqrt(dx**2 + dy**2)

# Normalize magnitude to range [0, 1]
magnitiude=cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)



direction = np.degrees(np.arctan2(dy, dx))  # Convert radians to degrees


hue = (direction + 360) % 360  # Ensure hue is in [0, 360]
hue=cv2.normalize(hue, None, 0, 180, cv2.NORM_MINMAX).astype(np.uint8)



# Normalize magnitude to [0, 255] for saturation
magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Set value to 255 (full brightness)
value = np.ones_like(hue) * 255  # Full brightness

# Create the HSV image
hsv_image = cv2.merge([hue, magnitude_normalized, value])  #HSV order is [H, S, V]

# Convert HSV to BGR (optional, if you want to display the image)
bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)




# Plot Magnitude and Direction
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Magnitude")
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Direction")
plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.axis('off')
plt.show(block=False)
plt.pause(0.1)

#--------------------------------------------------4 Non-maximum Suppression----------------------------------------------------------------


def non_maximum_suppression(arr_mag, arr_dir):
    """
    Perform Non-Maximum Suppression to thin the edges by keeping only local maxima along gradient direction.
    :param arr_mag: numpy.array(float), input image gradient magnitude
    :param arr_dir: numpy.array(float), input image gradient direction (in degrees, from -180 to 180)
    :return: arr_local_maxima: numpy.array(float), thinned edges after non-maximum suppression
    """

    # Initialize the output array (copy of magnitude array)
    arr_local_maxima = np.copy(arr_mag)

    # Get the image dimensions
    rows, cols = arr_mag.shape


    # Iterate over every pixel
    for i in range(0, rows - 1):
        for j in range(0, cols - 1):
            # Get the direction of the gradient for the current pixel
            direct = arr_dir[i, j]


            # Binning the gradient direction into 8 major directions
            # One if for each 2 opposite directions (up-down, right-left, diagonal1 and diagonal2)

            if (-22.5 <= direct < 22.5) or (157.5 <= direct <= 180) or (-180 <= direct <= -157.5):
                # Horizontal edge (0° or 180°), check left and right neighbors
                neighbors = (arr_mag[i, j - 1], arr_mag[i, j + 1])  # left, right
            elif (22.5 <= direct < 67.5) or (-157.5 <= direct < -112.5):
                # Diagonal edge (45° or -135°), check top-left and bottom-right
                neighbors = (arr_mag[i - 1, j - 1], arr_mag[i + 1, j + 1])  # top-left, bottom-right
            elif (67.5 <= direct < 112.5) or (-112.5 <= direct < -67.5):
                # Vertical edge (90° or -90°), check top and bottom neighbors
                neighbors = (arr_mag[i - 1, j], arr_mag[i + 1, j])  # top, bottom
            elif (112.5 <= direct < 157.5) or (-67.5 <= direct < -22.5):
                # Diagonal edge (135° or -45°), check top-right and bottom-left
                neighbors = (arr_mag[i - 1, j + 1], arr_mag[i + 1, j - 1])  # top-right, bottom-left

            # if the magnitude is smaller than the neighbors(not the maximum), set it to 0
            if arr_mag[i, j] < neighbors[0] or arr_mag[i, j] < neighbors[1]:
                arr_local_maxima[i, j] = 0

    return arr_local_maxima




edge_thin = non_maximum_suppression(magnitude, direction)

#Plot thinned edges
plt.figure()
plt.imshow(edge_thin, cmap='gray')
plt.title("Thinned Edges")
plt.axis("off")
plt.show(block=False)
plt.pause(0.1)




#--------------------------------------------------5 Hysteresis Thresholding----------------------------------------------------------------
def hysteresis_thresholding(arr, low_ratio, high_ratio):
    """
    Use the low and high ratios to threshold the non-maximum suppression image and then link
    non-weak edges
    :param arr: numpy.array(float), input non-maximum suppression image
    :param low_ratio: float, low threshold ratio
    :param high_ratio: float, high threshold ratio
    :return:
    edges: numpy.array(uint8), output edges
    """
    # Calculate thresholds
    high_threshold = arr.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    # Initialize the output edge map
    edges = np.zeros_like(arr, dtype=np.uint8)

    # Mark strong edges
    strong_edges = (arr >= high_threshold)
    weak_edges = (arr >= low_threshold) & (arr < high_threshold)

    edges[strong_edges] = 255  # Mark strong edges in output image

    # Create an array for iterative checking of weak edges
    weak_edge_indices = np.argwhere(weak_edges)

    # Function to check for strong edges in the neighborhood
    def check_neighbors(x, y):
        # 8-connectivity (8 neighbors)
        neighbors = [
            (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
            (x, y - 1), (x, y + 1),
            (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
        ]
        for nx, ny in neighbors:
            if 0 <= nx < arr.shape[0] and 0 <= ny < arr.shape[1]:
                if edges[nx, ny] == 255:  # Strong edge found
                    return True
        return False

    # Iteratively link weak edges to strong edges
    for x, y in weak_edge_indices:
        if check_neighbors(x, y):
            edges[x, y] = 255  # Mark as strong edge

    return edges



my_edges = hysteresis_thresholding(edge_thin, 0.1, 0.2)
opencv_edges = cv2.Canny(image = blurred_image, threshold1 = 100, threshold2 = 200)

#Plot my edges and openCV edges
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("OpenCV edges")
plt.imshow(opencv_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("My edges")
plt.imshow(my_edges, cmap='gray')
plt.axis('off')

plt.show()





import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to load raw image data
def load_raw_image(file_path, width=1920, height=1280):
    raw_image = np.fromfile(file_path, dtype=np.uint16)
    raw_image = raw_image.reshape((height, width))
    return raw_image

# Function to normalize the image to a range [0, 255] for display
def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- AI-inspired Hybrid Denoising Filter ---
def hybrid_denoising_filter(image, kernel_size=(5, 5), num_iterations=3, alpha=0.2, beta=0.1):
    """
    Optimized Hybrid Denoising Filter.
    Combines adaptive weights, median smoothing, and edge awareness efficiently.

    Parameters:
    - image: Input noisy image.
    - kernel_size: Size of the neighborhood kernel.
    - num_iterations: Number of refinement iterations.
    - alpha: Weighting parameter for intensity differences.
    - beta: Edge sensitivity parameter.

    Returns:
    - denoised_image: Image after hybrid denoising.
    """
    height, width = image.shape
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2

    # Add padding to the image
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    denoised_image = image.copy().astype(np.float32)

    kernel = np.arange(-pad_h, pad_h + 1)
    y, x = np.meshgrid(kernel, kernel)

    for iteration in range(num_iterations):
        # Compute global gradients for the image
        gradient_x, gradient_y = np.gradient(padded_image)
        gradient_magnitude = np.sqrt(gradient_x*2 + gradient_y*2)

        # Precompute median for the image
        med_filtered = cv2.medianBlur(padded_image.astype(np.uint8), kernel_size[0])

        # Vectorized operation for adaptive weights
        for i in range(pad_h, height + pad_h):
            for j in range(pad_w, width + pad_w):
                region = padded_image[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]

                # Compute intensity differences and spatial weights
                intensity_diff = np.abs(region - padded_image[i, j])
                spatial_weight = np.exp(-alpha * intensity_diff)

                # Compute edge weight using precomputed gradient magnitude
                edge_weight = np.exp(-beta * gradient_magnitude[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1])
                combined_weights = spatial_weight * edge_weight

                # Normalize weights
                combined_weights /= np.sum(combined_weights)

                # Compute weighted average
                adaptive_value = np.sum(region * combined_weights)

                # Combine adaptive value and median value
                denoised_image[i - pad_h, j - pad_w] = 0.7 * adaptive_value + 0.3 * med_filtered[i - pad_h, j - pad_w]

        # Update the padded image for the next iteration
        padded_image = np.pad(denoised_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    return np.clip(denoised_image, 0, 255).astype(np.uint8)

# Function to display images
def display_images(original, hybrid_denoised):
    plt.figure(figsize=(8, 4))

    # Display Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Display Hybrid Denoised Image
    plt.subplot(1, 2, 2)
    plt.imshow(hybrid_denoised, cmap='gray')
    plt.title("Hybrid Denoised Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Specify the path to the raw image
    raw_image_path = 'try3.raw'  # Replace with your raw file path

    # Load and normalize the raw image
    raw_image = load_raw_image(raw_image_path)
    normalized_image = normalize_image(raw_image)

    # Apply the hybrid denoising filter
    hybrid_denoised = hybrid_denoising_filter(normalized_image, kernel_size=(5, 5), num_iterations=3, alpha=0.2, beta=0.1)

    # Display the images
    display_images(normalized_image, hybrid_denoised)

if _name_ == "_main_":
    main()
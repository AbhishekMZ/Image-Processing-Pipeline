import numpy as np
import cv2

# Parameters
input_file = "try3.raw"
width = 1920
height = 1280
gamma = 0.9  # Adjusted gamma for balanced contrast
inv_gamma = 1.0 / gamma

# Precompute a gamma correction lookup table
lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

def apply_gamma_correction(image, table):
    """Apply gamma correction using a lookup table."""
    return cv2.LUT(image, table)

# Step 1: Load 12-bit Bayer RAW image and perform Demosaicing
raw_data = np.fromfile(input_file, dtype=np.uint16)
raw_data = raw_data.reshape((height, width))
raw_data = (raw_data >> 4).astype(np.uint8)  # Normalize to 8-bit

# Bayer pattern (update if required)
bayer_pattern = cv2.COLOR_BAYER_GB2BGR  # Change this if needed (e.g., RG2BGR, BG2BGR, GR2BGR)
rgb_image = cv2.cvtColor(raw_data, bayer_pattern)

# Apply gamma correction
rgb_image = apply_gamma_correction(rgb_image, lookup_table)

# Save demosaiced image
demosaiced_file = "demosaiced_image.png"
cv2.imwrite(demosaiced_file, rgb_image)
print(f"Demosaiced image saved to {demosaiced_file}")

# Step 2: Apply White Balance (Gray World Algorithm)
avg_r = np.mean(rgb_image[:, :, 2])  # Average of Red channel
avg_g = np.mean(rgb_image[:, :, 1])  # Average of Green channel
avg_b = np.mean(rgb_image[:, :, 0])  # Average of Blue channel

scale_r = avg_g / avg_r   # Slightly increase red scaling factor
scale_b = avg_g / avg_b  # Slightly increase blue scaling factor

rgb_image[:, :, 2] = np.clip(rgb_image[:, :, 2] * scale_r, 0, 255)  # Adjust Red channel
rgb_image[:, :, 0] = np.clip(rgb_image[:, :, 0] * scale_b, 0, 255)  # Adjust Blue channel

# Apply gamma correction
rgb_image = apply_gamma_correction(rgb_image, lookup_table)

# Save white-balanced image
white_balanced_file = "white_balanced_image.png"
cv2.imwrite(white_balanced_file, rgb_image)
print(f"White-balanced image saved to {white_balanced_file}")

# Step 3: Apply Denoising (Gaussian Filter)
denoised_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)

# Apply gamma correction
denoised_image = apply_gamma_correction(denoised_image, lookup_table)

# Save denoised image
denoised_file = "denoised_image.png"
cv2.imwrite(denoised_file, denoised_image)
print(f"Denoised image saved to {denoised_file}")

# Step 4: Apply Gamma Correction again (optional brightening and contrast adjustment)
alpha = 1.2  # Slightly increase contrast
beta = 20    # Moderate brightness boost
gamma_corrected_image = cv2.convertScaleAbs(denoised_image, alpha=alpha, beta=beta)

# Save gamma-corrected image
gamma_corrected_file = "gamma_corrected_image.png"
cv2.imwrite(gamma_corrected_file, gamma_corrected_image)
print(f"Fixed gamma-corrected image saved to {gamma_corrected_file}")

# Step 5: Apply Sharpening Filter (Unsharp Mask)
blurred = cv2.GaussianBlur(gamma_corrected_image, (5, 5), 0)
sharpened_image = cv2.addWeighted(gamma_corrected_image, 1.5, blurred, -0.5, 0)

# Apply gamma correction
sharpened_image = apply_gamma_correction(sharpened_image, lookup_table)

# Save sharpened image
sharpened_file = "sharpened_image.png"
cv2.imwrite(sharpened_file, sharpened_image)
print(f"Sharpened image saved to {sharpened_file}")

def apply_contrast_stretching(image):
    stretched = np.zeros_like(image)
    for c in range(3):
        min_val = np.min(image[:, :, c])
        max_val = np.max(image[:, :, c])
        stretched[:, :, c] = ((image[:, :, c] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

# Step 5 (Continued): Enhance Colors and Correct Pink Shades
def enhance_colors_and_correct_pink(image, saturation_factor=1.7, hue_shift=-5):
    """
    Enhance colors by boosting saturation and correcting pinkish shades in the image.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Boost saturation globally
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)

    # Define a wider range for pink shades
    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]
    pink_mask = ((hue >= 120) & (hue <= 180)) & (saturation > 50)

    # Adjust pink hue, saturation, and brightness
    hsv_image[pink_mask, 0] = np.clip(hsv_image[pink_mask, 0] + hue_shift, 0, 180)
    hsv_image[pink_mask, 1] = np.clip(hsv_image[pink_mask, 1] * 1.4, 0, 255)
    hsv_image[pink_mask, 2] = np.clip(hsv_image[pink_mask, 2] * 1.2, 0, 255)

    # Convert back to BGR
    enhanced_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return enhanced_image

# Apply contrast stretching for better dynamic range
stretched_image = apply_contrast_stretching(sharpened_image)

# Enhance colors and correct pink shades
final_image = enhance_colors_and_correct_pink(stretched_image, saturation_factor=1.7, hue_shift=-5)

# Save the final enhanced image
final_file = "final_enhanced_image.png"
cv2.imwrite(final_file, final_image)
print(f"Final enhanced image saved to {final_file}")
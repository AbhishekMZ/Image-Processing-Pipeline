"""
High Dynamic Range (HDR) Image Generation and Tone Mapping.

This module implements HDR image creation through exposure fusion and tone mapping.
It processes three differently exposed images (under, mid, and over exposed) to create
a high dynamic range image that captures a wider range of luminance than standard
photographs.

The pipeline consists of two main stages:
1. HDR Creation:
   - Loads three exposure-bracketed images
   - Merges them using Mertens fusion algorithm
   - Normalizes and validates the HDR result

2. Tone Mapping:
   - Applies tone mapping to create displayable LDR image
   - Performs gamma correction for optimal display
   - Saves both HDR and tone-mapped results

Dependencies:
    - OpenCV (cv2): For HDR processing and image I/O
    - NumPy: For numerical operations and array handling

Note:
    This implementation uses OpenCV's exposure fusion which doesn't require
    actual exposure values, making it more robust for general use.

Author: [Your Name]
Date: [Current Date]
"""

import cv2
import numpy as np

# Input image paths
# ----------------
INPUT_IMAGES = {
    'under': 'underexposed.jpg',   # Underexposed image path
    'mid': 'midexposed.jpg',       # Mid-exposed image path
    'over': 'overexposed.jpg'      # Overexposed image path
}

# Output file paths
# ----------------
OUTPUT_HDR = 'hdr_image.jpg'       # Path for saving HDR result
OUTPUT_LDR = 'ldr_image.jpg'       # Path for saving tone-mapped result

# Load exposure-bracketed images
img1 = cv2.imread(INPUT_IMAGES['under'])
img2 = cv2.imread(INPUT_IMAGES['mid'])
img3 = cv2.imread(INPUT_IMAGES['over'])

# Validate image loading
if img1 is None or img2 is None or img3 is None:
    print("Error: One or more images could not be loaded.")
    print(f"Please ensure the following files exist:")
    for img_type, path in INPUT_IMAGES.items():
        print(f"  - {img_type}: {path}")
    exit()

# Create HDR image using Mertens exposure fusion
# This algorithm doesn't require exposure values and works well for most scenes
merge_mertens = cv2.createMergeMertens()
hdr = merge_mertens.process([img1, img2, img3])

# Normalize HDR values to [0, 1] range for proper processing
hdr = cv2.normalize(hdr, None, 0, 1, cv2.NORM_MINMAX)

# Handle potential numerical instabilities
if np.any(np.isnan(hdr)) or np.any(np.isinf(hdr)):
    print("Warning: HDR image contains NaN or Inf values. Applying correction...")
    hdr = np.nan_to_num(hdr)  # Replace NaN and Inf with 0

# Convert HDR to 8-bit for standard display
hdr_8bit = (hdr * 255).astype('uint8')

# Save the HDR result
cv2.imwrite(OUTPUT_HDR, hdr_8bit)
print(f"HDR image saved to: {OUTPUT_HDR}")

# Apply tone mapping for improved display
# Using Reinhard global operator with gamma correction
tonemap = cv2.createTonemap(gamma=2.2)  # gamma 2.2 is standard for most displays
ldr = tonemap.process(hdr)

# Ensure proper value range
ldr = np.clip(ldr, 0, 1)

# Convert to 8-bit for display
ldr_8bit = (ldr * 255).astype('uint8')

# Save tone-mapped result
cv2.imwrite(OUTPUT_LDR, ldr_8bit)
print(f"Tone-mapped LDR image saved to: {OUTPUT_LDR}")

print("Processing complete. Check output files for results.")

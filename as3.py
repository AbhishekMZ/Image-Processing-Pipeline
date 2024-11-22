import cv2
import numpy as np

# Load images
img1 = cv2.imread('underexposed.jpg')
img2 = cv2.imread('midexposed.jpg')
img3 = cv2.imread('overexposed.jpg')

# Check if images are loaded properly
if img1 is None or img2 is None or img3 is None:
    print("Error: One or more images could not be loaded.")
    exit()

# Merge images to create HDR using exposure fusion
merge_mertens = cv2.createMergeMertens()
hdr = merge_mertens.process([img1, img2, img3])

# Ensure HDR values are within the range [0, 1] before converting to 8-bit
hdr = cv2.normalize(hdr, None, 0, 1, cv2.NORM_MINMAX)

# Check for NaN or Inf values in HDR and replace them with 0 if present
if np.any(np.isnan(hdr)) or np.any(np.isinf(hdr)):
    print("Warning: HDR image contains NaN or Inf values.")
    hdr = np.nan_to_num(hdr)  # Replace NaN and Inf with 0

# Convert HDR to 8-bit for display
hdr_8bit = (hdr * 255).astype('uint8')

# Save HDR image
cv2.imwrite('hdr_image.jpg', hdr_8bit)

# Tonemap HDR image for display in LDR (Low Dynamic Range)
tonemap = cv2.createTonemap(gamma=2.2)  # Try adjusting gamma if needed
ldr = tonemap.process(hdr)

# Clamp LDR values to [0, 1] before converting to 8-bit
ldr = np.clip(ldr, 0, 1)

# Convert to 8-bit for display
ldr_8bit = (ldr * 255).astype('uint8')

# Save LDR image
cv2.imwrite('ldr_image.jpg', ldr_8bit)

print("HDR and LDR images have been saved.")

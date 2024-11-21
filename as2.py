import numpy as np
import cv2
import matplotlib.pyplot as plt

# Initialize gamma correction parameters
gamma = 0.9
inv_gamma = 1.0 / gamma
lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

def apply_gamma_correction(image, table=None):
    """Apply gamma correction using a lookup table."""
    if table is None:
        table = lookup_table
    return cv2.LUT(image, table)

def load_bayer_image(file_path, width=1920, height=1280):
    try:
        raw_data = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
        raw_data = (raw_data >> 4).astype(np.uint8)
        return raw_data
    except Exception as e:
        print("Error loading Bayer raw image:", e)
        return None

def demosaic_bayer(bayer_image):
    """Apply demosaicing with gamma correction."""
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GB2BGR)
    return apply_gamma_correction(rgb_image)

def apply_white_balance(image):
    """Apply improved white balance using Gray World algorithm."""
    result = image.copy().astype(np.float32)
    
    # Calculate channel averages
    avg_r = np.mean(result[:, :, 2])
    avg_g = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 0])
    
    # Calculate scaling factors
    scale_r = avg_g / avg_r
    scale_b = avg_g / avg_b
    
    # Apply scaling with clipping
    result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)  # Red
    result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)  # Blue
    
    return apply_gamma_correction(result.astype(np.uint8))

def hybrid_denoising_filter(image, kernel_size=(5, 5), num_iterations=3, alpha=0.2, beta=0.1):
    """
    Optimized Hybrid Denoising Filter from as4.py.
    """
    print("Applying hybrid denoising...")
    
    if len(image.shape) == 3:
        # Process each channel separately
        result = np.zeros_like(image)
        for channel in range(3):
            result[:, :, channel] = hybrid_denoising_channel(
                image[:, :, channel],
                kernel_size,
                num_iterations,
                alpha,
                beta
            )
        return result
    else:
        return hybrid_denoising_channel(image, kernel_size, num_iterations, alpha, beta)

def hybrid_denoising_channel(image, kernel_size, num_iterations, alpha, beta):
    """Process a single channel with hybrid denoising."""
    height, width = image.shape
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    denoised_image = image.copy().astype(np.float32)

    for iteration in range(num_iterations):
        gradient_x, gradient_y = np.gradient(padded_image)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        med_filtered = cv2.medianBlur(padded_image.astype(np.uint8), kernel_size[0])

        for i in range(pad_h, height + pad_h):
            for j in range(pad_w, width + pad_w):
                region = padded_image[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
                intensity_diff = np.abs(region - padded_image[i, j])
                spatial_weight = np.exp(-alpha * intensity_diff)
                edge_weight = np.exp(-beta * gradient_magnitude[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1])
                combined_weights = spatial_weight * edge_weight
                combined_weights /= np.sum(combined_weights)
                adaptive_value = np.sum(region * combined_weights)
                denoised_image[i - pad_h, j - pad_w] = 0.7 * adaptive_value + 0.3 * med_filtered[i - pad_h, j - pad_w]

        padded_image = np.pad(denoised_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    return np.clip(denoised_image, 0, 255).astype(np.uint8)

def apply_laplacian_filter(image, kernel_size=3, alpha=0.4):
    """Apply Laplacian filter for edge enhancement."""
    print("Applying Laplacian filter...")
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for channel in range(3):
            img_float = image[:, :, channel].astype(np.float32) / 255.0
            laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=kernel_size)
            enhanced = img_float - alpha * laplacian
            result[:, :, channel] = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        return apply_gamma_correction(result)
    else:
        img_float = image.astype(np.float32) / 255.0
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=kernel_size)
        enhanced = img_float - alpha * laplacian
        return apply_gamma_correction(np.clip(enhanced * 255, 0, 255).astype(np.uint8))

def apply_contrast_stretching(image):
    """Apply contrast stretching to each channel."""
    print("Applying contrast stretching...")
    stretched = np.zeros_like(image)
    for c in range(3):
        min_val = np.min(image[:, :, c])
        max_val = np.max(image[:, :, c])
        stretched[:, :, c] = ((image[:, :, c] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

def adjust_brightness(image, brightness_factor=1.2):
    """
    Adjust image brightness while preserving details and preventing clipping.
    Uses HSV color space to modify V (Value) channel for better detail preservation.
    """
    print("Adjusting brightness...")
    
    # Convert to HSV for better brightness adjustment
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Adjust V channel (brightness)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    
    # Convert back to BGR
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result

def display_pipeline_stages(stages_dict):
    """Display all stages of the pipeline side by side."""
    num_stages = len(stages_dict)
    cols = min(4, num_stages)  # Maximum 4 images per row
    rows = (num_stages + cols - 1) // cols
    
    plt.figure(figsize=(16, 4 * rows))
    
    for idx, (title, img) in enumerate(stages_dict.items(), 1):
        plt.subplot(rows, cols, idx)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Input file
        file_path = 'try3.raw'
        print("\nStarting ISP pipeline processing...")
        
        stages = {}  # Dictionary to store all pipeline stages
        
        # Step 1: Load and Demosaic
        print("Loading Bayer image...")
        bayer_image = load_bayer_image(file_path)
        if bayer_image is None:
            raise Exception("Failed to load Bayer image")
        stages['Raw Bayer'] = bayer_image
        cv2.imwrite("as2_step1_raw_bayer.png", bayer_image)
        
        # Step 2: Demosaicing
        print("Applying demosaicing...")
        demosaiced = demosaic_bayer(bayer_image)
        stages['Demosaiced'] = demosaiced
        cv2.imwrite("as2_step2_demosaiced.png", demosaiced)
        
        # Step 3: White Balance
        print("Applying white balance...")
        white_balanced = apply_white_balance(demosaiced)
        stages['White Balanced'] = white_balanced
        cv2.imwrite("as2_step3_white_balanced.png", white_balanced)
        
        # Step 4: Denoising
        print("Applying denoising...")
        denoised = hybrid_denoising_filter(
            white_balanced,
            kernel_size=(5, 5),
            num_iterations=3,
            alpha=0.2,
            beta=0.1
        )
        stages['Denoised'] = denoised
        cv2.imwrite("as2_step4_denoised.png", denoised)
        
        # Step 5: Edge Enhancement
        print("Applying edge enhancement...")
        edge_enhanced = apply_laplacian_filter(denoised, kernel_size=3, alpha=0.4)
        stages['Edge Enhanced'] = edge_enhanced
        cv2.imwrite("as2_step5_edge_enhanced.png", edge_enhanced)
        
        # Step 6: Contrast Stretching
        print("Applying contrast stretching...")
        contrast_stretched = apply_contrast_stretching(edge_enhanced)
        stages['Contrast Stretched'] = contrast_stretched
        cv2.imwrite("as2_step6_contrast_stretched.png", contrast_stretched)
        
        # Step 7: Brightness Adjustment
        print("Adjusting brightness...")
        brightened = adjust_brightness(contrast_stretched, brightness_factor=1.2)
        stages['Final'] = brightened
        cv2.imwrite("as2_step7_final.png", brightened)
        
        # Display all stages
        print("\nDisplaying all pipeline stages...")
        display_pipeline_stages(stages)
        
        print("\nProcessing complete! All intermediate and final images have been saved.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

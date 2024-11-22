"""
Advanced Image Signal Processing Pipeline
This module implements a complete ISP pipeline including demosaicing,
white balance, denoising, and enhancement operations.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Pre-compute gamma correction lookup table for efficiency
gamma = 0.9
inv_gamma = 1.0 / gamma
lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

def apply_gamma_correction(image, table=None):
    """
    Apply gamma correction to adjust image luminance.
    Args:
        image: Input image array
        table: Optional pre-computed lookup table
    Returns:
        Gamma-corrected image
    """
    if table is None:
        table = lookup_table
    return cv2.LUT(image, table)

def load_bayer_image(file_path, width=1920, height=1280):
    """
    Load and preprocess raw Bayer pattern image.
    Args:
        file_path: Path to raw image file
        width: Image width (default: 1920)
        height: Image height (default: 1280)
    Returns:
        Processed Bayer pattern array or None if error
    """
    try:
        raw_data = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
        raw_data = (raw_data >> 4).astype(np.uint8)  # Convert 12-bit to 8-bit
        return raw_data
    except Exception as e:
        print("Error loading Bayer raw image:", e)
        return None

def demosaic_bayer(bayer_image):
    """
    Convert Bayer pattern to RGB with gamma correction.
    Args:
        bayer_image: Input Bayer pattern array
    Returns:
        Demosaiced RGB image
    """
    rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GB2BGR)
    return apply_gamma_correction(rgb_image)

def apply_white_balance(image):
    """
    Implement Gray World white balance algorithm.
    Assumes average scene color should be neutral gray.
    Args:
        image: Input RGB image
    Returns:
        White-balanced image
    """
    result = image.copy().astype(np.float32)
    
    # Calculate per-channel averages
    avg_r = np.mean(result[:, :, 2])
    avg_g = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 0])
    
    # Compute scaling factors using green as reference
    scale_r = avg_g / avg_r
    scale_b = avg_g / avg_b
    
    # Apply channel scaling with range clipping
    result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)  # Red
    result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)  # Blue
    
    return apply_gamma_correction(result.astype(np.uint8))

def hybrid_denoising_filter(image, kernel_size=(5, 5), num_iterations=3, alpha=0.2, beta=0.1):
    """
    Advanced hybrid denoising filter combining spatial and edge-aware filtering.
    
    Args:
        image: Input image array
        kernel_size: Size of the filtering kernel (default: 5x5)
        num_iterations: Number of filtering passes (default: 3)
        alpha: Spatial weight parameter (default: 0.2)
        beta: Edge weight parameter (default: 0.1)
    
    Returns:
        Denoised image array
    """
    print("Applying hybrid denoising...")
    
    if len(image.shape) == 3:
        # Process RGB channels independently
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
    """
    Apply hybrid denoising to a single image channel.
    
    Combines bilateral filtering with edge preservation and median filtering.
    Uses iterative refinement for better noise reduction.
    
    Args:
        image: Single-channel input image
        kernel_size: Filter kernel dimensions
        num_iterations: Number of refinement iterations
        alpha: Intensity weight factor
        beta: Edge weight factor
    
    Returns:
        Denoised single-channel image
    """
    height, width = image.shape
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2

    # Add reflection padding to handle borders
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    denoised_image = image.copy().astype(np.float32)

    for iteration in range(num_iterations):
        # Compute image gradients for edge detection
        gradient_x, gradient_y = np.gradient(padded_image)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Apply median filter for comparison
        med_filtered = cv2.medianBlur(padded_image.astype(np.uint8), kernel_size[0])

        # Process each pixel with adaptive filtering
        for i in range(pad_h, height + pad_h):
            for j in range(pad_w, width + pad_w):
                region = padded_image[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
                
                # Calculate bilateral filter weights
                intensity_diff = np.abs(region - padded_image[i, j])
                spatial_weight = np.exp(-alpha * intensity_diff)
                edge_weight = np.exp(-beta * gradient_magnitude[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1])
                
                # Combine weights and normalize
                combined_weights = spatial_weight * edge_weight
                combined_weights /= np.sum(combined_weights)
                
                # Weighted average with median filter
                adaptive_value = np.sum(region * combined_weights)
                denoised_image[i - pad_h, j - pad_w] = 0.7 * adaptive_value + 0.3 * med_filtered[i - pad_h, j - pad_w]

        # Update padded image for next iteration
        padded_image = np.pad(denoised_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    return np.clip(denoised_image, 0, 255).astype(np.uint8)

def apply_laplacian_filter(image, kernel_size=3, alpha=0.4):
    """
    Enhance image edges using Laplacian filter.
    
    Args:
        image: Input image array
        kernel_size: Size of Laplacian kernel (default: 3)
        alpha: Edge enhancement strength (default: 0.4)
    
    Returns:
        Edge-enhanced image
    """
    print("Applying Laplacian filter...")
    
    # Convert to float for processing
    float_img = image.astype(np.float32)
    
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(float_img, cv2.CV_32F, ksize=kernel_size)
    
    # Add weighted Laplacian to original image
    enhanced = float_img + alpha * laplacian
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def apply_contrast_stretching(image):
    """
    Enhance image contrast using histogram stretching.
    
    Args:
        image: Input image array
    
    Returns:
        Contrast-enhanced image
    """
    print("Applying contrast stretching...")
    
    # Process each channel independently for color images
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for channel in range(3):
            min_val = np.min(image[:, :, channel])
            max_val = np.max(image[:, :, channel])
            result[:, :, channel] = ((image[:, :, channel] - min_val) * 255 / 
                                   (max_val - min_val))
        return result.astype(np.uint8)
    else:
        min_val = np.min(image)
        max_val = np.max(image)
        return ((image - min_val) * 255 / (max_val - min_val)).astype(np.uint8)

def adjust_brightness(image, brightness_factor=1.2):
    """
    Adjust image brightness while preserving details.
    
    Uses HSV color space for better detail preservation.
    
    Args:
        image: Input RGB image
        brightness_factor: Brightness multiplier (default: 1.2)
    
    Returns:
        Brightness-adjusted image
    """
    print("Adjusting brightness...")
    
    # Convert to HSV for better brightness control
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    
    # Adjust V channel (brightness)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    
    # Convert back to BGR
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted

def display_pipeline_stages(stages_dict):
    """
    Display all processing stages in a grid layout.
    
    Args:
        stages_dict: Dictionary of {stage_name: image_array}
    """
    print("\nDisplaying all pipeline stages...")
    
    num_stages = len(stages_dict)
    cols = min(3, num_stages)  # Max 3 images per row
    rows = (num_stages + cols - 1) // cols
    
    plt.figure(figsize=(15, 5*rows))
    
    for idx, (stage_name, image) in enumerate(stages_dict.items()):
        plt.subplot(rows, cols, idx + 1)
        if len(image.shape) == 2:  # Grayscale
            plt.imshow(image, cmap='gray')
        else:  # RGB
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(stage_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Execute the complete image processing pipeline.
    
    Pipeline stages:
    1. Load raw Bayer image
    2. Demosaic to RGB
    3. Apply white balance
    4. Denoise image
    5. Enhance edges
    6. Stretch contrast
    7. Adjust brightness
    """
    try:
        # Load and process input image
        file_path = 'Assignment2/data/raw/assignmentrawinput2.raw'
        print("\nStarting ISP pipeline processing...")
        
        stages = {}  # Store intermediate results
        
        # Stage 1: Load raw image
        print("Loading Bayer image...")
        bayer_image = load_bayer_image(file_path)
        if bayer_image is None:
            raise Exception("Failed to load Bayer image")
        stages['Raw Bayer'] = bayer_image
        cv2.imwrite("Assignment2/data/output/as2_step1_raw_bayer.png", bayer_image)
        
        # Stage 2: Demosaic
        print("Applying demosaicing...")
        demosaiced = demosaic_bayer(bayer_image)
        stages['Demosaiced'] = demosaiced
        cv2.imwrite("Assignment2/data/output/as2_step2_demosaiced.png", demosaiced)
        
        # Stage 3: White Balance
        print("Applying white balance...")
        white_balanced = apply_white_balance(demosaiced)
        stages['White Balanced'] = white_balanced
        cv2.imwrite("Assignment2/data/output/as2_step3_white_balanced.png", white_balanced)
        
        # Stage 4: Denoising
        print("Applying denoising...")
        denoised = hybrid_denoising_filter(
            white_balanced,
            kernel_size=(5, 5),
            num_iterations=3,
            alpha=0.2,
            beta=0.1
        )
        stages['Denoised'] = denoised
        cv2.imwrite("Assignment2/data/output/as2_step4_denoised.png", denoised)
        
        # Stage 5: Edge Enhancement
        print("Applying edge enhancement...")
        edge_enhanced = apply_laplacian_filter(denoised, kernel_size=3, alpha=0.4)
        stages['Edge Enhanced'] = edge_enhanced
        cv2.imwrite("Assignment2/data/output/as2_step5_edge_enhanced.png", edge_enhanced)
        
        # Stage 6: Contrast Enhancement
        print("Applying contrast stretching...")
        contrast_stretched = apply_contrast_stretching(edge_enhanced)
        stages['Contrast Stretched'] = contrast_stretched
        cv2.imwrite("Assignment2/data/output/as2_step6_contrast_stretched.png", contrast_stretched)
        
        # Stage 7: Brightness Adjustment
        print("Adjusting brightness...")
        brightened = adjust_brightness(contrast_stretched, brightness_factor=1.2)
        stages['Final'] = brightened
        cv2.imwrite("Assignment2/data/output/as2_step7_final.png", brightened)
        
        # Display results
        print("\nDisplaying all pipeline stages...")
        display_pipeline_stages(stages)
        
        print("\nProcessing complete! All intermediate and final images have been saved.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

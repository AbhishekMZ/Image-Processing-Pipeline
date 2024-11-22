"""
Image Analysis Module
This module provides advanced analysis tools for evaluating image quality,
including SNR measurements and tone region analysis.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_raw_image(file_path, width=1920, height=1280):
    """
    Load raw image data and convert to 8-bit format.
    
    Args:
        file_path: Path to raw image file
        width: Image width (default: 1920)
        height: Image height (default: 1280)
    
    Returns:
        Processed image array or None if error
    """
    try:
        raw_data = np.fromfile(file_path, dtype=np.uint16)
        raw_data = raw_data.reshape((height, width))
        return (raw_data >> 4).astype(np.uint8)
    except Exception as e:
        print(f"Error loading raw image: {e}")
        return None

def compute_snr(signal_region, noise_region):
    """
    Calculate Signal-to-Noise Ratio in decibels.
    
    Args:
        signal_region: Region containing the signal
        noise_region: Region containing noise
    
    Returns:
        SNR value in dB
    """
    signal_power = np.mean(signal_region)
    noise_power = np.std(noise_region)
    if noise_power == 0:
        return float('inf')
    return 20 * np.log10(signal_power / noise_power)

def analyze_tone_regions(image):
    """
    Analyze different tone regions in the image.
    
    Segments image into dark, medium, and bright regions.
    
    Args:
        image: Input image array
    
    Returns:
        Dictionary of tone regions and their masks
    """
    # Define tone region thresholds
    dark_threshold = 64
    bright_threshold = 192
    
    # Create masks for each region
    dark_mask = image < dark_threshold
    bright_mask = image > bright_threshold
    medium_mask = ~(dark_mask | bright_mask)
    
    return {
        'dark': dark_mask,
        'medium': medium_mask,
        'bright': bright_mask
    }

def apply_denoising_filters(image):
    """
    Apply various denoising filters for comparison.
    
    Args:
        image: Input image array
    
    Returns:
        Dictionary of filtered images
    """
    # Gaussian filter
    gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
    
    # Median filter
    median = cv2.medianBlur(image, 5)
    
    # Bilateral filter
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    
    return {
        'gaussian': gaussian,
        'median': median,
        'bilateral': bilateral
    }

def compute_region_snr(image, region_mask):
    """
    Compute SNR for a specific image region.
    
    Args:
        image: Input image array
        region_mask: Boolean mask for region of interest
    
    Returns:
        SNR value in dB
    """
    if not np.any(region_mask):
        return 0.0
    
    region_data = image[region_mask]
    signal = np.mean(region_data)
    noise = np.std(region_data)
    
    if noise == 0:
        return float('inf')
    return 20 * np.log10(signal / noise)

def print_snr_analysis(snr_values):
    """
    Print formatted SNR analysis results.
    
    Args:
        snr_values: Dictionary of SNR values by region
    """
    print("----------------------------------------")
    for tone, snr in snr_values.items():
        print(f"{tone:>10} tone: {snr:>8.2f} dB")

def main():
    """
    Execute image analysis pipeline.
    
    Stages:
    1. Load and analyze raw image
    2. Compare with denoised reference
    3. Apply and evaluate different denoising methods
    4. Analyze SNR in different tone regions
    """
    # Load raw image
    raw_path = 'Assignment2/data/raw/assignmentrawinput2.raw'
    print("Loading raw image...")
    raw_image = load_raw_image(raw_path)
    if raw_image is None:
        return

    # Load denoised image from as2
    denoised_path = 'Assignment2/data/output/as2_step4_denoised.png'
    print("Loading reference denoised image...")
    reference_denoised = cv2.imread(denoised_path, cv2.IMREAD_GRAYSCALE)
    if reference_denoised is None:
        print(f"Error: Could not load {denoised_path}")
        return
    
    # Apply denoising filters
    print("\nApplying denoising filters...")
    filtered_images = apply_denoising_filters(raw_image)
    
    # Analyze tone regions
    print("\nAnalyzing image tone regions...")
    tone_regions = analyze_tone_regions(raw_image)
    
    print("\nComputing SNR for different regions and methods...")
    
    print("\nSpatial Signal-to-Noise Ratio Analysis")
    print("==================================================\n")
    
    # Analyze raw image
    print("Original Raw SNR Analysis:")
    raw_snr = {tone: compute_region_snr(raw_image, mask) 
               for tone, mask in tone_regions.items()}
    print_snr_analysis(raw_snr)
    
    # Analyze reference denoised
    print("\nReference Denoised SNR Analysis:")
    ref_snr = {tone: compute_region_snr(reference_denoised, mask) 
               for tone, mask in tone_regions.items()}
    print_snr_analysis(ref_snr)
    
    # Analyze filtered images
    for filter_name, filtered_img in filtered_images.items():
        print(f"\n{filter_name.title()} Filter SNR Analysis:")
        filter_snr = {tone: compute_region_snr(filtered_img, mask) 
                     for tone, mask in tone_regions.items()}
        print_snr_analysis(filter_snr)
    
    # Display images with region markers
    display_images = []
    display_titles = []
    
    for filter_name, filtered_img in filtered_images.items():
        # Create a copy for visualization
        marked_img = filtered_img.copy()
        if len(marked_img.shape) == 2:
            marked_img = cv2.cvtColor(marked_img, cv2.COLOR_GRAY2BGR)
        
        # Mark analyzed regions
        colors = {
            'dark': (0, 0, 255),    # Red
            'medium': (0, 255, 0),  # Green
            'bright': (255, 0, 0)   # Blue
        }
        
        for tone, mask in tone_regions.items():
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                continue
            
            # Find center of largest continuous region
            center_y = (np.min(y_coords) + np.max(y_coords)) // 2
            center_x = (np.min(x_coords) + np.max(x_coords)) // 2
            
            # Define region around center
            half_size = 100 // 2
            y1 = max(0, center_y - half_size)
            y2 = min(marked_img.shape[0], center_y + half_size)
            x1 = max(0, center_x - half_size)
            x2 = min(marked_img.shape[1], center_x + half_size)
            
            cv2.rectangle(marked_img, (x1, y1), (x2, y2), colors[tone], 2)
        
        display_images.append(marked_img)
        display_titles.append(f"{filter_name.title()}\nSNR: {filter_snr['medium']:.1f}dB")
    
    # Display results
    plt.figure(figsize=(20, 10))
    for idx, (img, title) in enumerate(zip(display_images, display_titles), 1):
        plt.subplot(2, 3, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nSaving processed images...")
    cv2.imwrite('Assignment2/data/output/as2_1_gaussian.png', filtered_images['gaussian'])
    cv2.imwrite('Assignment2/data/output/as2_1_median.png', filtered_images['median'])
    cv2.imwrite('Assignment2/data/output/as2_1_bilateral.png', filtered_images['bilateral'])
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()

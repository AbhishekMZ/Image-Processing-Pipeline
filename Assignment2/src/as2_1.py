import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_raw_image(file_path, width=1920, height=1280):
    """Load raw image data."""
    try:
        raw_image = np.fromfile(file_path, dtype=np.uint16)
        raw_image = raw_image.reshape((height, width))
        # Convert 12-bit to 8-bit
        raw_image = (raw_image >> 4).astype(np.uint8)
        return raw_image
    except Exception as e:
        print(f"Error loading raw image: {e}")
        return None

def apply_denoising_filters(image):
    """Apply different denoising filters."""
    # Gaussian Filter (5x5 kernel)
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Median Filter (5x5 kernel)
    median = cv2.medianBlur(image, 5)
    
    # Bilateral Filter
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    return gaussian, median, bilateral

def compute_spatial_snr(image, region_coords):
    """
    Compute spatial SNR for a specific region.
    SNR = 20 * log10(signal/noise) where:
    - signal is the mean pixel value in the region
    - noise is the standard deviation of pixel values
    """
    y1, y2, x1, x2 = region_coords
    region = image[y1:y2, x1:x2]
    
    signal = np.mean(region)
    noise = np.std(region)
    
    if noise == 0:
        return float('inf')
    
    snr = 20 * np.log10(signal / noise)
    return snr

def analyze_tone_regions(image):
    """
    Analyze different tone regions in the image and return their coordinates.
    Returns regions representing dark, medium, and bright areas.
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Find percentiles for dark, medium, and bright regions
    total_pixels = gray.shape[0] * gray.shape[1]
    cumsum = np.cumsum(hist.flatten())
    
    # Find threshold values for different tones
    dark_threshold = np.searchsorted(cumsum, 0.15 * total_pixels)
    bright_threshold = np.searchsorted(cumsum, 0.85 * total_pixels)
    
    # Create masks for different tone regions
    dark_mask = gray < dark_threshold
    medium_mask = (gray >= dark_threshold) & (gray < bright_threshold)
    bright_mask = gray >= bright_threshold
    
    # Find regions for each tone
    def find_region(mask, size=100):
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
        
        # Find center of largest continuous region
        center_y = (np.min(y_coords) + np.max(y_coords)) // 2
        center_x = (np.min(x_coords) + np.max(x_coords)) // 2
        
        # Define region around center
        half_size = size // 2
        return (
            max(0, center_y - half_size),
            min(gray.shape[0], center_y + half_size),
            max(0, center_x - half_size),
            min(gray.shape[1], center_x + half_size)
        )
    
    regions = {
        'Dark': find_region(dark_mask),
        'Medium': find_region(medium_mask),
        'Bright': find_region(bright_mask)
    }
    
    return regions

def print_snr_analysis(method_name, snr_values):
    """Print SNR analysis results in a formatted way."""
    print(f"\n{method_name} SNR Analysis:")
    print("-" * 40)
    for tone, snr in snr_values.items():
        print(f"{tone:>10} tone: {snr:>8.2f} dB")

def main():
    # Load raw image
    raw_path = 'assignmentrawinput2.raw'
    print("Loading raw image...")
    raw_image = load_raw_image(raw_path)
    if raw_image is None:
        return

    # Load denoised image from as2
    denoised_path = 'as2_step4_denoised.png'
    print("Loading reference denoised image...")
    reference_denoised = cv2.imread(denoised_path, cv2.IMREAD_GRAYSCALE)
    if reference_denoised is None:
        print(f"Error: Could not load {denoised_path}")
        return

    # Apply different denoising filters
    print("Applying denoising filters...")
    gaussian_denoised, median_denoised, bilateral_denoised = apply_denoising_filters(raw_image)

    # Analyze tone regions
    print("\nAnalyzing image tone regions...")
    regions = analyze_tone_regions(raw_image)

    # Dictionary of all methods and their results
    methods = {
        'Original Raw': raw_image,
        'Reference Denoised': reference_denoised,
        'Gaussian Filter': gaussian_denoised,
        'Median Filter': median_denoised,
        'Bilateral Filter': bilateral_denoised
    }

    # Compute and display SNR for each method and region
    print("\nComputing SNR for different regions and methods...")
    print("\nSpatial Signal-to-Noise Ratio Analysis")
    print("=" * 50)
    
    all_snr_results = {}
    for method_name, img in methods.items():
        snr_values = {}
        for tone, coords in regions.items():
            if coords is not None:
                snr = compute_spatial_snr(img, coords)
                snr_values[tone] = snr
        
        print_snr_analysis(method_name, snr_values)
        all_snr_results[method_name] = snr_values

    # Display images with region markers
    display_images = []
    display_titles = []
    
    for method_name, img in methods.items():
        # Create a copy for visualization
        marked_img = img.copy()
        if len(marked_img.shape) == 2:
            marked_img = cv2.cvtColor(marked_img, cv2.COLOR_GRAY2BGR)
        
        # Mark analyzed regions
        colors = {
            'Dark': (0, 0, 255),    # Red
            'Medium': (0, 255, 0),  # Green
            'Bright': (255, 0, 0)   # Blue
        }
        
        for tone, coords in regions.items():
            if coords is not None:
                y1, y2, x1, x2 = coords
                cv2.rectangle(marked_img, (x1, y1), (x2, y2), colors[tone], 2)
        
        display_images.append(marked_img)
        display_titles.append(f"{method_name}\nSNR: {all_snr_results[method_name]['Medium']:.1f}dB")

    # Display results
    plt.figure(figsize=(20, 10))
    for idx, (img, title) in enumerate(zip(display_images, display_titles), 1):
        plt.subplot(2, 3, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Save results
    print("\nSaving processed images...")
    cv2.imwrite("gaussian_filtered.png", gaussian_denoised)
    cv2.imwrite("median_filtered.png", median_denoised)
    cv2.imwrite("bilateral_filtered.png", bilateral_denoised)

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()

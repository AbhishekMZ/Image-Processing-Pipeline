"""
Edge Enhancement Module
This module implements and analyzes different edge enhancement techniques,
including Laplacian filtering and unsharp masking.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(path):
    """
    Load and validate image file.
    
    Args:
        path: Path to image file
    
    Returns:
        Image array or None if error
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception(f"Failed to load image: {path}")
    return image

def apply_laplacian_filter(image, kernel_size=3, alpha=0.5):
    """
    Enhance edges using Laplacian filter.
    
    Args:
        image: Input image array
        kernel_size: Size of Laplacian kernel (default: 3)
        alpha: Edge enhancement strength (default: 0.5)
    
    Returns:
        Edge-enhanced image
    """
    # Convert to float for processing
    float_img = image.astype(np.float32)
    
    # Process each channel independently
    enhanced = np.zeros_like(float_img)
    for channel in range(3):
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(float_img[:,:,channel], cv2.CV_32F, ksize=kernel_size)
        # Add weighted edges to original
        enhanced[:,:,channel] = float_img[:,:,channel] + alpha * laplacian
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def apply_unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.5):
    """
    Enhance edges using unsharp masking.
    
    Args:
        image: Input image array
        kernel_size: Gaussian blur kernel size (default: 5)
        sigma: Gaussian blur sigma (default: 1.0)
        amount: Enhancement strength (default: 1.5)
    
    Returns:
        Edge-enhanced image
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Calculate unsharp mask
    mask = image.astype(np.float32) - blurred.astype(np.float32)
    
    # Add weighted mask to original
    sharpened = image.astype(np.float32) + amount * mask
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def compute_edge_strength(image):
    """
    Compute edge strength map using Sobel operators.
    
    Args:
        image: Input image array
    
    Returns:
        Edge strength map
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255 range
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def display_images(images, titles):
    """
    Display multiple images in a grid layout.
    
    Args:
        images: List of image arrays
        titles: List of corresponding titles
    """
    num_images = len(images)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 5*rows))
    
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, idx + 1)
        if len(image.shape) == 2:  # Grayscale
            plt.imshow(image, cmap='gray')
        else:  # RGB
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_edge_strength(edge_map, method_name):
    """
    Analyze and print edge strength statistics.
    
    Args:
        edge_map: Edge strength map array
        method_name: Name of the enhancement method
    """
    print(f"\nEdge Strength Analysis for {method_name}:")
    print(f"Mean Strength: {np.mean(edge_map):.2f}")
    print(f"Median Strength: {np.median(edge_map):.2f}")
    print(f"Standard Deviation: {np.std(edge_map):.2f}")
    print(f"Maximum Strength: {np.max(edge_map):.2f}")

def main():
    """
    Execute edge enhancement pipeline.
    
    Pipeline stages:
    1. Load denoised image
    2. Apply Laplacian enhancement
    3. Apply unsharp mask
    4. Compute edge strength maps
    5. Analyze and compare results
    """
    try:
        # Load the denoised image
        input_image = load_image('Assignment2/data/output/as2_step4_denoised.png')
        
        print("Applying edge enhancement filters...")
        
        # Apply enhancement methods
        laplacian_result = apply_laplacian_filter(input_image)
        unsharp_result = apply_unsharp_mask(input_image)
        
        # Save enhanced images
        cv2.imwrite('Assignment2/data/output/as2_2_laplacian.png', laplacian_result)
        cv2.imwrite('Assignment2/data/output/as2_2_unsharp.png', unsharp_result)
        
        # Display results
        print("\nDisplaying enhancement results...")
        display_images(
            [input_image, laplacian_result, unsharp_result],
            ['Original Denoised', 'Laplacian Enhancement', 'Unsharp Mask Enhancement']
        )
        
        # Compute edge strength maps
        print("\nComputing edge strength...")
        edge_strength_original = compute_edge_strength(input_image)
        edge_strength_laplacian = compute_edge_strength(laplacian_result)
        edge_strength_unsharp = compute_edge_strength(unsharp_result)
        
        # Display edge maps
        print("\nDisplaying edge strength maps...")
        display_images(
            [edge_strength_original, edge_strength_laplacian, edge_strength_unsharp],
            ['Original Edge Strength', 'Laplacian Edge Strength', 'Unsharp Mask Edge Strength']
        )
        
        # Analyze results
        analyze_edge_strength(edge_strength_original, "Original")
        analyze_edge_strength(edge_strength_laplacian, "Laplacian")
        analyze_edge_strength(edge_strength_unsharp, "Unsharp Mask")
        
        # Save edge maps
        print("\nSaving edge maps...")
        cv2.imwrite('Assignment2/data/output/as2_2_edge_strength_original.png', edge_strength_original)
        cv2.imwrite('Assignment2/data/output/as2_2_edge_strength_laplacian.png', edge_strength_laplacian)
        cv2.imwrite('Assignment2/data/output/as2_2_edge_strength_unsharp.png', edge_strength_unsharp)
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    main()

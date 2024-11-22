import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(file_path):
    """Load image in grayscale."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception(f"Error loading image: {file_path}")
    return img

def apply_laplacian_filter(image, kernel_size=3):
    """Apply Laplacian filter for edge detection."""
    # Convert to float32 for better precision
    img_float = image.astype(np.float32) / 255.0
    
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=kernel_size)
    
    # Normalize and convert back to uint8
    laplacian = np.abs(laplacian)
    laplacian_normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    
    return laplacian_normalized.astype(np.uint8)

def apply_unsharp_mask(image, kernel_size=(5,5), sigma=1.0, amount=1.5, threshold=0):
    """
    Apply unsharp mask filter for edge enhancement.
    amount: Weight of enhancement (1.5 = 150% enhancement)
    threshold: Minimum brightness difference for enhancement
    """
    # Create the gaussian blur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Calculate the unsharp mask
    unsharp_mask = cv2.addWeighted(
        image.astype(np.float32), 1.0 + amount,
        blurred.astype(np.float32), -amount, 
        threshold
    )
    
    # Clip values to valid range
    return np.clip(unsharp_mask, 0, 255).astype(np.uint8)

def compute_edge_strength(image):
    """
    Compute edge strength using Sobel gradient approach.
    Returns both magnitude and normalized edge strength.
    """
    # Calculate gradients using Sobel
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize for visualization
    edge_strength = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    return edge_strength.astype(np.uint8)

def display_images(images, titles, figsize=(15, 5)):
    """Display multiple images side by side with titles."""
    plt.figure(figsize=figsize)
    
    for idx, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), idx)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_edge_strength(edge_strength, method_name):
    """Analyze and print edge strength statistics."""
    mean_strength = np.mean(edge_strength)
    median_strength = np.median(edge_strength)
    std_dev = np.std(edge_strength)
    max_strength = np.max(edge_strength)
    
    print(f"\nEdge Strength Analysis for {method_name}:")
    print(f"Mean Strength: {mean_strength:.2f}")
    print(f"Median Strength: {median_strength:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Maximum Strength: {max_strength:.2f}")

def main():
    try:
        # Load the denoised image
        input_image = load_image('as2_step4_denoised.png')
        
        print("Applying edge enhancement filters...")
        
        # Apply Laplacian filter
        laplacian_result = apply_laplacian_filter(input_image)
        
        # Apply Unsharp Mask
        unsharp_result = apply_unsharp_mask(input_image)
        
        # Display original and enhanced images
        print("\nDisplaying enhancement results...")
        display_images(
            [input_image, laplacian_result, unsharp_result],
            ['Original Denoised', 'Laplacian Enhancement', 'Unsharp Mask Enhancement']
        )
        
        # Compute edge strength for each version
        print("\nComputing edge strength...")
        edge_strength_original = compute_edge_strength(input_image)
        edge_strength_laplacian = compute_edge_strength(laplacian_result)
        edge_strength_unsharp = compute_edge_strength(unsharp_result)
        
        # Display edge strength maps
        print("\nDisplaying edge strength maps...")
        display_images(
            [edge_strength_original, edge_strength_laplacian, edge_strength_unsharp],
            ['Original Edge Strength', 'Laplacian Edge Strength', 'Unsharp Mask Edge Strength']
        )
        
        # Analyze edge strength for each method
        analyze_edge_strength(edge_strength_original, "Original")
        analyze_edge_strength(edge_strength_laplacian, "Laplacian")
        analyze_edge_strength(edge_strength_unsharp, "Unsharp Mask")
        
        # Save results
        print("\nSaving enhanced images and edge maps...")
        cv2.imwrite("laplacian_enhanced.png", laplacian_result)
        cv2.imwrite("unsharp_mask_enhanced.png", unsharp_result)
        cv2.imwrite("edge_strength_original.png", edge_strength_original)
        cv2.imwrite("edge_strength_laplacian.png", edge_strength_laplacian)
        cv2.imwrite("edge_strength_unsharp.png", edge_strength_unsharp)
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    main()

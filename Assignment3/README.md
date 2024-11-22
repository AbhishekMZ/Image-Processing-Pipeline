# Assignment 3: HDR Image Processing

## Overview
High Dynamic Range (HDR) image processing implementation, focusing on exposure fusion and tone mapping.

## Implementation Guide

### 1. Image Loading and Alignment
```python
def load_exposure_stack(image_paths):
    """
    Load and align multiple exposure images
    """
    # Load images
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
    images = [img.astype(np.float32) / 255 for img in images]
    
    # Align images
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
    
    return images

def get_exposure_times(image_paths):
    """
    Extract exposure times from image metadata
    """
    exposure_times = []
    for path in image_paths:
        img = Image.open(path)
        exif = img._getexif()
        exposure_time = float(exif[33434])  # EXIF tag for exposure time
        exposure_times.append(exposure_time)
    return np.array(exposure_times)
```

### 2. Exposure Fusion
```python
def exposure_fusion(images, weights=None):
    """
    Merge multiple exposures into a single well-exposed image
    """
    if weights is None:
        weights = {
            'contrast': 1.0,
            'saturation': 1.0,
            'well_exposed': 1.0
        }
    
    # Calculate weights for each image
    weight_maps = []
    for img in images:
        w = compute_weights(
            img,
            contrast_weight=weights['contrast'],
            saturation_weight=weights['saturation'],
            exposure_weight=weights['well_exposed']
        )
        weight_maps.append(w)
    
    # Normalize weights
    weight_sum = sum(weight_maps)
    normalized_weights = [w / weight_sum for w in weight_maps]
    
    # Merge images
    result = sum(img * w[:,:,np.newaxis] 
                for img, w in zip(images, normalized_weights))
    
    return np.clip(result, 0, 1)
```

### 3. HDR Reconstruction
```python
def reconstruct_hdr(images, exposure_times):
    """
    Reconstruct HDR radiance map from multiple exposures
    """
    # Create HDR merger
    merger = cv2.createMergeDebevec()
    
    # Estimate camera response function
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, exposure_times)
    
    # Merge into HDR
    hdr = merger.process(images, exposure_times, response)
    return hdr
```

### 4. Tone Mapping
```python
def tone_map_hdr(hdr_image, method='reinhard'):
    """
    Apply tone mapping to HDR image
    """
    if method == 'reinhard':
        tonemapper = cv2.createTonemapReinhard(
            gamma=1.0,
            intensity=0.0,
            light_adapt=0.8,
            color_adapt=0.0
        )
    elif method == 'mantiuk':
        tonemapper = cv2.createTonemapMantiuk(
            gamma=1.0,
            scale=0.7,
            saturation=1.0
        )
    elif method == 'drago':
        tonemapper = cv2.createTonemapDrago(
            gamma=1.0,
            saturation=1.0,
            bias=0.85
        )
    
    return tonemapper.process(hdr_image)
```

## Components

### Source Files
- `as3.py`: HDR processing implementation
  - Exposure fusion
  - HDR reconstruction
  - Tone mapping operators
  - Quality assessment

### Data Structure
```
Assignment3/
├── src/
│   └── as3.py
├── data/
│   ├── hdr/
│   │   ├── underexposed.jpg
│   │   ├── midexposed.jpg
│   │   └── overexposed.jpg
│   └── output/
└── tests/
```

## Advanced Features

### 1. Exposure Fusion
- Contrast weight computation
- Saturation analysis
- Well-exposedness measurement
- Multi-resolution blending

### 2. HDR Reconstruction
- Camera response calibration
- Radiance map estimation
- Ghost removal
- Alignment refinement

### 3. Tone Mapping
- Local/global operators
- Detail preservation
- Color management
- Artifact prevention

## Usage Instructions

### Basic Processing
```python
# Run HDR processing pipeline
python src/as3.py

# Parameters:
HDR_PARAMS = {
    'exposure_fusion': {
        'contrast_weight': 1.0,
        'saturation_weight': 1.0,
        'exposure_weight': 1.0
    },
    'tone_mapping': {
        'method': 'reinhard',
        'gamma': 1.0,
        'intensity': 0.0,
        'light_adapt': 0.8
    }
}
```

### Advanced Options
```python
# HDR reconstruction parameters
RECONSTRUCTION_PARAMS = {
    'calibration_samples': 70,
    'response_smoothness': 10.0,
    'ghost_threshold': 0.1
}

# Tone mapping parameters
TONE_MAPPING_PARAMS = {
    'reinhard': {
        'gamma': 1.0,
        'intensity': 0.0,
        'light_adapt': 0.8,
        'color_adapt': 0.0
    },
    'mantiuk': {
        'gamma': 1.0,
        'scale': 0.7,
        'saturation': 1.0
    },
    'drago': {
        'gamma': 1.0,
        'saturation': 1.0,
        'bias': 0.85
    }
}
```

## Performance Optimization

### 1. Memory Management
- Progressive loading
- Efficient HDR storage
- Intermediate cleanup
- Resource monitoring

### 2. Processing Speed
- Multi-threaded alignment
- GPU acceleration
- Optimized blending
- Selective processing

### 3. Quality Control
- Alignment verification
- Ghost detection
- Artifact monitoring
- Dynamic range validation

## Troubleshooting Guide

### 1. Alignment Issues
- Check exposure differences
- Verify motion content
- Adjust alignment parameters
- Review image order

### 2. Fusion Artifacts
- Check weight parameters
- Verify exposure spacing
- Review blending masks
- Adjust ghost removal

### 3. Tone Mapping Problems
- Verify HDR range
- Check operator settings
- Review color preservation
- Adjust local contrast

## Advanced Topics

### 1. Exposure Analysis
- Optimal exposure selection
- Dynamic range measurement
- Histogram analysis
- Saturation detection

### 2. Quality Assessment
- Dynamic range metrics
- Detail preservation
- Color accuracy
- Natural appearance

### 3. Advanced Techniques
- Deep learning integration
- Adaptive tone mapping
- Content-aware fusion
- Real-time processing

## References

### 1. HDR Imaging
- High Dynamic Range Imaging (Reinhard et al.)
- Digital HDR Photography
- Tone Mapping Techniques

### 2. Computational Photography
- Multiple Exposure Techniques
- Image Fusion Methods
- Camera Response Functions

### 3. Implementation
- OpenCV HDR Module
- Efficient Image Processing
- Color Science in HDR

# Advanced Image Signal Processing Pipeline

A sophisticated image processing pipeline implementing AI-inspired techniques for raw image enhancement, including advanced demosaicing, denoising, and color correction.

## Features

- **Raw Image Processing**: Handles 12-bit Bayer pattern raw images (1920x1280)
- **Advanced Denoising**: Hybrid AI-inspired denoising algorithm
- **Intelligent Edge Enhancement**: Adaptive Laplacian filtering
- **Color Correction**: Advanced white balance using Gray World algorithm
- **Progressive Enhancement**: Multi-stage processing pipeline
- **Visualization**: Step-by-step output visualization

## Pipeline Stages

1. **Raw Bayer Loading**: 12-bit to 8-bit conversion
2. **Demosaicing**: Color reconstruction from Bayer pattern
3. **White Balance**: Gray World algorithm implementation
4. **Hybrid Denoising**:
   - Adaptive weight calculation
   - Edge-aware processing
   - Multi-iteration refinement
5. **Edge Enhancement**: Laplacian filter with detail preservation
6. **Contrast Adjustment**: Dynamic range optimization
7. **Brightness Control**: HSV-space adjustment

## Technical Implementation

### AI-Inspired Techniques

1. **Hybrid Denoising Filter**:
   - Spatial and intensity-based weight calculation
   - Edge preservation using gradient magnitude
   - Iterative refinement process

2. **Adaptive Processing**:
   - Dynamic parameter adjustment
   - Statistical learning from image content
   - Multi-scale analysis

### Key Algorithms

- **White Balance**: Gray World algorithm
- **Edge Detection**: Gradient-based analysis
- **Denoising**: Hybrid filter combining spatial and edge awareness
- **Color Enhancement**: HSV space transformation

## Requirements

```
numpy
opencv-python
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Image-Processing-Pipeline.git
cd Image-Processing-Pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
python as2.py
```

This will:
1. Load the raw image
2. Process through all pipeline stages
3. Save intermediate outputs
4. Display side-by-side comparisons

### Output Files

- `as2_step1_raw_bayer.png`: Raw Bayer pattern
- `as2_step2_demosaiced.png`: After color reconstruction
- `as2_step3_white_balanced.png`: After white balance
- `as2_step4_denoised.png`: After hybrid denoising
- `as2_step5_edge_enhanced.png`: After Laplacian enhancement
- `as2_step6_contrast_stretched.png`: After contrast adjustment
- `as2_step7_final.png`: Final output with brightness adjustment

## Implementation Details

### File Structure

- `as1.py`: Original comprehensive pipeline
- `as2.py`: Enhanced pipeline with visualization
- `as3.py`: Simplified direct processing
- `as4.py`: Advanced hybrid denoising implementation

### Key Parameters

- Gamma correction: 0.9
- Kernel sizes: 3x3 to 5x5
- Denoising iterations: 3
- Edge enhancement alpha: 0.4
- Brightness factor: 1.2

## Performance Considerations

- Optimized for 1920x1280 raw images
- Memory-efficient processing
- Progressive quality improvement
- Balanced processing time vs. quality

## Future Improvements

- GPU acceleration support
- Machine learning-based parameter optimization
- Additional raw format support
- Real-time processing capabilities
- Advanced noise reduction techniques

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV library for core image processing functions
- NumPy for efficient array operations
- Matplotlib for visualization capabilities

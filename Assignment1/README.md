# Assignment 1: Raw Image Processing Pipeline

## Quick Start Guide

### 1. Initial Setup
```bash
# Install required dependencies
pip install numpy opencv-python PyQt5 matplotlib pillow
```

### 2. Running the Application
```bash
# Navigate to project directory
cd path/to/Assignment1

# Run the GUI application
python src/as1_gui.py
```

## Detailed Usage Guide

### 1. GUI Controls and Operations

#### Loading Images
- Click "Load Raw File" button
- Navigate to `data/raw/`
- Select either:
  * `assignmentrawinput1.raw`
  * `assignmentrawinput2.raw`

#### Parameter Adjustment Controls

1. **White Balance**
   - R Gain (0-4.0): Red channel intensity
   - G Gain (0-4.0): Green channel intensity
   - B Gain (0-4.0): Blue channel intensity
   ```python
   # Default values
   r_gain = 1.8
   g_gain = 1.0
   b_gain = 1.4
   ```

2. **Gamma Correction**
   - Range: 0.1-2.0
   - Default: 0.45
   - Higher values → brighter image
   - Lower values → darker image

3. **Sharpening**
   - Amount (0-2.0): Intensity of sharpening
   - Radius (0-2.0): Area of effect
   - Threshold (0-1.0): Edge detection sensitivity
   ```python
   # Default values
   amount = 0.5
   radius = 1.0
   threshold = 0.1
   ```

4. **Denoising**
   - Strength (0-100): Overall noise reduction
   - Color Strength (0-100): Color noise reduction
   ```python
   # Default values
   strength = 10
   color_strength = 10
   ```

### 2. Processing Pipeline

#### Output Files Generated
```
data/output/
├── demosaiced_image.png      # After Bayer conversion
├── white_balanced_image.png  # After white balance
├── denoised_image.png       # After noise reduction
├── gamma_corrected_image.png # After gamma correction
├── sharpened_image.png      # After sharpening
└── final_enhanced_image.png  # Final output
```

#### Processing Steps
1. **Raw Loading**
   ```python
   def load_raw_image(file_path):
       raw_data = np.fromfile(file_path, dtype=np.uint8)
       return raw_data.reshape((1280, 1920))
   ```

2. **Demosaicing**
   - Converts Bayer pattern to RGB
   - Uses OpenCV's demosaicing algorithm

3. **White Balance**
   ```python
   def apply_white_balance(image, gains):
       balanced = image.copy().astype(float)
       balanced[0::2, 0::2] *= gains[0]  # R
       balanced[0::2, 1::2] *= gains[1]  # G1
       balanced[1::2, 0::2] *= gains[1]  # G2
       balanced[1::2, 1::2] *= gains[2]  # B
       return np.clip(balanced, 0, 255)
   ```

4. **Gamma Correction**
   ```python
   def apply_gamma(image, gamma=0.45):
       return np.power(image / 255.0, gamma) * 255
   ```

5. **Denoising**
   - Reduces image noise while preserving edges
   - Separate control for color noise

6. **Sharpening**
   ```python
   def apply_sharpening(image, amount=0.5, radius=1.0, threshold=0.1):
       blurred = cv2.GaussianBlur(image, (0, 0), radius)
       mask = image - blurred
       sharp = image + amount * mask
       return np.clip(sharp, 0, 255)
   ```

### 3. Recommended Workflow

1. **Start with White Balance**
   - Begin with defaults: R=1.8, G=1.0, B=1.4
   - Adjust until colors appear natural
   - Check gray/white areas for color cast

2. **Adjust Gamma**
   - Start at 0.45
   - Increase for brighter image
   - Decrease for darker image
   - Monitor highlight and shadow detail

3. **Apply Denoising**
   - Start with low values (10-20)
   - Increase if image appears noisy
   - Watch for detail preservation
   - Adjust color noise separately if needed

4. **Fine-tune Sharpening**
   - Begin with defaults:
     * Amount: 0.5
     * Radius: 1.0
     * Threshold: 0.1
   - Adjust amount first
   - Fine-tune radius for edge control
   - Use threshold to prevent noise enhancement

### 4. Troubleshooting

#### GUI Issues
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify PyQt5
python -c "import PyQt5; print(PyQt5.__version__)"

# Check OpenCV
python -c "import cv2; print(cv2.__version__)"
```

#### Memory Issues
```python
# Clear Python cache if needed
import sys
sys.modules.clear()
```

#### Directory Structure
```bash
# Verify directories
dir Assignment1
dir Assignment1\src
dir Assignment1\data\raw
dir Assignment1\data\output
```

### 5. Project Structure
```
Assignment1/
├── src/
│   ├── as1.py           # Core processing
│   └── as1_gui.py       # GUI implementation
├── data/
│   ├── raw/             # Input images
│   └── output/          # Processed results
└── requirements.txt     # Dependencies
```

## Dependencies
```
numpy>=1.19.2
opencv-python>=4.5.1
PyQt5>=5.15.4
matplotlib>=3.3.4
pillow>=8.1.0
```

## Performance Notes

### Memory Management
- Large images processed in chunks
- Temporary files cleaned automatically
- Memory released after processing

### Processing Speed
- Optimized for common image sizes
- Vectorized operations where possible
- Progress feedback during processing

### Quality Considerations
- 12-bit raw data handling
- Proper color space management
- Edge-aware processing
- Artifact prevention

## Support and Resources

### Documentation
- OpenCV: https://docs.opencv.org/
- PyQt5: https://www.riverbankcomputing.com/static/Docs/PyQt5/
- NumPy: https://numpy.org/doc/

### Common Issues
1. Raw File Loading
   - Verify file format
   - Check file permissions
   - Confirm dimensions

2. Processing
   - Monitor parameter ranges
   - Check available memory
   - Verify color spaces

3. Display
   - Update graphics drivers
   - Check screen resolution
   - Verify color calibration

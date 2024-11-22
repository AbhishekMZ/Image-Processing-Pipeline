import streamlit as st
import numpy as np
import cv2
from as1 import (
    apply_gamma_correction,
    enhance_colors_and_correct_pink,
    apply_contrast_stretching
)

# Set page config
st.set_page_config(
    page_title="Raw Image Processing",
    layout="wide"
)

# Title and description
st.title("Raw Image Processing Pipeline")
st.sidebar.header("Upload and Parameters")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload RAW Image", type=['raw'])

# Parameters in sidebar
width = st.sidebar.number_input("Image Width", value=1920, step=1)
height = st.sidebar.number_input("Image Height", value=1280, step=1)

# White balance parameters
st.sidebar.subheader("White Balance")
wb_r = st.sidebar.slider("Red Gain", 0.5, 2.5, 1.8, 0.1)
wb_g = st.sidebar.slider("Green Gain", 0.5, 2.5, 1.0, 0.1)
wb_b = st.sidebar.slider("Blue Gain", 0.5, 2.5, 1.4, 0.1)

# Gamma correction
st.sidebar.subheader("Gamma Correction")
gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 0.9, 0.1)

# Enhancement parameters
st.sidebar.subheader("Image Enhancement")
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.2, 0.1)
brightness = st.sidebar.slider("Brightness", -50, 50, 20, 1)
sharpness = st.sidebar.slider("Sharpness", 0.0, 2.0, 1.5, 0.1)
saturation = st.sidebar.slider("Saturation", 0.5, 3.0, 1.7, 0.1)
hue_shift = st.sidebar.slider("Hue Shift", -50, 50, -5, 1)

if uploaded_file is not None:
    # Load and process RAW image
    try:
        # Read RAW file
        raw_data = np.frombuffer(uploaded_file.read(), dtype=np.uint16)
        raw_data = raw_data.reshape((height, width))
        raw_data = (raw_data >> 4).astype(np.uint8)

        # Display original RAW data
        st.subheader("Original RAW Data")
        st.image(raw_data, caption="Original RAW Data", use_column_width=True)

        # Demosaicing
        bayer_pattern = cv2.COLOR_BAYER_GB2BGR
        rgb_image = cv2.cvtColor(raw_data, bayer_pattern)
        
        # Create lookup table for gamma
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
        demosaiced = apply_gamma_correction(rgb_image, lookup_table)
        
        st.subheader("Demosaiced Image")
        st.image(demosaiced, channels="BGR", caption="Demosaiced Image", use_column_width=True)

        # White Balance
        balanced = demosaiced.copy()
        balanced[:, :, 2] = np.clip(balanced[:, :, 2] * wb_r, 0, 255)  # Red
        balanced[:, :, 1] = np.clip(balanced[:, :, 1] * wb_g, 0, 255)  # Green
        balanced[:, :, 0] = np.clip(balanced[:, :, 0] * wb_b, 0, 255)  # Blue
        
        st.subheader("White Balanced Image")
        st.image(balanced, channels="BGR", caption="White Balanced", use_column_width=True)

        # Denoising
        denoised = cv2.GaussianBlur(balanced, (5, 5), 0)
        
        st.subheader("Denoised Image")
        st.image(denoised, channels="BGR", caption="Denoised", use_column_width=True)

        # Contrast and Brightness
        adjusted = cv2.convertScaleAbs(denoised, alpha=contrast, beta=brightness)
        
        st.subheader("Contrast Adjusted Image")
        st.image(adjusted, channels="BGR", caption="Contrast Adjusted", use_column_width=True)

        # Sharpening
        blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
        sharpened = cv2.addWeighted(adjusted, sharpness, blurred, 1-sharpness, 0)
        
        st.subheader("Sharpened Image")
        st.image(sharpened, channels="BGR", caption="Sharpened", use_column_width=True)

        # Contrast stretching
        stretched = apply_contrast_stretching(sharpened)
        
        st.subheader("Contrast Stretched Image")
        st.image(stretched, channels="BGR", caption="Contrast Stretched", use_column_width=True)

        # Final color enhancement
        final = enhance_colors_and_correct_pink(stretched, saturation, hue_shift)
        
        st.subheader("Final Enhanced Image")
        st.image(final, channels="BGR", caption="Final Result", use_column_width=True)

        # Add download button for final image
        if st.button("Save Final Image"):
            cv2.imwrite("final_enhanced_image.png", final)
            st.success("Image saved as 'final_enhanced_image.png'")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload a RAW image file to begin processing.")
    st.markdown("""
    ### Instructions:
    1. Use the sidebar to upload your RAW image file
    2. Adjust the parameters to enhance your image:
        - White Balance: Adjust color temperature
        - Gamma: Control image brightness and contrast
        - Enhancement: Fine-tune final image appearance
    3. View the step-by-step processing results
    4. Save the final enhanced image
    """)
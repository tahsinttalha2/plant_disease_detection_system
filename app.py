"""
Plant Disease Detection - Streamlit App
Upload a plant image to detect diseases using a trained CNN model
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f9f0;
        border-left: 5px solid #2e7d32;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #2e7d32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .confidence-low {
        color: #d32f2f;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load the trained model (cached to avoid reloading)"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_class_names(class_names_file='class_names.json'):
    """Load class names from JSON file"""
    if os.path.exists(class_names_file):
        with open(class_names_file, 'r') as f:
            return json.load(f)
    else:
        # Return placeholder names if file doesn't exist
        return [f"Disease_Class_{i}" for i in range(38)]


def preprocess_image(image):
    """
    Preprocess the uploaded image for model prediction
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array
    """
    # Resize to 128x128 (same as training)
    img = image.resize((128, 128))
    
    # Convert to RGB (in case it's grayscale or has alpha channel)
    img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def get_confidence_color(confidence):
    """Return CSS class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def predict_disease(model, image, class_names, top_k=5):
    """
    Predict plant disease from image
    
    Args:
        model: Loaded TensorFlow model
        image: PIL Image object
        class_names: List of disease class names
        top_k: Number of top predictions to return
        
    Returns:
        List of tuples (class_name, probability)
    """
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get probabilities
    probabilities = predictions[0]
    
    # Get top k predictions
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        class_name = class_names[idx]
        probability = float(probabilities[idx])
        results.append((class_name, probability))
    
    return results


def main():
    # Header
    st.markdown('<p class="main-header">🌱 Plant Disease Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to identify plant diseases using AI</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_path = st.text_input(
            "Model Path",
            value="plant_disease_model.h5",
            help="Path to your trained model file (.h5 or .keras)"
        )
        
        top_k = st.slider(
            "Number of predictions to show",
            min_value=1,
            max_value=10,
            value=5,
            help="Show top N predictions"
        )
        
        st.markdown("---")
        st.markdown("### 📋 About")
        st.info(
            "This app uses a Convolutional Neural Network (CNN) "
            "trained on 38 different plant disease categories to identify "
            "diseases from plant leaf images."
        )
    
    # Load model
    model = load_model(model_path)
    
    if model is None:
        st.error("❌ Failed to load model. Please check the model path in the sidebar.")
        st.stop()
    
    # Load class names
    class_names = load_class_names()
    
    st.success(f"✅ Model loaded successfully! Ready to classify {len(class_names)} disease types.")
    
    # File uploader with multiple input options
    st.markdown("### 📤 Upload Plant Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file or drag and drop",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Supported formats: JPG, JPEG, PNG, WEBP"
    )
    
    # Alternative: Paste from clipboard (using camera input as proxy)
    col1, col2 = st.columns(2)
    with col1:
        camera_image = st.camera_input("📷 Or take a photo")
    
    # Process the uploaded image
    image_to_process = None
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
    elif camera_image is not None:
        image_to_process = Image.open(camera_image)
    
    if image_to_process is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Uploaded Image")
            st.image(image_to_process, use_container_width=True)
        
        with col2:
            st.markdown("### 🔍 Analysis")
            
            # Add a predict button
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Get predictions
                    predictions = predict_disease(model, image_to_process, class_names, top_k=top_k)
                    
                    # Display top prediction prominently
                    top_disease, top_confidence = predictions[0]
                    
                    st.markdown("#### 🎯 Top Prediction")
                    confidence_class = get_confidence_color(top_confidence)
                    st.markdown(
                        f'<div class="prediction-box">'
                        f'<h3 style="margin:0; color:#2e7d32;">{top_disease}</h3>'
                        f'<p class="{confidence_class}" style="font-size:1.5rem; margin:0.5rem 0 0 0;">'
                        f'{top_confidence*100:.2f}% confidence</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display all predictions
                    st.markdown("#### 📊 All Predictions")
                    
                    for i, (disease, confidence) in enumerate(predictions, 1):
                        # Create progress bar for each prediction
                        col_rank, col_name, col_conf = st.columns([0.5, 3, 1.5])
                        
                        with col_rank:
                            st.markdown(f"**{i}.**")
                        
                        with col_name:
                            st.markdown(f"**{disease}**")
                        
                        with col_conf:
                            confidence_class = get_confidence_color(confidence)
                            st.markdown(f'<span class="{confidence_class}">{confidence*100:.1f}%</span>', unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(confidence)
                        
                        if i < len(predictions):
                            st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Additional information
                    st.markdown("---")
                    st.info(
                        "💡 **Tip:** Higher confidence scores indicate more certain predictions. "
                        "If confidence is low, try uploading a clearer image or one taken in better lighting."
                    )
    
    else:
        # Instructions when no image is uploaded
        st.info(
            "👆 Please upload an image of a plant leaf using one of the methods above. "
            "You can:\n"
            "- **Upload** a file from your computer\n"
            "- **Drag and drop** an image into the upload box\n"
            "- **Take a photo** using your camera\n"
        )
        
        # Show example of what to expect
        with st.expander("ℹ️ What kind of images work best?"):
            st.markdown("""
            **For best results:**
            - Use clear, well-lit images of plant leaves
            - Ensure the diseased area is visible
            - Avoid blurry or low-resolution images
            - Single leaf images work better than full plant photos
            
            **Supported plants:** This model can identify diseases across 38 different categories.
            """)


if __name__ == "__main__":
    main()
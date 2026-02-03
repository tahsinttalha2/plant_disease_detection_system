# 🌱 Plant Disease Detection - Streamlit App

A web application for detecting plant diseases using a trained Convolutional Neural Network (CNN) model.

## 📋 Prerequisites

- Python 3.8+
- TensorFlow
- Streamlit
- PIL/Pillow
- NumPy

## 🚀 Quick Start

### Step 1: Extract Your Class Names

First, you need to extract the 38 disease class names from your training directory:

```bash
python extract_class_names.py
```

This will:
- Read your training directory structure
- Extract all 38 disease category names
- Save them to `class_names.json`

**Important:** Make sure your `train` directory is in the same location, or provide the correct path when prompted.

### Step 2: Prepare Your Model

Make sure your trained model file is ready:
- `plant_disease_model.h5` (your model file)

Place it in the same directory as `app.py`, or note its location.

### Step 3: Run the Streamlit App

```bash
streamlit run app.py
```

This will:
- Start a local web server
- Open your default browser automatically
- Display the Plant Disease Detection interface

## 📁 File Structure

```
your-project/
├── app.py                          # Main Streamlit application
├── extract_class_names.py          # Script to extract class names
├── class_names.json                # Generated file with disease names
├── plant_disease_model.h5          # Your trained model
└── train/                          # Your training directory (optional)
    ├── disease_1/
    ├── disease_2/
    └── ...
```

## 🎯 How to Use the App

1. **Open the app** in your browser (automatically opens after running streamlit)

2. **Upload an image** using one of three methods:
   - Click "Browse files" to upload from your computer
   - Drag and drop an image into the upload box
   - Use "Take a photo" to capture from your camera

3. **Click "Analyze Image"** to get predictions

4. **View results:**
   - Top prediction with confidence score
   - Top 5 (or more) predictions ranked by confidence
   - Visual progress bars for each prediction

## ⚙️ Configuration

You can adjust settings in the sidebar:
- **Model Path:** Change if your model is in a different location
- **Number of predictions:** Show more or fewer predictions (1-10)

## 📸 Image Requirements

For best results:
- Use clear, well-lit images
- Focus on plant leaves showing disease symptoms
- Avoid blurry or low-resolution images
- Single leaf images work better than full plant photos

## 🔧 Troubleshooting

### "Failed to load model" error
- Check that the model path is correct
- Ensure the model file exists
- Try using the full absolute path

### "No class_names.json found" warning
- Run `extract_class_names.py` first
- The app will use placeholder names if this file is missing

### App runs slowly
- First prediction is slower (model loading)
- Subsequent predictions are faster (model is cached)
- Large images take longer to process

## 💡 Tips

- The app caches the model, so it only loads once
- You can upload multiple images without restarting
- Confidence scores above 70% are generally reliable
- Lower confidence might indicate:
  - Poor image quality
  - Disease not in training data
  - Ambiguous symptoms

## 🛠️ Advanced Usage

### Using a different model

If you have a different model file:

```bash
streamlit run app.py
```

Then update the "Model Path" in the sidebar.

### Customizing the number of predictions

Change the `top_k` slider in the sidebar (1-10 predictions).

## 📊 Model Information

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 128x128 RGB images
- **Classes:** 38 different plant disease categories
- **Training Data:** 70,295 training images, 17,117 validation images

## 🤝 Support

If you encounter any issues:
1. Check that all files are in the correct location
2. Verify your model file is valid
3. Ensure all dependencies are installed
4. Run `extract_class_names.py` if you haven't already

## 📝 Notes

- This app is designed for the plant disease model trained in your notebook
- Make sure image preprocessing matches your training setup (128x128, RGB)
- The app works offline once the model is loaded
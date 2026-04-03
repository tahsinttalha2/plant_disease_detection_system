# Plant Disease Detection System

[![Open in Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://plant-disease-detection-tahsinttalha.streamlit.app/)

This project features a CNN model trained on the PlantVillage dataset, capable of classifying 38 plant disease categories from leaf images. The purpose of this project was to explore real-world applications of deep learning in agriculture. A Streamlit app was built to create an interactive environment where users can upload plant leaf images and receive ranked disease predictions with confidence scores.

## Tech Stack
- Python
- TensorFlow
- Streamlit
- NumPy
- Pillow

## Run Locally
```
git clone https://github.com/tahsinttalha2/plant_disease_detection_system
pip install -r requirements.txt
python extract_class_names.py
streamlit run app.py
```
Check out the Streamlit app here: https://plant-disease-detection-tahsinttalha.streamlit.app

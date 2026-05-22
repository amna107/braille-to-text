# Braille to Text Converter

An end-to-end Braille OCR system comparing classical ML and deep learning,
built to help teachers in inclusive classrooms read Braille written by 
visually impaired students.

## Live Demo
- Streamlit Cloud: https://YOUR-APP.streamlit.app
- Hugging Face Spaces: https://huggingface.co/spaces/YOUR_USERNAME/braille-to-text

## Models
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM (RBF kernel) | 61.9% | 69.9% | 61.9% | 63.3% |
| Random Forest | 69.9% | 73.5% | 69.9% | 69.7% |
| CNN | 83.7% | 86.0% | 83.7% | 84.0% |

## Features
- Single character prediction with majority voting across 3 models
- Full document conversion with per-character confidence scores
- Model comparison with charts
- Dockerized for reproducible deployment

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Run with Docker
docker build -t braille-app .
docker run -p 8501:8501 braille-app
Open http://localhost:8501

## Tech stack
Python · TensorFlow/Keras · scikit-learn · Streamlit · OpenCV · Docker

## Dataset
Braille Character Dataset — 1560 images, 26 classes (a–z), 64×64 grayscale
(Kaggle: shanks0465/braille-character-dataset)
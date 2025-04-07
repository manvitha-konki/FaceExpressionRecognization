# Facial Expression Recognition 😄😢😠😮

A deep learning project that classifies human facial expressions into seven categories using a CNN-based architecture built on EfficientNetB0. The model is trained on the FER-2013 dataset with data augmentation and class balancing techniques for improved accuracy.

## 📌 Project Highlights

- Uses EfficientNetB0 as the base model for feature extraction
- Image preprocessing and augmentation using `ImageDataGenerator`
- Trained on FER-2013 facial expression dataset
- Handles class imbalance using class weights
- Achieved high accuracy on validation data

## 🧠 Emotion Classes

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

## 🗂️ Dataset

- [FER-2013 Facial Expression Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- Downloaded using KaggleHub directly into the notebook

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- EfficientNet (transfer learning)

# Facial Expression Recognition 😄😢😠😮

A deep learning project that classifies human facial expressions into seven categories using a CNN-based architecture built on EfficientNetB0. The model is trained on the FER-2013 dataset with data augmentation and class balancing techniques for improved accuracy.

## 📌 Project Highlights

- Uses EfficientNetB0 as the base model for feature extraction
- Image preprocessing and augmentation using `ImageDataGenerator`
- Trained on FER-2013 facial expression dataset
- Handles class imbalance using class weights

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

## 📊 Results
- **Best Validation Accuracy:** 61.7%
- **Final Validation Accuracy:** 60.5%
- **Test Accuracy:** 55%
- **Top Performing Classes:**  
  - **Happy:** F1-score = 0.79  
  - **Surprise:** F1-score = 0.67  

## 📌 Future Improvements
- Trying to improve Accuracy further
- Improve performance on underperforming classes (e.g., fear, disgust)
- Integrate real-time emotion detection using webcam (OpenCV)
- Deploy model with Streamlit or Flask for demo


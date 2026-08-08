# 🏭 Industrial Steel Surface Defect Classification

An end-to-end traditional Machine Learning project for automatic steel surface defect classification using handcrafted HOG features and Support Vector Machine (SVM).

---

## 📌 Overview

This project classifies six different types of industrial steel surface defects using classical Computer Vision and Machine Learning techniques.

Instead of Deep Learning, this project focuses on handcrafted feature extraction using Histogram of Oriented Gradients (HOG) followed by traditional ML classifiers.

The project also includes an interactive Streamlit web application for real-time defect prediction.

---

## ✨ Features

- Traditional Machine Learning Pipeline
- Histogram of Oriented Gradients (HOG)
- Support Vector Machine (Best Model)
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Interactive Streamlit Application
- Confusion Matrix
- Model Comparison

---

## 📂 Dataset

NEU Surface Defect Database

Classes:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

Training Images : 1440

Validation Images : 360

---

## 🏗 Project Pipeline

Input Image

↓

Image Preprocessing

↓

HOG Feature Extraction

↓

Machine Learning Model

↓

Defect Prediction

---

## 📊 Model Comparison

| Model | Validation Accuracy |
|--------|--------------------|
| Logistic Regression | 65.83% |
| Decision Tree | 28.06% |
| Random Forest | 55.00% |
| SVM | **78.06%** |
| XGBoost | 58.06% |

---

## 📈 Best Model

Support Vector Machine (SVM)

Validation Accuracy

**78.06%**

---

## 🖥 Streamlit Application

Upload a steel surface image and the application predicts the defect type instantly.

---

## 🛠 Tech Stack

- Python
- OpenCV
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Matplotlib
- scikit-image

---

## 📁 Project Structure

```
industrial-steel-surface-defect-classification

├── app.py

├── train.py

├── models/

├── screenshots/

├── requirements.txt

└── README.md
```

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Better handcrafted features
- Deep Learning comparison
- Live Streamlit deployment
- Explainable AI (Grad-CAM / SHAP)

---

## 👨‍💻 Author

Kushagra Singh

GitHub:

https://github.com/kush0074
# 🦴 Knee Arthritis Detection

An end-to-end deep learning application for classifying knee osteoarthritis severity from knee X-ray images using **TensorFlow**, **DenseNet121**, and **Streamlit**.

## 📖 Overview

This project predicts the severity of knee osteoarthritis from X-ray images by classifying them into one of five grades (Grade 0–4). The application provides an interactive web interface where users can upload an X-ray image and receive the predicted arthritis grade along with the model's confidence.

> **Note:** This project is intended for educational and research purposes only. It is **not** a medical diagnostic tool.

---

## 🚀 Features

- Upload knee X-ray images (PNG, JPG, JPEG)
- Deep learning-based severity classification
- Five-class osteoarthritis grading
- Prediction confidence score
- Interactive Streamlit web interface
- Clean and responsive user interface

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- DenseNet121 (Transfer Learning)
- Streamlit
- NumPy
- Pillow

---

## 📂 Project Structure

```
Arthritis-Detection/
│
├── app.py
├── train_model.py
├── train_densenet.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best_finetuned_densenet.keras
│
└── images/
```

---

## 📊 Model

- Architecture: DenseNet121
- Transfer Learning
- Five-class classification
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

Classes:

- Grade 0 – Normal
- Grade 1 – Doubtful
- Grade 2 – Mild
- Grade 3 – Moderate
- Grade 4 – Severe

---

## 📈 Performance

The best performing model achieved approximately:

- **Test Accuracy:** 44.75%

Due to the challenging nature of multi-class medical image classification and dataset limitations, predictions should be interpreted only as an educational demonstration.

---



---

## 📚 Dataset

This project uses the **KneeXrayMini** dataset available on Kaggle.

Dataset:
https://www.kaggle.com/datasets/tommyngx/kneexraymini


---

## ⚠️ Disclaimer

This application is designed solely for educational and research purposes.

It is **not intended for clinical use**, medical diagnosis, or treatment decisions.

---

## 👩‍💻 Author

**Bhavana Manda**


---

## ⭐ Future Improvements

- Improve classification accuracy using larger datasets
- Experiment with newer deep learning architectures
- Add Grad-CAM visualizations for model interpretability
- Improve user interface and user experience

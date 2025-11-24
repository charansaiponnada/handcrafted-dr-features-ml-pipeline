
# Severity-Aware Diabetic Retinopathy Classification  
### PCA-Reduced Handcrafted Image Features + Ensemble Machine Learning Models

This repository provides a complete end-to-end pipeline for **Diabetic Retinopathy (DR) severity classification** using **handcrafted fundus image features**, **PCA-based dimensionality reduction**, and **ensemble classical machine learning models**.  
A **Streamlit web application** is included for interactive testing and cloud deployment.

---

## 🚀 Project Highlights

### 🔍 Handcrafted Feature Extraction
- **GLCM Texture Features** (multi-angle, multi-distance)
- **LBP (Local Binary Patterns)**
- **RGB Color Histograms**
- **CLAHE Preprocessing** for contrast enhancement

### ⚙️ Machine Learning
Models trained on PCA-reduced features:
- **Random Forest (best performer)**
- **Support Vector Machine (RBF)**
- **Logistic Regression**
- **Naive Bayes**

### 📉 Dimensionality Reduction (PCA)
- Automatic component selection using **99% variance**
- Reduces ~109 handcrafted features → ~11–40 PCA components

### 🌐 Streamlit Application
- Upload fundus image
- Preprocessing preview
- Per-model predictions
- Confidence scores (softmax probabilities)
- **Automatic label lookup from test.csv**
- **Manual label override (Streamlit Cloud-safe)**
- Correct/Incorrect evaluation
- Downloadable prediction report (CSV)

---

## 📊 Model Performance (Validation Set)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **Random Forest** | **0.765** | **0.715** |
| SVM (RBF) | 0.697 | 0.705 |
| Logistic Regression | 0.664 | 0.680 |
| Naive Bayes | 0.587 | 0.597 |

---

## 📁 Folder Structure

```

project/
│── app.py
│── training.py
│── metrics.csv
│── README.md
│── models/
│     ├── scaler.pkl
│     ├── pca.pkl
│     ├── rf.pkl
│     ├── svm.pkl
│     ├── lr.pkl
│     ├── nb.pkl
│── datasets/
│     ├── train_images/
│     ├── val_images/
│     ├── test_images/
│     ├── train_1.csv
│     ├── valid.csv
│     ├── test.csv
│── requirements.txt

````

---

## ▶️ Training the Models

Run the full training pipeline:

```bash
python training.py
````

Outputs:

* `models/*.pkl`
* `metrics.csv`

---

## ▶️ Running the Streamlit App

Local run:

```bash
streamlit run app.py
```

Access the app in your browser:

```
http://localhost:8501
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push the repository to GitHub.
2. Go to: [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repository.
4. Set the entry point to:

   ```
   app.py
   ```
5. Add the following to `requirements.txt`:

```
streamlit
scikit-learn
scikit-image
opencv-python-headless
joblib
numpy
pandas
matplotlib
```

6. Click **Deploy**.

---

## 🧠 Methodology Overview

### 1. Preprocessing

* Border removal
* CLAHE contrast enhancement
* Resize to 224×224

### 2. Feature Engineering

| Feature Type | Description                        |
| ------------ | ---------------------------------- |
| GLCM         | contrast, ASM, homogeneity, energy |
| LBP          | uniform pattern histogram          |
| Color        | 32-bin RGB histograms              |

Total features ≈ **109**

### 3. PCA

* StandardScaler → PCA
* 99% variance threshold
* Creates compact, noise-reduced feature vector

### 4. ML Model Training

Each classifier trained on PCA embeddings.

### 5. Evaluation

* Accuracy, precision, recall, F1
* Saved to `metrics.csv`

---

## 📈 Streamlit Features

* Original & Preprocessed image visualization
* DR severity prediction (0–4)
* Per-model prediction + probabilities
* Ensemble majority prediction
* **Auto correctness from test.csv**
* **Manual correctness selection**
* Prediction Report Download (CSV)

---

## 📜 Citation (If used in academic work)

```
Ponnada Charan Sai.
Severity-Aware Diabetic Retinopathy Classification using PCA-Reduced Handcrafted Fundus Image Features and Ensemble Machine Learning Algorithms.
2025.
```

---

## 🤝 Contributing

Pull requests and issues are welcome.

---

## 📬 Contact

Maintainer: **Ponnada Charan Sai**
Project: **DR Severity Classification using PCA + ML Ensemble**

```


import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

st.set_page_config(page_title="DR Classification", layout="wide")

# --------------------------------------------
# LOAD MODELS
# --------------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    rf = joblib.load("models/rf.pkl")
    svm = joblib.load("models/svm.pkl")
    lr = joblib.load("models/lr.pkl")
    nb = joblib.load("models/nb.pkl")
    return scaler, pca, rf, svm, lr, nb

scaler, pca, rf, svm, lr, nb = load_models()

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


# --------------------------------------------
# PREPROCESS FUNCTION
# --------------------------------------------
def preprocess_image_from_bytes(file_bytes, size=224):
    file_bytes = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(th)
    if w > 10 and h > 10:
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (size, size))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img


# --------------------------------------------
# FEATURE EXTRACTOR
# --------------------------------------------
def extract_features(image):
    features = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1, 2, 3],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True,
        normed=True,
        levels=256
    )

    features += [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'ASM').mean()
    ]

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    features += hist.tolist()

    for ch in cv2.split(image):
        h = cv2.calcHist([ch], [0], None, [32], [0, 256]).flatten()
        features += h.tolist()

    return np.array(features).reshape(1, -1)


# --------------------------------------------
# PREDICT FUNCTION
# --------------------------------------------
def predict(image):
    feats = extract_features(image)
    feats_scaled = scaler.transform(feats)
    feats_pca = pca.transform(feats_scaled)

    predictions = {
        "RandomForest": int(rf.predict(feats_pca)[0]),
        "SVM": int(svm.predict(feats_pca)[0]),
        "LogisticRegression": int(lr.predict(feats_pca)[0]),
        "NaiveBayes": int(nb.predict(feats_pca)[0]),
    }

    final_pred = max(set(predictions.values()), key=list(predictions.values()).count)
    return predictions, final_pred


# --------------------------------------------
# STREAMLIT UI
# --------------------------------------------
st.title("👁️ Diabetic Retinopathy Classifier (Automatic Ground Truth Check)")

uploaded_file = st.file_uploader("Upload a Test Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    file_bytes = uploaded_file.read()
    processed = preprocess_image_from_bytes(file_bytes)

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

    with col2:
        st.image(processed, caption="Preprocessed Image", use_column_width=True)

    preds, final_pred = predict(processed)

    st.subheader("Model Predictions")
    st.table({
        "Model": preds.keys(),
        "Predicted Class": preds.values(),
        "Severity": [CLASS_NAMES[v] for v in preds.values()]
    })

    st.subheader("Ensemble Prediction")
    st.success(f"Final Severity Prediction: {CLASS_NAMES[final_pred]}")


    # ----------------------------------------------------------
    # AUTO FETCH GROUND TRUTH FROM test.csv
    # ----------------------------------------------------------
    TEST_CSV_PATH = r"C:\Users\cherr\Downloads\ml-dataset\test.csv"

    if os.path.exists(TEST_CSV_PATH):
        test_df = pd.read_csv(TEST_CSV_PATH)

        filename = uploaded_file.name   # e.g. "1ae8c165fd53.png"
        file_id = os.path.splitext(filename)[0]  # remove .png

        if file_id in test_df["id_code"].values:
            actual_label = int(test_df.loc[test_df["id_code"] == file_id, "diagnosis"].values[0])

            st.subheader("Ground Truth from test.csv")
            st.info(f"Actual Severity: **{CLASS_NAMES[actual_label]}**")

            if final_pred == actual_label:
                st.success(f"✔ Correct Prediction! ({CLASS_NAMES[final_pred]})")
            else:
                st.error(
                    f"✖ Incorrect Prediction\n"
                    f"Predicted: {CLASS_NAMES[final_pred]}\n"
                    f"Actual: {CLASS_NAMES[actual_label]}"
                )
        else:
            st.warning("⚠ This image ID was not found in test.csv.")
    else:
        st.warning("⚠ test.csv not found. Place test.csv in the dataset folder.")

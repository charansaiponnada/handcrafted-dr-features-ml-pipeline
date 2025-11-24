import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from io import StringIO

st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="wide")

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# PREPROCESS IMAGE
# ---------------------------------------------------------
def preprocess_image(file_bytes, size=224):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
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


# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
def extract_features(image):
    features = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1,2,3],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256, symmetric=True, normed=True
    )

    features += [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'ASM').mean()
    ]

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0,10), range=(0,9))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    features += hist.tolist()

    for ch in cv2.split(image):
        h = cv2.calcHist([ch], [0], None, [32], [0,256]).flatten()
        features += h.tolist()

    return np.array(features).reshape(1, -1)


# ---------------------------------------------------------
# PREDICT MODELS + CONFIDENCE SCORES
# ---------------------------------------------------------
def predict(image):
    feats = extract_features(image)
    scaled = scaler.transform(feats)
    reduced = pca.transform(scaled)

    models = {
        "Random Forest": rf,
        "SVM": svm,
        "Logistic Regression": lr,
        "Naive Bayes": nb
    }

    results = {}
    probas = {}

    for name, model in models.items():
        pred = int(model.predict(reduced)[0])
        results[name] = pred

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(reduced)[0]
        else:
            proba = np.zeros(5)

        probas[name] = proba

    final_pred = max(
        set(results.values()),
        key=list(results.values()).count
    )

    return results, probas, final_pred


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("👁️ Diabetic Retinopathy Classification")
st.write("Upload a fundus image. The app predicts severity using PCA-reduced handcrafted features and ML models.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    file_bytes = uploaded_file.read()
    processed = preprocess_image(file_bytes)

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

    with col2:
        st.image(processed, caption="Preprocessed Image", use_column_width=True)

    # MODEL PREDICTIONS
    preds, probas, final_pred = predict(processed)

    st.subheader("Model Predictions")
    pred_table = pd.DataFrame({
        "Model": preds.keys(),
        "Predicted Class": preds.values(),
        "Severity": [CLASS_NAMES[v] for v in preds.values()]
    })
    st.table(pred_table)

    # Probability Table
    st.subheader("Confidence Scores (Probabilities)")
    prob_df = pd.DataFrame(probas).T
    prob_df.columns = [CLASS_NAMES[i] for i in range(5)]
    st.dataframe(prob_df.style.highlight_max(axis=1))


    st.subheader("Final Ensemble Prediction")
    st.success(f"Ensemble Severity: {CLASS_NAMES[final_pred]}")


    # ---------------------------------------------------------
    # AUTOMATIC + MANUAL CORRECTNESS
    # ---------------------------------------------------------
    st.subheader("Correctness Check")

    TEST_CSV_PATH = r"C:\Users\cherr\Downloads\ml-dataset\test.csv"
    actual_label = None
    auto_matched = False

    if os.path.exists(TEST_CSV_PATH):
        test_df = pd.read_csv(TEST_CSV_PATH)
        file_id = os.path.splitext(uploaded_file.name)[0]

        if file_id in test_df["id_code"].values:
            actual_label = int(test_df.loc[test_df["id_code"] == file_id, "diagnosis"].values[0])
            auto_matched = True
            st.info(f"Auto-Fetched Actual Severity: **{CLASS_NAMES[actual_label]}**")

    if not auto_matched:
        actual_label = st.selectbox(
            "Actual Severity (Manual Input):",
            options=[0,1,2,3,4],
            format_func=lambda x: CLASS_NAMES[x]
        )

    # Correctness Evaluation
    if final_pred == actual_label:
        st.success(f"✔ Correct Prediction! ({CLASS_NAMES[final_pred]})")
    else:
        st.error(
            f"✖ Incorrect Prediction\n"
            f"Predicted: {CLASS_NAMES[final_pred]}\n"
            f"Actual: {CLASS_NAMES[actual_label]}"
        )


    # ---------------------------------------------------------
    # DOWNLOAD PREDICTION REPORT
    # ---------------------------------------------------------
    st.subheader("Download Prediction Report")

    report = pd.DataFrame({
        "Model": preds.keys(),
        "Predicted_Class": preds.values(),
        "Severity": [CLASS_NAMES[v] for v in preds.values()]
    })

    report["Actual_Label"] = actual_label
    report["Actual_Severity"] = CLASS_NAMES[actual_label]
    report["Ensemble"] = CLASS_NAMES[final_pred]

    csv_buffer = StringIO()
    report.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Report CSV",
        data=csv_buffer.getvalue(),
        file_name="prediction_report.csv",
        mime="text/csv"
    )

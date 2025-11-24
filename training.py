# training.py (FULL VERSION WITH METRICS CSV)

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib


# --------------------------------------------------------
# CONFIG PATHS
# --------------------------------------------------------
BASE_DIR = r"C:\Users\cherr\Downloads\ml-dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
VAL_DIR   = os.path.join(BASE_DIR, "val_images")

TRAIN_CSV = os.path.join(BASE_DIR, "train_1.csv")
VAL_CSV   = os.path.join(BASE_DIR, "valid.csv")

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

print("[INFO] Loaded TRAIN rows:", len(train_df))
print("[INFO] Loaded VAL rows:", len(val_df))


# --------------------------------------------------------
# IMAGE LOADER
# --------------------------------------------------------
def load_image_path(image_id, folder):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(folder, image_id + ext)
        if os.path.exists(path):
            return path
    return None


# --------------------------------------------------------
# BUILD TRAIN LIST
# --------------------------------------------------------
train_paths = []
train_labels = []

for idx, row in train_df.iterrows():
    p = load_image_path(row["id_code"], TRAIN_DIR)
    if p:
        train_paths.append(p)
        train_labels.append(row["diagnosis"])
    else:
        print("[WARNING] Missing train image:", row["id_code"])

print("[INFO] Total train images found:", len(train_paths))


# --------------------------------------------------------
# BUILD VALIDATION LIST
# --------------------------------------------------------
val_paths = []
val_labels = []

for idx, row in val_df.iterrows():
    p = load_image_path(row["id_code"], VAL_DIR)
    if p:
        val_paths.append(p)
        val_labels.append(row["diagnosis"])
    else:
        print("[WARNING] Missing val image:", row["id_code"])

print("[INFO] Total validation images found:", len(val_paths))


# --------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------
def preprocess_image(path, size=224):
    img = cv2.imread(path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # remove black borders
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(th)
    if w > 10 and h > 10:
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (size, size))

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img


# --------------------------------------------------------
# SHOW SAMPLE IMAGE BEFORE/AFTER
# --------------------------------------------------------
sample_path = train_paths[0]
orig = cv2.imread(sample_path)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
proc = preprocess_image(sample_path)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(orig); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(proc); plt.title("Preprocessed"); plt.axis("off")
plt.show()


# --------------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------------
def extract_features(image):
    features = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # GLCM FEATURES
    glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    features += [
        graycoprops(glcm, 'contrast')[0,0],
        graycoprops(glcm, 'energy')[0,0],
        graycoprops(glcm, 'homogeneity')[0,0],
        graycoprops(glcm, 'ASM')[0,0],
    ]

    # LBP FEATURES
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    features += hist.tolist()

    # RGB COLOR HISTOGRAMS
    for channel in cv2.split(image):
        hist = cv2.calcHist([channel], [0], None, [32], [0, 256]).flatten()
        features += hist.tolist()

    return features


# --------------------------------------------------------
# BUILD FEATURE MATRIX (TRAIN)
# --------------------------------------------------------
print("[INFO] Extracting TRAINING features...")

X_train = []
y_train = train_labels

for path in tqdm(train_paths):
    img = preprocess_image(path)
    feats = extract_features(img)
    X_train.append(feats)

X_train = np.array(X_train)
y_train = np.array(y_train)

print("[INFO] Training features:", X_train.shape)


# --------------------------------------------------------
# PCA (AUTO SELECT 95% VARIANCE)
# --------------------------------------------------------
print("[INFO] Running PCA...")
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)

print("[INFO] PCA selected:", pca.n_components_)
print("[INFO] PCA output:", X_train_pca.shape)


# --------------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------------
print("[INFO] Training models...")

rf = RandomForestClassifier(n_estimators=300, random_state=42)
svm = SVC(kernel='rbf', probability=True)
lr = LogisticRegression(max_iter=2000)
nb = GaussianNB()

rf.fit(X_train_pca, y_train)
svm.fit(X_train_pca, y_train)
lr.fit(X_train_pca, y_train)
nb.fit(X_train_pca, y_train)

print("[INFO] Training complete!")


# --------------------------------------------------------
# VALIDATION FEATURES
# --------------------------------------------------------
print("[INFO] Extracting VALIDATION features...")

X_val = []

for path in tqdm(val_paths):
    img = preprocess_image(path)
    feats = extract_features(img)
    X_val.append(feats)

X_val = np.array(X_val)
y_val = np.array(val_labels)

print("[INFO] Validation features:", X_val.shape)


# --------------------------------------------------------
# APPLY PCA TO VALIDATION SET
# --------------------------------------------------------
X_val_pca = pca.transform(X_val)


# --------------------------------------------------------
# EVALUATE MODELS & SAVE CSV
# --------------------------------------------------------
models = {
    "RandomForest": rf,
    "SVM": svm,
    "LogisticRegression": lr,
    "NaiveBayes": nb
}

results = []

print("\n[INFO] Computing validation metrics...")

for name, model in models.items():
    preds = model.predict(X_val_pca)

    acc  = accuracy_score(y_val, preds)
    f1   = f1_score(y_val, preds, average="weighted")
    prec = precision_score(y_val, preds, average="weighted")
    rec  = recall_score(y_val, preds, average="weighted")

    results.append([name, acc, prec, rec, f1])

    print(f"{name}:  Acc={acc:.4f},  Prec={prec:.4f},  Rec={rec:.4f},  F1={f1:.4f}")


metrics_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

metrics_df.to_csv("metrics.csv", index=False)
print("\n[INFO] Metrics saved to metrics.csv")


# --------------------------------------------------------
# SAVE MODELS
# --------------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(pca, "models/pca.pkl")
joblib.dump(rf, "models/rf.pkl")
joblib.dump(svm, "models/svm.pkl")
joblib.dump(lr, "models/lr.pkl")
joblib.dump(nb, "models/nb.pkl")

print("[INFO] All models saved to /models/")

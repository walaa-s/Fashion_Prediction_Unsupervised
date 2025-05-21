import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
from PIL import Image
import numpy as np
import joblib
from skimage.feature import hog
import hashlib

# --- Load pre-trained models ---
scaler = joblib.load("/Users/mohannedalsahaf/iCloud Drive (Archive)/Mac Desktop/Tuwaiq Data Science Bootcamp/scaler.pkl")
pca = joblib.load("/Users/mohannedalsahaf/iCloud Drive (Archive)/Mac Desktop/Tuwaiq Data Science Bootcamp/pca.pkl")
kmeans = joblib.load("/Users/mohannedalsahaf/iCloud Drive (Archive)/Mac Desktop/Tuwaiq Data Science Bootcamp/kmeans_hog_svd_labels.pkl")

# Fashion MNIST class names
class_names = ['T-shirt/top', 
               'Trouser', 
               'Pullover',
                 'Dress', 
                 'Coat',
               'Sandal', 
               'Shirt', 
               'Sneaker', 
               'Bag', 
               'Ankle boot']

# --- Hardcoded cluster-to-class mapping (from user) ---
cluster_to_class = {
    0: 1,  # Trouser
    1: 3,  # Dress
    2: 7,  # Sneaker
    3: 2,  # Pullover
    4: 9,  # Ankle boot
    5: 0,  # T-shirt/top
    6: 5,  # Sandal (fixed typo "Sabdal")
    7: 8,  # Bag
    8: 9,  # Ankle boot (assumed correction from "Bag boot")
    9: 7   # Sneaker
}

# --- Streamlit UI ---
st.set_page_config(page_title="Fashion Clustering (HOG + PCA)", layout="wide")
st.title("👕 Predict Clothing Type with HOG + PCA Clustering")
st.write("Upload a 28×28 grayscale image to predict its cluster and matching fashion label.")

uploaded_file = st.file_uploader("Upload a grayscale image (28x28)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Step 1: Load and preprocess image
    image = Image.open(uploaded_file).convert("L").resize((28, 28), resample=Image.NEAREST)
    st.image(image, caption="Uploaded Image", width=150)

    # Step 2: Normalize and compute hash (to check consistency)
    image_np = np.array(image).astype("float32") / 255.0
    img_hash = hashlib.sha256(image_np.tobytes()).hexdigest()
    #st.write("🧪 Image SHA-256 Hash (stability check):", img_hash)

    # Step 3: Extract HOG features
    hog_features = hog(image_np, orientations=8, pixels_per_cell=(7, 7),
                       cells_per_block=(1, 1), visualize=False).reshape(1, -1)

    # Step 4: Apply scaler and PCA
    hog_std = scaler.transform(hog_features)
    hog_pca = pca.transform(hog_std)

    # Step 5: Predict cluster
    cluster = kmeans.predict(hog_pca)[0]

    # Step 6: Map cluster to class label
    predicted_class_id = cluster_to_class.get(cluster, None)
    if predicted_class_id is not None and 0 <= predicted_class_id < len(class_names):
        label_name = class_names[predicted_class_id]
    else:
        label_name = "Unknown"

    # Debug info
    st.write("🔎 HOG vector (first 5):", hog_features[0][:5])
    st.write("🔬 PCA vector (first 5):", hog_pca[0][:5])

    # Step 7: Show prediction
    st.success(f"Predicted Cluster ID: {cluster}")
    st.info(f"Predicted Class: {label_name}")

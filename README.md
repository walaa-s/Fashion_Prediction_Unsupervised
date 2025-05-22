# 👕 Fashion MNIST Clustering App (Streamlit)

This Streamlit web app performs unsupervised image clustering using:
- HOG (Histogram of Oriented Gradients) features
- PCA (Principal Component Analysis)
- KMeans clustering

It predicts the fashion category of uploaded 28×28 grayscale images from the Fashion MNIST dataset.

## You can use any image and the model will predict it 
---

## 🚀 How It Works

1. You upload a 28×28 grayscale image (PNG or JPG).
2. The app:
   - Extracts HOG features
   - Applies StandardScaler and PCA
   - Uses a trained KMeans model to predict the cluster
3. The cluster ID is mapped to the corresponding Fashion MNIST class name.

---

## 🧠 Model Architecture

- Feature Extraction: `skimage.feature.hog`
- Dimensionality Reduction: `sklearn.decomposition.PCA`
- Clustering: `sklearn.cluster.KMeans`

---

## 🛠️ Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt

```
## ScreenShot of streamlit
<img width="1728" alt="Screenshot 1446-11-24 at 11 56 32 AM" src="https://github.com/user-attachments/assets/a9226fb7-1470-4fc5-8398-714824bdf8e8" />

<img width="1728" alt="Screenshot 1446-11-24 at 11 55 36 AM" src="https://github.com/user-attachments/assets/beacf4fc-4f22-4550-b93c-47c70d292dad" />





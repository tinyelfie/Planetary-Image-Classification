# 🪐 Planetary Image Generation and Classification

A deep learning project that explores **image classification** and **generative modeling** using solar system planetary images. Built entirely in Python using TensorFlow/Keras and run on Google Colab.

---

## 📌 Project Overview

This project trains and compares multiple deep learning architectures to classify images of planets (and dwarf planets) in our solar system. It also implements a **Generative Adversarial Network (GAN)** to synthesize new planetary images. The entire pipeline — from raw image loading to final model comparison — is contained in a single, self-contained Jupyter Notebook.

**Dataset:** Images of 11 celestial bodies stored on Google Drive:
> Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Makemake, Moon

---

## 🧠 What's Inside the Notebook

The notebook is organized into **12 sections**, each covering a key deep learning concept:

| Section | Topic |
|---|---|
| 1 | Setup, Data Loading & Preprocessing |
| 2 | Data Augmentation |
| 3 | Convolutional Autoencoder (Feature Extraction) |
| 4 | MLP Baseline Classifier |
| 5 | Custom CNN Classifier (Raw Data) |
| 6 | CNN Classifier with Augmented Data |
| 7 | Transfer Learning with EfficientNetB0 |
| 8 | DCGAN for Planetary Image Generation |
| 9 | CNN Trained on GAN-Augmented Data |
| 10 | CNN + LSTM Hybrid Classifier |
| 11 | Optimizer Comparison (SGD vs RMSProp vs Adam) |
| 12 | Final Results & Conclusion |

---

## 🏗️ Architecture Summary

- **Preprocessing:** Images resized to `128×128`, normalized to `[0, 1]`, and split 80/20 for train/test.
- **Augmentation:** Random rotations (±20°), horizontal flips, zoom (±20%), brightness variation (80–120%).
- **Autoencoder:** Encoder compresses images to a `16×16×128` bottleneck; used as a feature extractor for the MLP.
- **CNN:** 3-block convolutional architecture with BatchNorm and Dropout; trained on raw and augmented data.
- **Transfer Learning:** EfficientNetB0 (ImageNet weights, frozen) with a custom classification head.
- **GAN:** DCGAN with a 100-dim latent noise vector; trained for 500 steps with label smoothing.
- **CNN + LSTM:** CNN feature maps reshaped into sequences fed into an LSTM for spatial-temporal classification.

---

## 🚀 How to Run

1. **Open in Google Colab** (recommended — the notebook mounts Google Drive automatically).
2. **Upload your dataset** to Google Drive at the path:
   ```
   /content/drive/MyDrive/Planets/
   ```
   Each planet should have its own subfolder (e.g., `Planets/Earth/`, `Planets/Mars/`, etc.).
3. **Run all cells** top-to-bottom. The notebook handles missing folders gracefully.

> **Note:** If the dataset path is not found, the notebook auto-generates a small synthetic (random noise) dataset so cells do not crash — useful for dry runs.

---

## 📦 Requirements

All dependencies are pre-installed on Google Colab. For local runs:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn pandas
```

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | Model building and training |
| `OpenCV (cv2)` | Image loading and color conversion |
| `NumPy` | Array operations |
| `Matplotlib` | Visualizations |
| `scikit-learn` | Label encoding, train/test split |
| `Pandas` | Result summary tables |

---

## 📊 Key Results

> *(Values are representative; actual results vary with dataset size and hardware.)*

| Model | Test Accuracy |
|---|---|
| MLP Baseline (on Autoencoder features) | ~60–70% |
| CNN (Original Data) | ~70–80% |
| CNN (Augmented Data) | Higher than CNN Original |
| Transfer Learning (EfficientNetB0) | **Highest (~85–92%)** |
| CNN + GAN Augmented Data | Comparable to CNN Original |
| CNN + LSTM Hybrid | ~65–75% |

**Best Model:** Transfer Learning with EfficientNetB0 consistently achieves the highest accuracy by leveraging ImageNet pre-trained weights.





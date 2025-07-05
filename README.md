# 🧼 Clean vs Messy Room Classifier using CNN

<div align="center">
  
📸 **Room Cleanliness Detector** powered by TensorFlow  
Classify room images as `Clean` 🧽 or `Messy` 🧹 with a custom-trained CNN model.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🎯 Overview

This project uses **Convolutional Neural Networks** to classify images of rooms based on their cleanliness.

> 🧠 Useful for smart homes, cleaning bots, or even just for fun in AI image classification projects!

---

## ✨ Highlights

- ✔️ Clean vs Messy classification
- 📁 Custom dataset with labeled folders
- 🧪 Trainable CNN using `model_building.py`
- 🔮 Real-time predictions using `predict_image.py`
- 💾 Pre-trained model provided via Google Drive

---

## 📂 Folder Structure

clean-vs-messy-room/
├── dataset/
│ ├── clean/
│ └── dirty/
├── data_preprocessing.py
├── model_building.py
├── predict_image.py
├── README.md
└── .gitignore

> The file `cleanliness_model.h5` is **not included in this repo** due to GitHub's 100MB limit. See download below.

---

## 🔽 Download the Pretrained Model

The trained model (`cleanliness_model.h5`, ~218 MB) is hosted on Google Drive:

👉 **[Click here to download cleanliness_model.h5]([https://drive.google.com/file/d/1AbCDeFgHiJkLmNopQRStUvWxYz123456/view?usp=sharing](https://drive.google.com/drive/folders/1cxYpmxDMiDTAfF4qmJW_UzfRp6Rom4Rk?usp=sharing))**

> After downloading, place the file in your project’s root directory:
>
> ```
> clean-vs-messy-room/
> ├── cleanliness_model.h5  ✅
> ```

---

## ⚙️ How to Use

### ▶️ 1. Install dependencies

```bash
pip install tensorflow opencv-python numpy matplotlib

🧠 2. Train the model (optional)
You can train the model from scratch if you prefer:

🧑‍💻 Author
Made with 💡 and ☕ by @Ravenn19
Feel free to open issues, fork the repo, or suggest improvements!

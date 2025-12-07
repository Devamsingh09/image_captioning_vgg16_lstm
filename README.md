
# 🖼️🔊 Image Caption Generator with Audio (VGG16 + LSTM)

An end-to-end **Image Captioning application** that generates descriptive captions for images and converts the caption into **audio** using Google Text-to-Speech.  
The project is deployed using **Streamlit** and demonstrates the complete deep learning workflow — from preprocessing to deployment.

---

## 🚀 Demo Features

- 📸 Upload an image (jpg / png / jpeg)
- 🧠 Generate an English caption using a trained CNN–LSTM model
- 🔊 Convert the generated caption into speech (audio output)
- ⚡ Fast inference with locally saved models (no re-downloads)

---

## 🧠 Model Architecture

### 🔹 Image Encoder
- **VGG16** (pretrained on ImageNet)
- Uses the **penultimate fully connected layer** as image features

### 🔹 Caption Decoder
- **LSTM-based language model**
- Generates captions word-by-word using greedy decoding

### 🔹 Training Dataset
- Flickr-style image caption dataset  
- Start / end tokens used for sequence modeling

---

## 📈 Evaluation

The model was evaluated using BLEU metrics:

| Metric | Score |
|------|------|
| BLEU-1 | **0.52** |
| BLEU-2 | **0.30** |

✅ These scores indicate good object recognition and reasonable sentence fluency for a CNN–LSTM baseline model.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras**
- **Streamlit**
- **VGG16 (CNN)**
- **LSTM**
- **Google Text-to-Speech (gTTS)**
- **Git & Git LFS**

---

## 📁 Project Structure

```

image_captioning/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
├── model.keras                     # Trained captioning model
├── tokenizer.pkl                  # Tokenizer
├── features.pkl                   # Extracted image features
└── vgg16_feature_extractor.keras  # Locally saved VGG16

````

> ⚠️ Large model files are tracked using **Git LFS**.

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone <repo-url>
cd image_captioning
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser.

---

## 🔊 Caption + Audio Generation Flow

```
Image Upload
     ↓
VGG16 Feature Extraction
     ↓
LSTM Caption Generation
     ↓
Text Caption
     ↓
Google Text-to-Speech
     ↓
Audio Output
```

---

## ⚠️ Known Limitations

* Uses **greedy decoding** (can cause repetitive captions)
* No attention mechanism
* CNN features are global (no object-level focus)

These are known limitations of CNN–LSTM architectures.

---

## 🌱 Future Improvements

* ✅ Beam Search decoding
* ✅ Attention mechanism
* ✅ Transformer-based models (BLIP / ViT)
* ✅ Multilingual captioning
* ✅ Deployment to Streamlit Cloud / HuggingFace Spaces

---

## 🎯 Learning Outcomes

* Built an end-to-end image captioning pipeline
* Understood sequence modeling with LSTMs
* Debugged real deployment issues (RGBA images, pickle errors, model loading)
* Deployed a multimodal AI app with audio output
* Used **Git LFS** for large deep learning models

---

## 👤 Author

**Devam Singh**
B.Tech CSE (DSAI), Class of 2026

📧 Email: [devamsingh0009@gmail.com](mailto:devamsingh0009@gmail.com)
🔗 GitHub: [https://github.com/Devamsingh09](https://github.com/Devamsingh09)
🔗 LinkedIn: [https://linkedin.com/in/devam-singh-248025265/](https://linkedin.com/in/devam-singh-248025265/)

---

## ⭐ Acknowledgements

* TensorFlow & Keras
* Streamlit
* Google Text-to-Speech
* Flickr Image Caption Dataset

---

> ⭐ If you find this project useful, feel free to star the repository!

```



Just tell me 👍
```

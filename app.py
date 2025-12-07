import streamlit as st
import numpy as np
import pickle
import os
import tempfile

from PIL import Image
from gtts import gTTS

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
MAX_LENGTH = 35   # MUST be same as training
MODEL_DIR = "models"

st.set_page_config(page_title="Image Caption + Audio", layout="centered")
st.title("🖼️🔊 Image Caption Generator with Audio")

# --------------------------------------------------------
# LOAD MODELS & FILES (CACHED)
# --------------------------------------------------------
@st.cache_resource
def load_assets():
    # Captioning model
    caption_model = load_model(os.path.join(MODEL_DIR, "model.keras"))

    # Tokenizer
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    # Dataset image features (optional but loaded)
    with open(os.path.join(MODEL_DIR, "features.pkl"), "rb") as f:
        features = pickle.load(f)

    # ✅ Load saved VGG16 feature extractor (NO DOWNLOAD)
    vgg_model = load_model(
        os.path.join(MODEL_DIR, "vgg16_feature_extractor.keras")
    )

    return caption_model, tokenizer, features, vgg_model


model, tokenizer, features, vgg_model = load_assets()

# --------------------------------------------------------
# UTILITY FUNCTIONS (UNCHANGED LOGIC)
# --------------------------------------------------------
def convert_to_word(number, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == number:
            return word
    return None


def predict_caption(model, image_feature, tokenizer, max_length):
    in_text = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([image_feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)

        word = convert_to_word(y_pred, tokenizer)
        if word is None:
            break

        in_text += " " + word
        if word == "endseq":
            break

    return in_text


def extract_features(image, vgg_model):
   
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature


def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# --------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption and audio..."):
        feature = extract_features(image, vgg_model)

        caption = predict_caption(
            model,
            feature,
            tokenizer,
            MAX_LENGTH
        )

        caption = caption.replace("startseq", "").replace("endseq", "").strip()
        audio_path = text_to_speech(caption)

    st.success("✅ Caption Generated")

    st.subheader("📝 Caption")
    st.write(caption)

    st.subheader("🔊 Audio Output")
    st.audio(audio_path, format="audio/mp3")

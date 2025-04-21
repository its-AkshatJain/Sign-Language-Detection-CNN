import streamlit as st
import cv2
import numpy as np
import json
import os
import time
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Sign Language Detection", layout="wide")
st.title("🧠 Sign Language Real-Time Classifier")

# Load model
model = None
model_path = r'working MobileNetV2\sign_language_model.keras'


if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Model loaded from {model_path}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
else:
    st.warning(f"Model not found at {model_path}. Using simulated predictions.")

# Load class indices
if not os.path.exists('class_indices.json'):
    st.warning("class_indices.json not found. Creating dummy class index map.")
    sample_classes = {str(i): chr(i + 97) for i in range(26)}
    sample_classes.update({"26": "del", "27": "space", "28": "nothing"})
    with open('class_indices.json', 'w') as f:
        json.dump(sample_classes, f)

with open('class_indices.json', 'r', encoding='utf-8') as f:
    class_indices = json.load(f)

inv_class_indices = {int(v): k for k, v in class_indices.items()}
num_classes = len(inv_class_indices)
st.write(f"Ready for {num_classes} sign language classes")

# -------------------------
# Image preprocessing block
# -------------------------

def simplify_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Optional adaptive thresholding
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY_INV, 11, 2)
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

def jpeg_compress(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc_img = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(enc_img, 1)

def preprocess_image(image):
    image = cv2.resize(image, (200, 200))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# -------------------------
# Main image handler
# -------------------------

def process_image(img):
    img_array = np.array(img)
    img_cv = img_array[:, :, ::-1].copy()  # RGB -> BGR
    display_image = img_cv.copy()

    h, w = img_cv.shape[:2]
    size = int(min(h, w) * 0.8)
    x = (w - size) // 2
    y = (h - size) // 2

    # Draw green ROI box
    cv2.rectangle(display_image, (x, y), (x+size, y+size), (0, 255, 0), 2)

    roi = img_cv[y:y+size, x:x+size]

    # Simplify captured image to match training data
    roi = simplify_image(roi)
    roi = jpeg_compress(roi, quality=50)

    try:
        input_image = preprocess_image(roi)

        if model is not None:
            probs = model.predict(input_image)[0]
        else:
            probs = np.zeros(num_classes)
            rand_idx = np.random.randint(0, num_classes)
            probs[rand_idx] = np.random.uniform(0.5, 0.95)
            for i in range(num_classes):
                if i != rand_idx:
                    probs[i] = (1.0 - probs[rand_idx]) / (num_classes - 1)

        top3_indices = np.argsort(probs)[::-1][:3]
        top3_labels = [(inv_class_indices.get(i, str(i)), probs[i]) for i in top3_indices]

        y_pos = 30
        for label, conf in top3_labels:
            cv2.putText(display_image, f"{label}: {conf:.2f}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_pos += 40

    except Exception as e:
        cv2.putText(display_image, f"Error: {str(e)[:20]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return display_image[:, :, ::-1]  # BGR -> RGB for Streamlit

# -------------------------
# UI Section
# -------------------------

st.markdown("""
## Instructions:
1. Allow camera access or upload an image
2. Place your hand inside the green box
3. Hold still for best results
4. Use Right hand for making signs
""")

col1, col2 = st.columns(2)

with col1:
    camera_image = st.camera_input("📷 Capture via webcam")
    uploaded_image = st.file_uploader("📁 Or upload an image", type=["jpg", "jpeg", "png"])

with col2:
    result_placeholder = st.empty()

    if camera_image is not None:
        img = Image.open(camera_image)
        result_img = process_image(img)
        result_placeholder.image(result_img, caption="Processed from camera")

    elif uploaded_image is not None:
        img = Image.open(uploaded_image)
        result_img = process_image(img)
        result_placeholder.image(result_img, caption="Processed uploaded image")

    if st.button("🔁 Capture / Upload Again"):
        st.rerun()

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("About")
st.sidebar.info("""
This app detects hand signs using a trained CNN model.
- Use sign_language_model.h5 and class_indices.json in the same directory
- Make sure images resemble training data
""")
st.sidebar.header("Classes")
st.sidebar.write(", ".join([inv_class_indices[i] for i in range(num_classes)]))
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using TensorFlow + Streamlit")
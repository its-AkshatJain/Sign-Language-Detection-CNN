# Sign Language Real-Time Classifier

A real-time hand sign detection application built using TensorFlow and Streamlit. This app leverages a pre-trained MobileNetV2-based CNN model to classify American Sign Language (ASL) gestures captured via webcam or image uploads.

---

## Features

- Real-time webcam-based gesture recognition  
- Option to upload images for classification  
- Top-3 class predictions with confidence scores  
- Preprocessing pipeline for grayscale conversion, blurring, and JPEG compression  
- Simple and interactive user interface built with Streamlit  
- Fallback simulated predictions if the model is not found  

---

## Requirements

Install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```text
streamlit>=1.32.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
tensorflow>=2.13.0
Pillow>=10.0.0
```

---

## Project Structure

```
sign-language-app/
│
├── app.py                         # Main Streamlit application
├── class_indices.json             # Class label to character mapping
├── requirements.txt               # Python package dependencies
└── working MobileNetV2/
    └── sign_language_model.keras  # Trained Keras model
```

> **Note:** Ensure `class_indices.json` is located in the project root (same level as `app.py`), **not inside** the `working MobileNetV2` folder.

---

## How to Run the Application

1. Open a terminal and navigate to the project root directory.  
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. The app will open in your browser automatically. If not, visit http://localhost:8501.

---

## Usage Instructions

1. Allow camera access when prompted or upload an image manually.  
2. Ensure your **right hand** is inside the green box displayed on screen.  
3. Hold the gesture steadily to improve prediction accuracy.  
4. The model will output the top-3 predictions along with confidence levels.

---

## Notes

- This app uses a Keras model saved in the `.keras` format and loaded via TensorFlow.  
- The image preprocessing pipeline includes grayscale conversion, Gaussian blur, and JPEG compression to match training data.  
- If `sign_language_model.keras` is missing, the app will fall back to generating simulated predictions for demo purposes.  
- Class labels are dynamically loaded from `class_indices.json`.

---

## Troubleshooting

- **Model not found:**  
  Ensure `sign_language_model.keras` is correctly placed under `working MobileNetV2/`.  
- **Class labels missing:**  
  Ensure `class_indices.json` is present in the project root.  
- **Camera not working:**  
  Try refreshing the browser or use the image-upload option instead.

---

## Acknowledgments

- Developed using [TensorFlow](https://www.tensorflow.org/), [OpenCV](https://opencv.org/), and [Streamlit](https://streamlit.io/)  
- Inspired by ASL gesture recognition and real-time computer vision projects  
```

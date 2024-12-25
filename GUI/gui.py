import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
import data_creation as d

# Custom metrics registration (same as before)
@register_keras_serializable()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

@register_keras_serializable()
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

@register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Timer class (ensure this is defined before main())
class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            raise ValueError("Timer is not running. Use .start() first.")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return f"{elapsed_time:0.4f} seconds"

# Load model components
l = os.getcwd()
order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
         'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
         'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
         'sscr', 'urlIsLive', 'urlLength']

# Existing functions for loading resources and making predictions
def load_resources():
    encoder = LabelEncoder()
    encoder.classes_ = np.load(os.path.join(l, 'GUI/lblenc.npy'), allow_pickle=True)
    scaler = pickle.load(open(os.path.join(l, 'GUI/scaler.sav'), 'rb'))
    return encoder, scaler

def make_prediction(user_input, model_type, scaler, encoder):
    features = d.UrlFeaturizer(user_input).run()
    test = [features[i] for i in order]

    if model_type == 'TF':
        model = load_model(os.path.join(l, 'GUI/Model_v1.keras'),
                           custom_objects={"f1_m": f1_m, "precision_m": precision_m, "recall_m": recall_m})
        predicted = np.argmax(model.predict(scaler.transform([test])), axis=1)
    else:
        interpreter = tf.lite.Interpreter(model_path=os.path.join(l, "GUI/tflite_quant_model.tflite"))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        test_np = np.array(test, dtype="float32").reshape(1, -1)
        interpreter.set_tensor(input_details[0]['index'], scaler.transform(test_np))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output_data, axis=1)

    return encoder.inverse_transform(predicted)[0], test

# Streamlit page configuration
st.set_page_config(
    page_title="URL Shield", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .prediction-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-description {
        background-color: #e9ecef;
        border-radius: 5px;
        padding: 15px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Title with custom styling
st.markdown('<h1 class="main-title">🛡️ URL Shield: Intelligent Web Protection</h1>', unsafe_allow_html=True)

# URL Descriptions
URL_DESCRIPTIONS = {
    "Benign": {
        "description": "Safe and trustworthy URLs that pose no known threats.",
        "color": "green",
        "icon": "✅"
    },
    "Spam": {
        "description": "Unwanted URLs designed to distribute unsolicited content or advertisements.",
        "color": "orange", 
        "icon": "🚨"
    },
    "Defacement": {
        "description": "Malicious URLs that alter the appearance of legitimate websites.",
        "color": "red",
        "icon": "🖥️"
    },
    "Malware": {
        "description": "URLs containing harmful software intended to damage or gain unauthorized access.",
        "color": "darkred",
        "icon": "🦠"
    },
    "Phishing": {
        "description": "Deceptive URLs attempting to steal sensitive personal information.",
        "color": "purple",
        "icon": "🎣"
    }
}

# Sidebar for model and input
with st.sidebar:
    st.header("🔧 Configuration")
    model_type = st.radio("Select Model Type", 
        ['TF', 'TF-Lite'], 
        help="Choose between TensorFlow and TensorFlow Lite models"
    )
    
    st.divider()
    
    st.header("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "92%")
    with col2:
        st.metric("Precision", "89%")
    with col3:
        st.metric("Recall", "90%")

# Main application logic
def main():
    # Load resources
    encoder, scaler = load_resources()

    # URL Input with improved validation
    user_input = st.text_input(
        "Enter URL", 
        placeholder="https://example.com", 
        help="Paste a complete URL to analyze its potential risk"
    )

    # Validation and Prediction
    if st.button('Analyze URL 🕵️', type='primary'):
        if not user_input:
            st.error("Please enter a valid URL!")
            return

        try:
            # Start timer
            t = Timer()
            t.start()

            # Make prediction
            pred_label, test = make_prediction(user_input, model_type, scaler, encoder)
            
            # Stop timer
            prediction_time = t.stop()

            # URL Type Card
            st.markdown(f'<div class="prediction-card">', unsafe_allow_html=True)
            
            # Dynamic URL type display
            url_info = URL_DESCRIPTIONS.get(pred_label, {})
            st.markdown(f"""
                ## {url_info.get('icon', '')} URL Type: **{pred_label}**
                <p style="color:{url_info.get('color', 'black')}">
                {url_info.get('description', 'Unknown URL type')}
                </p>
            """, unsafe_allow_html=True)

            # Prediction metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Used", model_type)
            with col2:
                st.metric("Analysis Time", prediction_time)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Feature Visualization
            st.markdown('<h2 class="subheader">Feature Analysis</h2>', unsafe_allow_html=True)
            
            # Feature comparison plot
            ben = [1, 1, 1, 1, 0.56, 0, 1, 0, 0.58, 1, 1, 0.16, 1, 0.16, 1, 1, 1, 0.95, 1, 0, 0.71]
            plt.figure(figsize=(12, 8))
            plt.plot(scaler.transform([test])[0], order, color='red', marker='>', linestyle=":", alpha=0.7, label="Extracted Features")
            plt.plot(ben, order, color='blue', marker='o', linestyle="--", alpha=0.7, label="Average Safe URL")
            plt.xlabel("Normalized Mean Values")
            plt.ylabel("Features")
            plt.title("Feature Comparison")
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Error analyzing URL: {e}")

# Run the main application
if __name__ == "__main__":
    main()
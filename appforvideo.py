import sys
import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor
import tempfile
import time
import threading

# Streamlit app
st.title("Deepfake Video Detection using ViT")

# Load the model and processor
@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k', 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load("vit_deepfake_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_processor():
    return ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Frame extraction function
def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        if len(frames) == num_frames:
            break
    cap.release()
    return frames

# Prediction function
def predict_video(video_path, model, processor):
    frames = extract_frames(video_path)
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
    preds = torch.argmax(outputs, dim=1).numpy()
    return preds.mean(), preds

# Plot histogram
def plot_confidence_histogram(preds):
    fig, ax = plt.subplots()
    ax.hist(preds, bins=10, color='blue', alpha=0.7)
    ax.set_title("Prediction Confidence Distribution")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Load the model and processor
model = load_model()
processor = load_processor()

# Step 1: Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# This function will run the prediction asynchronously in a thread
def run_prediction(temp_file_path, result_container):
    st.write("Running the prediction...")
    prediction_score, preds = predict_video(temp_file_path, model, processor)
    
    # Display the result asynchronously
    with result_container:
        st.write(f"Prediction Score (mean): {prediction_score:.2f}")
        plot_confidence_histogram(preds)

        # Determine if the video is likely fake or real
        if prediction_score > 0.5:
            st.error("The video is likely **fake**!")
        else:
            st.success("The video is likely **real**.")

if uploaded_file is not None:
    # Step 2: Play the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Step 3: Display the video
    st.video(temp_file_path)

    # Step 4: Add a Predict button to trigger the prediction process asynchronously
    result_container = st.empty()

    if st.button("Predict"):
        # Start the prediction in a new thread to avoid blocking the UI
        threading.Thread(target=run_prediction, args=(temp_file_path, result_container)).start()

        # Provide feedback while prediction is happening
        with result_container:
            st.write("Processing the video, please wait...")

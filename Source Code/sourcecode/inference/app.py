import streamlit as st
import numpy as np
import torch
import librosa
from sklearn.preprocessing import LabelEncoder
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import snntorch as snn
import torch.nn as nn

beta = 0.95

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 128)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(128, 64)
        self.lif2 = snn.Leaky(beta=beta)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.to(device)
        batch_size, time_steps, _ = x.shape
        spk2_out = 0
        mem1 = None
        mem2 = None
        for t in range(time_steps):
            input_t = x[:, t, :]
            cur1 = self.fc1(input_t)
            if mem1 is None:
                mem1 = torch.zeros_like(cur1, device=device)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            if mem2 is None:
                mem2 = torch.zeros_like(cur2, device=device)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_out += spk2
        out = self.fc3(spk2_out / time_steps)
        return out

# Load model and label encoder
model = SNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

label_encoder = joblib.load("label_encoder.pkl")  

# ===== Feature extraction functions =====
def extract_features(data, sample_rate):
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
           (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
    return mfcc.T  # Shape: (time_steps, 40)

def pad_feature_single(feat, max_len=200):
    if feat.shape[0] > max_len:
        feat = feat[:max_len, :]
    else:
        pad_width = max_len - feat.shape[0]
        feat = np.pad(feat, ((0, pad_width), (0, 0)), mode='constant')
    return feat

def predict_emotion(path):
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    feat = extract_features(data, sr)
    feat = pad_feature_single(feat)
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(feat_tensor)
        _, predicted = torch.max(outputs.data, 1)
    pred_idx = predicted.item()
    return label_encoder.inverse_transform([pred_idx])[0]

# ===== Streamlit Interface =====
st.title("🎵 Emotion Prediction from Audio")

uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        predicted_emotion = predict_emotion("temp.wav")
        st.success(f"🎯 Predicted Emotion: **{predicted_emotion}**")
    except Exception as e:
        st.error(f"Error: {e}")

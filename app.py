import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# Class names
class_names = ["fake", "real"]

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("fake_detector.pth", map_location="cpu"))
model.eval()

# UI
st.title("Fake vs Real Image Detector")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    pred = model(transform(image).unsqueeze(0)).argmax(1).item()
    st.write(f"Prediction: **{class_names[pred]}**")

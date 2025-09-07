import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from pathlib import Path

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="SolarGuard: Solar Panel Health Monitor", layout="wide")

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
/* Make tab font bigger */
.css-1v3fvcr.e16nr0p30 {  /* class for tab label container */
    font-size: 22px !important;
    font-weight: bold;
}
/* Fullscreen home page overlay */
.home-container {
    position: relative;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}
.home-text {
    font-family: "Comic Sans MS", cursive, sans-serif;
    text-align: center;
}
.home-text h1 {
    font-size: 60px;
    color: #FF8C00;
}
.home-text p {
    font-size: 24px;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# TAB SETUP
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🖼️ Classification", "🚀 Obstruction Detection"])

# ----------------------------
# HOME PAGE
# ----------------------------
with tab1:
    # Text first
    st.markdown(
        """
        <div class="home-text">
            <h1>🌞 SolarGuard: AI-Powered Solar Panel Health Monitor</h1>
            <p>Intelligent defect detection and obstruction monitoring for solar panels</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    home_img_path = Path("Homepage.jpg")
    if home_img_path.exists():
        home_img = Image.open(home_img_path)
        st.image(home_img, use_container_width=True)
    else:
        st.warning("Homepage image 'Homepage.jpg' not found.")

# ----------------------------
# CLASSIFICATION
# ----------------------------
with tab2:
    # Same heading as homepage
    st.markdown(
        """
        <div class="home-text">
            <h1>🌞 SolarGuard: AI-Powered Solar Panel Health Monitor</h1>
            <p>Intelligent defect detection and obstruction monitoring for solar panels</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.header("Solar Panel Defect Classification")
    uploaded_file = st.file_uploader("Upload a solar panel image", type=None)

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Load MobileNet model
        num_classes = 6
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = Path("MobileNet.pth")
        if model_path.exists():
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Transform image
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(img).unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1)

            classes = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
            st.success(f"Predicted Class: **{classes[pred.item()]}**")
        else:
            st.error("Classification model MobileNet.pth not found!")

# ----------------------------
# OBJECT DETECTION
# ----------------------------
with tab3:
    # Same heading as homepage
    st.markdown(
        """
        <div class="home-text">
            <h1>🌞 SolarGuard: AI-Powered Solar Panel Health Monitor</h1>
            <p>Intelligent defect detection and obstruction monitoring for solar panels</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.header("Solar Panel Obstruction Detection (YOLO)")
    uploaded_file_obj = st.file_uploader("Upload a solar panel image for detection", type=None, key="obj_det")

    if uploaded_file_obj is not None:
        img_obj = Image.open(uploaded_file_obj).convert('RGB')
        st.image(img_obj, caption="Uploaded Image", use_container_width=True)

        # Load YOLO model
        yolo_path = Path("best.pt")
        if yolo_path.exists():
            from ultralytics import YOLO
            model_yolo = YOLO(str(yolo_path))

            # Run inference
            results = model_yolo.predict(img_obj, imgsz=640)

            # Display results
            for result in results:
                result_img = result.plot()
                st.image(result_img, caption="Detected Obstructions", use_container_width=True)
        else:
            st.error("Object detection model best.pt not found!")

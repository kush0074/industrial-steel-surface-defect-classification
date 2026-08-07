import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog

IMG_SIZE = 128

model = joblib.load("models/svm_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

st.set_page_config(
    page_title="Steel Surface Defect Classifier",
    page_icon="⚙️",
    layout="wide"
)
with st.sidebar:

    st.title("🏭 Project Information")

    st.metric("Training Images", "1440")

    st.metric("Validation Images", "360")

    st.metric("Classes", "6")

    st.divider()

    st.success("Algorithm : Support Vector Machine")

    st.success("Feature Extraction : HOG")

    st.success("Dataset : NEU Surface Defect")

st.title("🏭 Industrial Steel Surface Defect Classification")

st.markdown("""
This application classifies **steel surface defects** using
traditional computer vision techniques.

### Pipeline

- 📷 Image Upload
- 🔍 Histogram of Oriented Gradients (HOG)
- 🤖 Support Vector Machine (SVM)
- 📊 Defect Prediction
""")

uploaded_file = st.file_uploader(
    "Upload Steel Surface Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm="L2-Hys"
    )

    prediction = model.predict([hog_features])

    prediction = encoder.inverse_transform(prediction)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.success(
            prediction[0].replace("_", " ").title()
        )

        st.metric(
            "Validation Accuracy",
            "78.06%"
        )

        st.info("Model : Support Vector Machine")

        st.info("Feature Extraction : HOG")
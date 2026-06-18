import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import joblib
import pytesseract

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load Model
model = joblib.load("random_forest_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Advertisement Intelligence System",
    layout="wide"
)

st.title("📢 Advertisement Intelligence System")

uploaded_file = st.file_uploader(
    "Upload Advertisement Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # -------------------------
    # DISPLAY IMAGE
    # -------------------------

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Advertisement",
        use_container_width=True
    )

    img = np.array(image)

    # -------------------------
    # IMAGE FEATURES
    # -------------------------

    height, width = img.shape[:2]

    ad_area = width * height

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    brightness = np.mean(gray)

    contrast = np.std(gray)

    # Colorfulness

    (B, G, R) = cv2.split(
        img.astype("float")
    )

    rg = np.abs(R - G)

    yb = np.abs(
        0.5 * (R + G) - B
    )

    std_rg = np.std(rg)
    std_yb = np.std(yb)

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    colorfulness = (
        np.sqrt(std_rg**2 + std_yb**2)
        + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    )

    st.success("Image uploaded successfully!")

    # -------------------------
    # IMAGE FEATURES DISPLAY
    # -------------------------

    st.subheader("📊 Extracted Image Features")

    st.write(f"**Width:** {width}")
    st.write(f"**Height:** {height}")
    st.write(f"**Ad Area:** {ad_area}")
    st.write(f"**Brightness:** {brightness:.2f}")
    st.write(f"**Contrast:** {contrast:.2f}")
    st.write(f"**Colorfulness:** {colorfulness:.2f}")

    # -------------------------
    # OCR TEXT EXTRACTION
    # -------------------------

    extracted_text = pytesseract.image_to_string(image)

    if extracted_text.strip() == "":
        extracted_text = "No text detected"

    word_count = len(extracted_text.split())

    character_count = len(extracted_text)

    text_sentiment = 0

    # -------------------------
    # TEXT FEATURES DISPLAY
    # -------------------------

    st.subheader("📝 Extracted Text")

    st.write(extracted_text)

    st.subheader("📄 Text Features")

    st.write(f"**Word Count:** {word_count}")
    st.write(f"**Character Count:** {character_count}")
    st.write(f"**Text Sentiment:** {text_sentiment}")

    # -------------------------
    # MODEL INPUT
    # -------------------------

    features = pd.DataFrame({
        "Brightness": [brightness],
        "Contrast": [contrast],
        "Colorfulness": [colorfulness],
        "WordCount": [word_count],
        "CharacterCount": [character_count],
        "TextSentiment": [text_sentiment],
        "AdArea": [ad_area]
    })

    st.subheader("📋 Model Input")

    st.write(features)

    # -------------------------
    # PREDICTION
    # -------------------------

    try:

        prediction = model.predict(features)[0]

        st.subheader("🎯 Advertisement Prediction")

        st.success(
            f"Predicted Performance: {prediction}"
        )

        # -------------------------
        # RECOMMENDATION
        # -------------------------

        st.subheader("💡 Recommendation")

        if prediction == "Excellent":

            st.success(
                "Continue this campaign. Audience engagement is likely to be high."
            )

        elif prediction == "Good":

            st.info(
                "Advertisement has good potential. Minor improvements may increase performance."
            )

        else:

            st.warning(
                "Improve advertisement design, content quality and visual appeal."
            )

    except Exception as e:

        st.error(f"Prediction Error: {e}")

    # -------------------------
    # USER FEEDBACK
    # -------------------------

    st.subheader("👍 Why Users May Like This Ad")

    likes = []

    if brightness > 120:
        likes.append("Bright and visually appealing")

    if colorfulness > 40:
        likes.append("Attractive colors")

    if contrast > 50:
        likes.append("Good image quality")

    if len(likes) == 0:
        likes.append("Simple advertisement design")

    for item in likes:
        st.write("•", item)

    st.subheader("👎 Why Users May Dislike This Ad")

    dislikes = []

    if brightness < 80:
        dislikes.append("Advertisement may appear dull")

    if colorfulness < 25:
        dislikes.append("Colors may not attract attention")

    if contrast < 30:
        dislikes.append("Low visual impact")

    if len(dislikes) == 0:
        dislikes.append("No major visual issues detected")

    for item in dislikes:
        st.write("•", item)
import streamlit as st
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

st.title("Emotion Detector 😄😐😢")

# Choose input method
mode = st.radio("Select Input Mode:", ("Upload Image", "Use Webcam"))

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inject custom CSS
st.markdown(
    """
    <style>
    /* Background image */
    .stApp {
        background-image: url('https://d1sr9z1pdl3mb7.cloudfront.net/wp-content/uploads/2022/03/07162020/synthetic-data-scaled.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* Optional: Make text easier to read */
    .stApp * {
        color: white;
    }

    /* Make the sidebar semi-transparent if needed */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.6);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Font setup for PIL
def get_font(size=20):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

# Annotate image with bounding boxes and emotion labels
def annotate_image(image):
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Emotion detection
    results = DeepFace.analyze(img_bgr, actions=['emotion'], enforce_detection=False, silent=True)

    # Face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(20)

    emotion_texts = []

    for i, (x, y, w, h) in enumerate(faces):
        emotion = results[0]['dominant_emotion']
        emotion_texts.append(emotion)

        draw.rectangle([(x, y), (x+w, y+h)], outline="lime", width=3)
        draw.text((x, y-25), emotion, fill="red", font=font)

    # Add combined emotion text at the bottom
    combined_emotions = " | ".join(emotion_texts) if emotion_texts else "No face detected"
    img_width, img_height = img_pil.size
    draw.rectangle([(0, img_height - 30), (img_width, img_height)], fill="black")
    draw.text((10, img_height - 25), combined_emotions, fill="white", font=font)

    return img_pil

# Image mode
if mode == "Upload Image":
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

            if st.button("Detect Emotion"):
                with st.spinner("Analyzing..."):
                    result_img = annotate_image(image)
                    st.image(result_img, caption="Detected Emotions", width=400)
    with col2:
        print("Do Nothing")

# Webcam mode 
elif mode == "Use Webcam":
    st.warning("Click 'Start' and press 'q' to quit the webcam window.")

    if st.button("Start Webcam Detection"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = DeepFace.analyze(frame, actions=['emotion'], silent=True, enforce_detection=False)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                emotion_texts = []

                for i, (x, y, w, h) in enumerate(faces):
                    emotion = result[0]['dominant_emotion']
                    emotion_texts.append(emotion)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Print emotion at bottom
                combined_text = " | ".join(emotion_texts) if emotion_texts else "No face detected"
                cv2.rectangle(frame, (0, frame.shape[0]-30), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.putText(frame, combined_text, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("Webcam - Press 'q' to Quit", frame)

                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
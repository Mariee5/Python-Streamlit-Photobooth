import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from io import BytesIO
import os
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    DEEPFACE_ERROR = None
except Exception as e:
    # Don't crash the whole app if DeepFace (or native deps) fail to import.
    DEEPFACE_AVAILABLE = False
    DEEPFACE_ERROR = str(e)
import pandas as pd
import plotly.express as px
from datetime import datetime
from skimage import filters, exposure
import matplotlib.pyplot as plt

# Initialize session state for storing emotion data
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = []

def apply_filter(image, filter_name):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    img_float = image.astype(float) / 255.0
    
    if filter_name == "Original":
        return image
    elif filter_name == "Sepia":
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                               [0.349, 0.686, 0.168],
                               [0.272, 0.534, 0.131]])
        sepia_img = cv2.transform(image, sepia_filter)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return sepia_img
    elif filter_name == "Vintage":
        vintage = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        return vintage
    elif filter_name == "Cool":
        cool = cv2.applyColorMap(image, cv2.COLORMAP_COOL)
        return cool
    elif filter_name == "Summer":
        summer = exposure.adjust_gamma(img_float, 1.2)
        summer = exposure.adjust_sigmoid(summer, cutoff=0.5, gain=10)
        return (summer * 255).astype(np.uint8)
    elif filter_name == "Dramatic":
        dramatic = exposure.adjust_gamma(img_float, 2.0)
        dramatic = filters.unsharp_mask(dramatic, radius=2, amount=2)
        return (dramatic * 255).astype(np.uint8)
    return image

def analyze_face(image):
    if not DEEPFACE_AVAILABLE:
        st.error(
            "DeepFace is not available in the environment. "
            "This usually means a native dependency (like OpenCV) failed to load. "
            "Check the Streamlit build logs and ensure required system packages (libgl, libsm6, libxrender, libxext) are installed.\n"
            f"Import error: {DEEPFACE_ERROR}"
        )
        return None

    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        return result
    except Exception as e:
        st.error(f"Could not detect face: {str(e)}")
        return None

def update_emotion_data(emotion_dict):
    emotion_data = {
        **emotion_dict
    }
    st.session_state.emotion_data.append(emotion_data)

def plot_emotions(emotions):
    plt.figure(figsize=(10, 6))
    plt.plot(emotions)
    plt.xlabel('Frame')
    plt.ylabel('Emotion Intensity')
    plt.title('Emotion Plot')
    st.pyplot(plt)

def add_frame(image, frame_width=50):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    h, w = image.shape[:2]
    framed = np.ones((h + 2*frame_width, w + 2*frame_width, 3), dtype=np.uint8) * 255
    framed[frame_width:h+frame_width, frame_width:w+frame_width] = image
    return framed

def capture_photos(num_photos=3, delay=3, black_white=False, selected_filter="Original"):
    photos = []
    analysis_results = []
    
    # Try multiple camera indices
    for camera_index in range(3):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            break
    
    if not cap.isOpened():
        st.error("Error: Could not access any camera! Please check your camera connection.")
        st.info("If you're on Windows, try these steps:\n1. Make sure your camera is enabled\n2. Check camera permissions\n3. Try restarting your computer")
        return None, None
    
    preview_placeholder = st.empty()
    status_placeholder = st.empty()
    
    for i in range(num_photos):
        status_placeholder.info(f"📸 Getting ready for photo {i+1} of {num_photos}")
        time.sleep(1)
        
        # Countdown
        for j in range(delay, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(frame, channels="RGB", caption=f"Get ready! Taking photo in {j} seconds...")
                time.sleep(1)
            
        status_placeholder.success("SMILE! 📸 Taking photo NOW!")
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze face
            analysis = analyze_face(frame)
            if analysis:
                analysis_results.append(analysis)
                update_emotion_data(analysis['emotion'])
            
            if black_white:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Apply selected filter
            frame = apply_filter(frame, selected_filter)
            
            # Add frame
            frame = add_frame(frame)
            
            photos.append(frame)
            preview_placeholder.image(frame, channels="RGB", caption=f"Photo {i+1} captured!")
            
            if analysis:
                emotions = analysis['emotion']
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                status_placeholder.info(f"Dominant emotion in photo {i+1}: {dominant_emotion}")
            
            time.sleep(1.5)
        else:
            st.error("Failed to capture photo!")
    
    preview_placeholder.empty()
    status_placeholder.empty()
    cap.release()
    return photos, analysis_results

def create_photo_strip(photos):
    if not photos:
        return None
    
    padding = 20
    photo_height = photos[0].shape[0]
    photo_width = photos[0].shape[1]
    
    # Add extra space at bottom for date
    date_height = 40
    strip_height = (photo_height * len(photos)) + (padding * (len(photos) - 1)) + date_height
    
    strip = np.ones((strip_height, photo_width, 3), dtype=np.uint8) * 255
    
    # Add photos
    for i, photo in enumerate(photos):
        y_start = i * (photo_height + padding)
        y_end = y_start + photo_height
        strip[y_start:y_end] = photo
    
    # Convert to PIL Image to add date text
    strip_pil = Image.fromarray(strip)
    draw = ImageDraw.Draw(strip_pil)
    
    # Add date at the bottom
    current_date = datetime.now().strftime("%B %d, %Y")
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text size and center it
    text_bbox = draw.textbbox((0, 0), current_date, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (photo_width - text_width) // 2
    text_y = strip_height - date_height + 10
    
    draw.text((text_x, text_y), current_date, fill=(100, 100, 100), font=font)
    
    return np.array(strip_pil)

def main():
    st.set_page_config(
        page_title="AI Photobooth",
        page_icon="📸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set modern dark theme with teal accents
    st.markdown("""
        <style>
        /* Main content background */
        [data-testid="stAppViewContainer"] {
            background-color: #1e1e1e;  /* Dark gray */
        }
        
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #121212;  /* Darker gray */
        }
        
        /* Text colors */
        h1, h2, h3, p, li, label, div {
            color: #ffffff !important;  /* White text */
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #00bcd4;  /* Teal */
            color: white;
            font-size: 20px;
            padding: 20px 40px;
            border-radius: 10px;
            border: none;
            transition: background-color 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #0097a7;  /* Darker teal on hover */
        }
        
        /* Sidebar text */
        .sidebar .sidebar-content {
            color: #ffffff;  /* White text for sidebar */
        }
        
        /* Alert boxes */
        .stAlert {
            background-color: #121212;  /* Darker gray */
            color: #ffffff;  /* White text */
            border-radius: 10px;
            border: 1px solid #00bcd4;  /* Teal border */
        }
        
        /* Code blocks and other elements */
        code {
            color: #00bcd4 !important;  /* Teal code text */
        }
        
        /* Links */
        a {
            color: #00bcd4 !important;  /* Teal links */
        }
        
        /* Dropdown and selectbox */
        .stSelectbox > div > div {
            background-color: #121212;  /* Darker gray */
            color: #ffffff;  /* White text */
            border: 1px solid #00bcd4;  /* Teal border */
            border-radius: 5px;
        }

        /* Grid styling */
        .stImage {
            margin: 5px;
            border-radius: 10px;
            border: 2px solid #00bcd4;  /* Teal border for images */
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #1e1e1e;  /* Dark gray */
        }

        ::-webkit-scrollbar-thumb {
            background: #00bcd4;  /* Teal */
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #0097a7;  /* Darker teal on hover */
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📸 Marie's PhotoBooth!")
    st.header("Welcome and get your photostrip with your friends, family or just you yourself! in just 5 mins!")
    st.markdown("""
    ### Create your perfect photo strip with AI-powered features!
    
    Take 3 consecutive photos with:
    - Real-time emotion detection 🎭
    - Instagram-style filters ✨
    - Beautiful photo frame 🖼️
    
    ### Instructions:
    1. Select your preferred filter from the sidebar
    2. Click the "Start Photoshoot!" button
    3. Strike a pose during the countdown
    4. Watch for the "SMILE!" message - that's when the photo is taken!
    5. View your emotions analysis
    6. Download your photo strip when done
    """)
    
    # Sidebar options
    st.sidebar.title("Settings")
    black_white = st.sidebar.checkbox("Black & White Mode")
    filter_options = ["Original", "Sepia", "Vintage", "Cool", "Summer", "Dramatic"]
    selected_filter = st.sidebar.selectbox("Select Filter", filter_options)
    
    # Initialize session state for storing photos and emotions
    if 'photos_with_emotions' not in st.session_state:
        st.session_state.photos_with_emotions = []
    
    # Main interface with three columns
    col1, col2, col3 = st.columns([2, 0.1, 1])
    
    with col1:
        if st.button("Start Photoshoot! 📸"):
            photos, analysis_results = capture_photos(black_white=black_white, selected_filter=selected_filter)
            
            if photos and analysis_results:
                # Store photos with their emotions
                for photo, analysis in zip(photos, analysis_results):
                    emotions = analysis['emotion']
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                    st.session_state.photos_with_emotions.append({
                        'photo': photo,
                        'emotion': dominant_emotion,
                    })
                
                st.write("Creating your photo strip...")
                photo_strip = create_photo_strip(photos)
                
                if photo_strip is not None:
                    st.write("Here's your photo strip! 🎉")
                    st.image(photo_strip, caption="Your Photo Strip")
                    
                    # Convert to PIL Image for saving
                    pil_image = Image.fromarray(photo_strip)
                    buf = BytesIO()
                    pil_image.save(buf, format="PNG")
                    
                    # Download button
                    st.download_button(
                        label="Download Photo Strip 📥",
                        data=buf.getvalue(),
                        file_name="photo_strip.png",
                        mime="image/png"
                    )
    
    with col3:
        st.subheader("📊 Photo Gallery with Emotions")
        
        if st.session_state.photos_with_emotions:
            # Create a grid layout for photos
            for i in range(0, len(st.session_state.photos_with_emotions), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(st.session_state.photos_with_emotions):
                        item = st.session_state.photos_with_emotions[i + j]
                        with cols[j]:
                            # Resize photo for thumbnail
                            photo = Image.fromarray(item['photo'])
                            photo.thumbnail((200, 200))
                            st.image(np.array(photo), caption=f"{item['emotion'].title()}")
        else:
            st.info("Take some photos to see them here!")
        
        if len(st.session_state.photos_with_emotions) > 0:
            if st.button("Clear Gallery"):
                st.session_state.photos_with_emotions = []
                st.session_state.emotion_data = []
                st.rerun()
        
        # Show emotion trends below the gallery
        st.subheader("Emotion Trends")
        if st.session_state.emotion_data:
            emotions = [list(emotion.values()) for emotion in st.session_state.emotion_data]
            plot_emotions([sum(x) for x in zip(*emotions)])

if __name__ == "__main__":
    main()
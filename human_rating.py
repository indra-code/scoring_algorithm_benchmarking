import streamlit as st
import os
import json
from PIL import Image
from streamlit_star_rating import st_star_rating

IMAGE_DIR = "images"
LABEL_FILE = "human_outputs.json"

# Load images
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
]
image_files.sort()

# Load or create label file
if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "r") as f:
        labels = json.load(f)
else:
    labels = {}

# Session state for navigation
if "index" not in st.session_state:
    st.session_state.index = 0

def save_labels():
    with open(LABEL_FILE, "w") as f:
        json.dump(labels, f, indent=4)

def go_next():
    # Save current rating before moving
    current_file = image_files[st.session_state.index]
    if "rating" in st.session_state:
        labels[current_file] = st.session_state.rating
        save_labels()
    if st.session_state.index < len(image_files) - 1:
        st.session_state.index += 1



# UI
st.title("⭐ Human Footpath Rating Tool — Star Edition ⭐")

if not image_files:
    st.warning("No images found in directory.")
    st.stop()

current_file = image_files[st.session_state.index]

st.subheader(f"Image {st.session_state.index + 1} / {len(image_files)}")
st.text(f"Filename: {current_file}")

# Show the image
img = Image.open(os.path.join(IMAGE_DIR, current_file))
st.image(img, use_container_width=True)

# Check if already rated
if current_file in labels:
    st.success(f"✅ Already rated: {labels[current_file]} stars")
    st.write("### This image has been rated. Click Next to continue.")
else:
    # ⭐ Animated Star Rating Input
    st.write("### Rate the Footpath (1-5 stars) ⭐")
    stars = st_star_rating(maxValue = 5, defaultValue = 3, key = "rating", dark_theme = True )

# Navigation Button
if st.button("Next ➡", use_container_width=True):
    go_next()
    st.rerun()

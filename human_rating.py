import streamlit as st
import os
import json
from PIL import Image

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
    # ⭐ Half-Star Rating Input using slider
    st.write("### Rate the Footpath (0.5 to 5 stars, in 0.5 increments) ⭐")
    
    # Create a slider with 0.5 increments
    stars = st.slider(
        label="Select rating",
        min_value=0.5,
        max_value=5.0,
        value=3.0,
        step=0.5,
        key="rating"
    )
    
    # Display the rating visually
    full_stars = int(stars)
    half_star = (stars % 1) >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    star_display = "★" * full_stars + ("⯨" if half_star else "") + "☆" * empty_stars
    st.markdown(f"### {star_display} ({stars} stars)")

# Navigation Button
if st.button("Next ➡", use_container_width=True):
    go_next()
    st.rerun()

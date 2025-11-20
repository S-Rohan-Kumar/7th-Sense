import os

# --- HARDWARE ---
# Use '0' for webcam, or your IP Webcam URL
# CAMERA_SOURCE = "http://192.168.1.3:8080/video"
CAMERA_SOURCE = "http://192.168.1.5:8080/video"
# CAMERA_SOURCE = 0
USE_GPU = True  # Set to True for your RTX 3050

# --- AI ---
GEMINI_API_KEY = "AIzaSyDKZOhXoR5xRLgaALNRSJbG5yHZSq5KLSk" # PASTE KEY HERE
YOLO_MODEL_PATH = "yolov8l.pt"

# --- THRESHOLDS ---
CONFIDENCE_THRESHOLD = 0.5
DANGER_CLASSES = [2, 3, 5, 7, 67, 39]  # Car, Motorcycle, Bus, Truck, Cell Phone , Bottle
SAFE_CLASSES = [0, 56, 57]     # Person, Chair, Couch
BRIGHTNESS_TRIGGER = 30        # Low light trigger for Gemini

# --- AUDIO PATHS ---
AUDIO_DIR = "audio"
SOUNDS = {
    "beep": os.path.join(AUDIO_DIR, "beep_fast.wav"),
    "siren": os.path.join(AUDIO_DIR, "siren.wav"),
    "shutter": os.path.join(AUDIO_DIR, "click.wav")
}
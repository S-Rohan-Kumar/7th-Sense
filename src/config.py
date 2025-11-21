import os
from dotenv import load_dotenv

load_dotenv()
# --- HARDWARE ---
# Use '0' for webcam, or your IP Webcam URL
# CAMERA_SOURCE = "http://172.18.16.204:8080/video"
# CAMERA_SOURCE = "http://192.168.1.5:8080/video"
#CAMERA_SOURCE = "http://100.124.59.78:8080/video"
CAMERA_SOURCE = 0
USE_GPU = True  # Set to True for your RTX 3050

# --- AI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOLO_MODEL_PATH = "yolov8l.pt"

ORS_API_KEY = os.getenv("ORS_API_KEY")
DEMO_ORIGIN_COORDS = (77.534, 12.935)
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
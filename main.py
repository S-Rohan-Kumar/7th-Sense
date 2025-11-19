import cv2
import numpy as np
print("Imports loaded. Initializing YOLO...") 
from ultralytics import YOLO
import pygame
import pyttsx3
import threading
import time
import google.generativeai as genai
import os
import subprocess

# ==========================================
# CONFIGURATION
# ==========================================
CAMERA_SOURCE = "http://192.168.1.3:8080/video" 
GEMINI_API_KEY = "AIzaSyDKZOhXoR5xRLgaALNRSJbG5yHZSq5KLSk"

# Thresholds
DANGER_CLASSES = [2, 3, 5, 7]  # YOLO IDs: Car(2), Motorcycle(3), Bus(5), Truck(7)
SAFE_CLASSES = [0, 56, 57]     # Person(0), Chair(56), Couch(57)
BRIGHTNESS_THRESHOLD = 30      # If average pixel value < 30, trigger Gemini
SONAR_MAX_INTERVAL = 1.0       # Slowest beep (seconds)
SONAR_MIN_INTERVAL = 0.1       # Fastest beep (seconds)

# ==========================================
# SYSTEM SETUP
# ==========================================

# 1. Setup Gemini (With Robust Fallback)
print("Initializing Gemini AI...")
genai.configure(api_key=GEMINI_API_KEY)

model_gemini = None
# List of likely model names to try in order
possible_models = ['gemini-1.5-flash-latest', 'gemini-1.5-flash', 'gemini-1.5-flash-001', 'gemini-pro-vision']

for model_name in possible_models:
    try:
        print(f"Attempting to load model: {model_name}...")
        # Test load
        model = genai.GenerativeModel(model_name)
        model_gemini = model
        print(f"Success! Using model: {model_name}")
        break
    except Exception:
        continue

if model_gemini is None:
    print("\nCRITICAL WARNING: Could not auto-connect to a Gemini Vision model.")
    print("Printing available models for your API Key:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except:
        print("Could not list models. Check API Key.")
    print("AI Description feature will be DISABLED.\n")

# 2. Setup YOLO
print("Loading YOLOv8 Nano model...")
try:
    model_yolo = YOLO('yolov8n.pt')
    print("YOLO Loaded.")
except Exception as e:
    print(f"Error loading YOLO: {e}")

# 3. Setup Audio Engine (Pygame)
print("Initializing Audio...")
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def generate_sound(frequency=440, duration=0.1, wave_type='sine'):
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    if wave_type == 'sine':
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'sawtooth':
        wave = 0.5 * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
        
    audio = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

beep_sound = generate_sound(440, 0.1, 'sine')
siren_sound = generate_sound(880, 0.3, 'sawtooth')

# 4. Setup TTS
print("Initializing TTS Engine...")
is_speaking = False

def run_tts(text):
    global is_speaking
    if is_speaking: return
    
    def _speak():
        global is_speaking
        is_speaking = True
        try:
            engine = pyttsx3.init() 
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        is_speaking = False

    threading.Thread(target=_speak).start()

def trigger_haptic():
    try:
        subprocess.Popen(
            ["adb", "shell", "cmd", "vibrator", "vibrate", "200"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        pass

# ==========================================
# MAIN LOOP
# ==========================================

def main():
    print(f"Connecting to Camera: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

    if not cap.isOpened():
        print("ERROR: Could not connect to camera. Check URL.")
        return

    last_beep_time = time.time()
    ai_cooldown = 0

    print("SixthSense System Active. Press 'q' to quit.")

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame error - skipping...")
                time.sleep(0.1) # wait briefly if stream stutters
                continue

            # Resize for performance
            frame = cv2.resize(frame, (640, 640))
            height, width, _ = frame.shape

            # 1. CHECK LIGHT LEVEL (Gemini Trigger)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)

            if avg_brightness < BRIGHTNESS_THRESHOLD and time.time() > ai_cooldown:
                if model_gemini:
                    print("Lens covered! Triggering AI Brain...")
                    run_tts("Analyzing scene.")
                    
                    cv2.imwrite("temp_vision.jpg", frame)
                    
                    try:
                        # Using Pillow is often safer than file upload for quick tests,
                        # but sticking to file upload API as requested:
                        img_file = genai.upload_file(path="temp_vision.jpg", display_name="Vision Input")
                        
                        response = model_gemini.generate_content([
                            "I am blind. In one very short sentence, tell me what is directly in front of me and if it is safe.", 
                            img_file
                        ])
                        
                        text_resp = response.text if response.text else "I saw something but couldn't describe it."
                        print(f"Gemini: {text_resp}")
                        run_tts(text_resp)
                        
                    except Exception as e:
                        print(f"API Error: {e}")
                        run_tts("Connection error.")
                    
                    ai_cooldown = time.time() + 5.0 
                    continue 
                else:
                    print("AI Triggered but no model loaded.")
                    ai_cooldown = time.time() + 5.0

            # 2. OBJECT DETECTION (YOLO)
            results = model_yolo(frame, verbose=False, stream=True)
            
            danger_detected = False
            closest_obj_area = 0
            closest_obj_center = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf < 0.5: continue 

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    center_x = (x1 + x2) // 2
                    
                    color = (0, 255, 0)
                    
                    if cls_id in DANGER_CLASSES:
                        color = (0, 0, 255)
                        if width * 0.25 < center_x < width * 0.75:
                            danger_detected = True
                            cv2.putText(frame, "DANGER!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                    if area > closest_obj_area:
                        closest_obj_area = area
                        closest_obj_center = center_x

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 3. FEEDBACK SYSTEM
            current_time = time.time()

            if danger_detected:
                if current_time - last_beep_time > 0.5:
                    siren_sound.play()
                    trigger_haptic()
                    last_beep_time = current_time
            
            elif closest_obj_area > 0:
                pan = (closest_obj_center - (width / 2)) / (width / 2)
                left_vol = 1.0 - max(0, pan)
                right_vol = 1.0 + min(0, pan)
                
                closeness = min(closest_obj_area / (width * height * 0.5), 1.0)
                interval = SONAR_MAX_INTERVAL - (closeness * (SONAR_MAX_INTERVAL - SONAR_MIN_INTERVAL))
                
                if current_time - last_beep_time > interval:
                    channel = beep_sound.play()
                    if channel:
                        channel.set_volume(left_vol, right_vol)
                    last_beep_time = current_time

            cv2.imshow("SixthSense Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as main_e:
            print(f"Main Loop Error: {main_e}")
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
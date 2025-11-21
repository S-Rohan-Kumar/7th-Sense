import cv2
import time
import numpy as np

from vision_stream import VisionStream
from danger_engine import DangerEngine
from audio_manager import AudioManager
from context_engine import ContextEngine
from config import BRIGHTNESS_TRIGGER

def main():
    print("[Init] Starting Vision Stream...")
    vision = VisionStream().start()
    time.sleep(1.0) 

    print("[Init] Loading Engines...")
    danger_ai = DangerEngine()
    audio = AudioManager()
    audio.start()
    context_ai = ContextEngine(tts_callback=audio.speak)

    print("\n=== SIXTHSENSE ONLINE ===")
    audio.speak("System Online.")

    # --- STATE MEMORY ---
    last_danger_time = 0
    DANGER_HOLD_DURATION = 1.0 
    
    # --- CONTEXT TRIGGER MEMORY ---
    darkness_start_time = 0
    is_dark_state = False
    TRIGGER_DURATION = 2.0 # Must cover lens for 2 seconds

    try:
        while True:
            # 1. Get Frame
            frame = vision.read()
            if frame is None: continue

            inf_frame = cv2.resize(frame, (640, 640))
            height, width = inf_frame.shape[:2]

            # 2. CONTEXT TRIGGER LOGIC (Lens Cover)
            gray_avg = np.mean(cv2.cvtColor(inf_frame, cv2.COLOR_BGR2GRAY))
            
            if gray_avg < BRIGHTNESS_TRIGGER:
                if not is_dark_state:
                    # First moment of darkness
                    darkness_start_time = time.time()
                    is_dark_state = True
                
                # Check how long it has been dark
                elapsed = time.time() - darkness_start_time
                
                if elapsed > TRIGGER_DURATION:
                    # TRIGGER!
                    audio.silence()
                    audio.speak("Scanning.") # Feedback that trigger worked
                    
                    # Capture the frame *now* (better quality than the dark one)
                    # Actually, if lens is covered, the frame is black.
                    # TRICK: We need the user to UNCOVER the lens to take the picture?
                    # OR: We assume they point the camera at the object and THEN cover it?
                    # Usually "Cover to Trigger" implies:
                    # 1. Cover lens (Trigger mode)
                    # 2. Uncover lens (Take photo)
                    
                    # Let's stick to your requirement: "Activates when closed... takes input"
                    # If the lens is covered, the image is black. Gemini can't see it.
                    # LOGIC CHANGE: 
                    # We wait for them to UNCOVER the lens to snap the photo.
                    
                    audio.speak("Remove hand to capture.")
                    
                    # Wait for light to return
                    while True:
                        frame = vision.read()
                        if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > BRIGHTNESS_TRIGGER:
                            break
                        time.sleep(0.1)
                    
                    # Snap photo immediately after uncovering
                    audio.speak("Capturing.")
                    context_ai.describe_scene(frame)
                    
                    # Reset logic
                    is_dark_state = False
                    darkness_start_time = 0
                    time.sleep(3.0) # Cooldown so it doesn't double-trigger
                    continue
            else:
                # Reset if they uncover too early (aborted trigger)
                is_dark_state = False
                darkness_start_time = 0


            # 3. Danger Analysis (Standard Loop)
            is_danger, danger_name, closest_obj = danger_ai.analyze(inf_frame)

            # ... (Rest of your Danger Logic remains same) ...
            
            # 1. DANGER CHECK (Priority)
            current_time = time.time()
            if is_danger: last_danger_time = current_time
            in_danger_mode = (current_time - last_danger_time) < DANGER_HOLD_DURATION

            if in_danger_mode:
                coverage = 0
                pan = 0
                if closest_obj:
                    max_area = width * height
                    coverage = closest_obj['area'] / max_area
                    pan = (closest_obj['center_x'] - (width/2)) / (width/2)
                
                label = danger_name if is_danger else "DANGER"
                
                if coverage > 0.40:
                    audio.set_danger_critical(pan)
                    cv2.putText(inf_frame, f"CRITICAL: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                elif coverage > 0.15:
                    audio.set_danger_approaching(pan, label)
                    cv2.putText(inf_frame, f"WARNING: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                else:
                    audio.set_danger_far(pan)
                    cv2.putText(inf_frame, f"DETECTED: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            elif closest_obj:
                pan = (closest_obj['center_x'] - (width/2)) / (width/2)
                max_area = width * height
                coverage = closest_obj['area'] / max_area 
                x1,y1,x2,y2 = closest_obj['box']
                cv2.rectangle(inf_frame, (x1,y1), (x2,y2), (0,255,0), 2)

                if coverage > 0.40:
                    audio.announce_proximity(closest_obj['label'], pan)
                    cv2.putText(inf_frame, f"CLOSE: {closest_obj['label']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                else:
                    audio.silence()
            else:
                audio.silence()

            cv2.imshow("SixthSense Brain", inf_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        vision.stop()
        audio.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
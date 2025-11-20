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

    try:
        while True:
            frame = vision.read()
            if frame is None: continue

            inf_frame = cv2.resize(frame, (640, 640))
            height, width = inf_frame.shape[:2]

            # Context Trigger
            if np.mean(cv2.cvtColor(inf_frame, cv2.COLOR_BGR2GRAY)) < BRIGHTNESS_TRIGGER:
                audio.silence()
                audio.speak("Thinking.")
                context_ai.describe_scene(inf_frame)
                time.sleep(2.0)
                continue

            is_danger, danger_name, closest_obj = danger_ai.analyze(inf_frame)

            # --- LOGIC FLOW ---
            
            # 1. DANGER CHECK (Priority)
            current_time = time.time()
            if is_danger: last_danger_time = current_time
            in_danger_mode = (current_time - last_danger_time) < DANGER_HOLD_DURATION

            if in_danger_mode:
                # Calculate Coverage for Distance even in Danger Mode
                coverage = 0
                if closest_obj:
                    max_area = width * height
                    coverage = closest_obj['area'] / max_area
                
                label = danger_name if is_danger else "DANGER"
                
                # Check Range: 0.5m - 0.7m (Coverage roughly 0.15 to 0.40)
                if 0.15 < coverage <= 0.40:
                    # Danger Approaching: Siren + Voice
                    audio.set_hazard_approaching(label)
                    cv2.putText(inf_frame, f"WARNING: {label}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                else:
                    # Too Close (<0.5m) or Far (>0.7m): Siren Only
                    audio.set_hazard_mode()
                    cv2.putText(inf_frame, f"HAZARD: {label}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            elif closest_obj:
                # SAFE OBJECTS: Voice ONLY (and only when close)
                pan = (closest_obj['center_x'] - (width/2)) / (width/2)
                max_area = width * height
                coverage = closest_obj['area'] / max_area 
                
                x1,y1,x2,y2 = closest_obj['box']
                cv2.rectangle(inf_frame, (x1,y1), (x2,y2), (0,255,0), 2)

                # Check Distance (< 0.5m is roughly > 40% screen coverage)
                if coverage > 0.40:
                    audio.announce_proximity(closest_obj['label'], pan)
                    cv2.putText(inf_frame, f"CLOSE: {closest_obj['label']}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
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
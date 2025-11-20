import cv2
import time
import numpy as np

from vision_stream import VisionStream
from danger_engine import DangerEngine
from audio_manager import AudioManager
from context_engine import ContextEngine
from config import BRIGHTNESS_TRIGGER

def main():
    # 1. Init Modules
    print("[Init] Starting Vision Stream...")
    vision = VisionStream().start()
    time.sleep(1.0) # Warmup

    print("[Init] Loading Engines...")
    danger_ai = DangerEngine()
    
    # Init Audio and Start the Stream (Real-time synth)
    audio = AudioManager()
    audio.start()
    
    # Context engine needs audio to speak back
    context_ai = ContextEngine(tts_callback=audio.speak)

    print("\n=== SIXTHSENSE ONLINE ===")
    print("Press 'q' to quit.")
    audio.speak("System Online.")

    try:
        while True:
            # 1. Get Freshest Frame (Non-blocking)
            frame = vision.read()
            if frame is None: continue

            # 2. Resize for Inference Speed
            inf_frame = cv2.resize(frame, (640, 640))
            height, width = inf_frame.shape[:2]

            # 3. Context Trigger (Lens Cover)
            gray = cv2.cvtColor(inf_frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < BRIGHTNESS_TRIGGER:
                audio.silence() # Stop beeping while thinking
                audio.speak("Thinking.")
                context_ai.describe_scene(inf_frame)
                time.sleep(2.0) # Debounce
                continue

            # 4. Danger & Depth Inference
            is_danger, danger_name, closest_obj = danger_ai.analyze(inf_frame)

            # 5. Audio Feedback Logic
            if is_danger:
                # PRIORITY 1: Danger (High Pitch, Fast Beep)
                audio.play_danger()
                cv2.putText(inf_frame, f"DANGER: {danger_name}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            elif closest_obj:
                # PRIORITY 2: Navigation (Variable Pitch/Speed)
                
                # Map center_x (0 to 640) -> Pan (-1.0 to 1.0)
                pan = (closest_obj['center_x'] - (width/2)) / (width/2)
                
                # Map Area (Small to Big) -> Interval (Slow to Fast)
                closeness = min(closest_obj['area'] / (width * height * 0.5), 1.0)
                
                # Interval: 0.1s (Close) to 1.0s (Far)
                interval = 1.0 - (closeness * 0.9)
                
                # Update the synthesizer
                audio.update_sonar(pan=pan, interval=interval, freq=880)
                
                # Draw Box
                x1,y1,x2,y2 = closest_obj['box']
                cv2.rectangle(inf_frame, (x1,y1), (x2,y2), (0,255,0), 2)
            
            else:
                # PRIORITY 3: Silence (No objects)
                audio.silence()

            # 6. Debug UI
            cv2.imshow("SixthSense Brain", inf_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        vision.stop()
        audio.stop() # Stop the audio stream
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import time
import numpy as np
import pyaudio
import wave
import os

from vision_stream import VisionStream
from danger_engine import DangerEngine
from audio_manager import AudioManager
from context_engine import ContextEngine
from config import BRIGHTNESS_TRIGGER

# --- AUDIO RECORDING CONFIG ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "user_query.wav"

def record_audio_input():
    """
    Records audio until silence is detected OR a timeout is reached.
    Dynamic stopping for faster response times.
    """
    p = pyaudio.PyAudio()
    
    # --- SILENCE DETECTION CONFIG ---
    # Increase THRESHOLD if the system cuts off too early in noisy rooms
    THRESHOLD = 600       
    SILENCE_LIMIT = 1.2   # Seconds of silence to wait before stopping
    MAX_DURATION = 10.0   # Hard limit
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("[System] Listening... (Speak now)")
        frames = []
        
        start_time = time.time()
        last_sound_time = time.time()
        speech_started = False
        
        while True:
            # Read data
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Convert buffer to numpy array to check loudness
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            
            current_time = time.time()
            total_duration = current_time - start_time
            
            # 1. Check for Speech (Activity)
            if volume > THRESHOLD:
                last_sound_time = current_time
                if not speech_started:
                    speech_started = True
                    print("[System] Speech detected...")

            # 2. Check for Silence (Stop Condition)
            if speech_started:
                silence_duration = current_time - last_sound_time
                if silence_duration > SILENCE_LIMIT:
                    print(f"[System] Silence detected ({SILENCE_LIMIT}s). Stopping.")
                    break
            
            # 3. Hard Timeout (Safety)
            if total_duration > MAX_DURATION:
                print("[System] Max duration reached.")
                break
                
            # 4. Initial Timeout (If no one speaks at all for 4 seconds)
            if not speech_started and total_duration > 4.0:
                 print("[System] No speech detected.")
                 break

        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Only save if we actually captured speech
        if len(frames) > 0 and speech_started:
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            return True
        return False

    except Exception as e:
        print(f"[Audio Error] Could not record: {e}")
        return False

def main():
    print("[Init] Starting Vision Stream...")
    vision = VisionStream().start()
    time.sleep(1.0) 

    print("[Init] Loading Engines...")
    danger_ai = DangerEngine()
    audio = AudioManager()
    audio.start()
    
    # Initialize the Context Engine
    context_ai = ContextEngine(tts_callback=audio.speak)

    print("\n=== SIXTHSENSE ONLINE ===")
    audio.speak("System Online.")

    # --- STATE MEMORY ---
    last_danger_time = 0
    DANGER_HOLD_DURATION = 1.0 
    
    # --- CONTEXT TRIGGER MEMORY ---
    darkness_start_time = 0
    is_dark_state = False
    TRIGGER_DURATION = 2.0 

    try:
        while True:
            # 1. Get Frame
            frame = vision.read()
            if frame is None: continue

            inf_frame = cv2.resize(frame, (640, 640))
            height, width = inf_frame.shape[:2]

            # ==================================================
            # 2. CONTEXT TRIGGER LOGIC (The "Ask" Mode)
            # ==================================================
            gray_avg = np.mean(cv2.cvtColor(inf_frame, cv2.COLOR_BGR2GRAY))
            
            if gray_avg < BRIGHTNESS_TRIGGER:
                if not is_dark_state:
                    darkness_start_time = time.time()
                    is_dark_state = True
                
                elapsed = time.time() - darkness_start_time
                
                if elapsed > TRIGGER_DURATION:
                    # --- TRIGGER ACTIVATED ---
                    audio.silence() # Stop any danger noises
                    audio.speak("Ready. Remove hand.")
                    
                    # A. Wait for Uncover (Just wait for light, DON'T capture yet)
                    while True:
                        temp = vision.read()
                        if temp is None: continue
                        if np.mean(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)) > BRIGHTNESS_TRIGGER:
                            # User has removed hand, proceed
                            break
                        time.sleep(0.05)
                    
                    # B. Prompt User & Record (Dynamic Silence Detection)
                    audio.speak("Listening.")
                    success = record_audio_input()
                    
                    # [OPTIMIZATION] Capture the frame AFTER audio finishes
                    # This ensures the camera sees what the user was describing/pointing at
                    target_frame = vision.read() 
                    
                    audio.speak("Thinking.")
                    
                    # C. Process Audio & Image
                    if success and target_frame is not None:
                        # 1. Transcribe Audio to Text
                        user_question = context_ai.transcribe_audio(WAVE_OUTPUT_FILENAME)
                        print(f"[User Asked] {user_question}")

                        if user_question and len(user_question) > 2:
                            # 2. If user spoke, answer their specific question
                            context_ai.answer_question(target_frame, user_question)
                        else:
                            # 3. If silence/noise, just describe the scene
                            print("[System] No clear question. Defaulting to description.")
                            context_ai.describe_scene(target_frame)
                    
                    elif target_frame is not None:
                        # Fallback if audio recording failed but we have a frame
                        context_ai.describe_scene(target_frame)
                    
                    # D. Reset & Cooldown
                    is_dark_state = False
                    darkness_start_time = 0
                    
                    # Delete the temp audio file to stay clean
                    if os.path.exists(WAVE_OUTPUT_FILENAME):
                        os.remove(WAVE_OUTPUT_FILENAME)

                    time.sleep(3.0) 
                    continue
            else:
                is_dark_state = False
                darkness_start_time = 0

            # ==================================================
            # [OPTIMIZATION] PAUSE/RESUME LOGIC
            # ==================================================
            # Pause danger detection if Gemini is thinking OR Audio is speaking
            if context_ai.is_busy or audio.speaking_lock:
                # Update display but skip heavy processing
                cv2.putText(inf_frame, "PAUSED: Processing...", (50, height - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("SixthSense Brain", inf_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                # Yield execution to let other threads run smoothly
                time.sleep(0.05) 
                continue 

            # ==================================================
            # 3. Danger Analysis (Standard Loop)
            # ==================================================
            
            is_danger, danger_name, closest_obj = danger_ai.analyze(inf_frame)
            
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
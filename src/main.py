import cv2
import time
import numpy as np
import pyaudio
import wave
import os
import re  # Essential for cleaning Gemini timestamps

from vision_stream import VisionStream
from danger_engine import DangerEngine
from audio_manager import AudioManager
from context_engine import ContextEngine
from navigation_engine import NavigationEngine 
from config import BRIGHTNESS_TRIGGER, ORS_API_KEY

# --- AUDIO RECORDING CONFIG ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "user_query.wav"

def record_audio_input():
    """
    Records audio until silence is detected OR a timeout is reached.
    """
    p = pyaudio.PyAudio()
    THRESHOLD = 600       
    SILENCE_LIMIT = 1.2   
    MAX_DURATION = 10.0   
    
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("[System] Listening... (Speak now)")
        frames = []
        start_time = time.time()
        last_sound_time = time.time()
        speech_started = False
        
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            current_time = time.time()
            total_duration = current_time - start_time
            
            if volume > THRESHOLD:
                last_sound_time = current_time
                if not speech_started: speech_started = True
            if speech_started and (current_time - last_sound_time > SILENCE_LIMIT): break
            if total_duration > MAX_DURATION: break
            if not speech_started and total_duration > 4.0: break

        stream.stop_stream()
        stream.close()
        p.terminate()
        
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
        print(f"[Audio Error] {e}")
        return False

def main():
    print("[Init] Starting Vision Stream...")
    vision = VisionStream().start()
    time.sleep(1.0) 

    print("[Init] Loading Engines...")
    danger_ai = DangerEngine()
    audio = AudioManager()
    audio.start()
    
    context_ai = ContextEngine(tts_callback=audio.speak)
    nav_engine = NavigationEngine(api_key=ORS_API_KEY)

    print("\n=== SIXTHSENSE ONLINE ===")
    audio.speak("System Online.")

    last_danger_time = 0
    DANGER_HOLD_DURATION = 1.0 
    darkness_start_time = 0
    is_dark_state = False
    TRIGGER_DURATION = 2.0 

    try:
        while True:
            frame = vision.read()
            if frame is None: continue

            inf_frame = cv2.resize(frame, (640, 640))
            height, width = inf_frame.shape[:2]

            # ==================================================
            # 1. CONTEXT TRIGGER (ROUTER)
            # ==================================================
            gray_avg = np.mean(cv2.cvtColor(inf_frame, cv2.COLOR_BGR2GRAY))
            
            if gray_avg < BRIGHTNESS_TRIGGER:
                if not is_dark_state:
                    darkness_start_time = time.time()
                    is_dark_state = True
                
                elapsed = time.time() - darkness_start_time
                
                if elapsed > TRIGGER_DURATION:
                    # --- TRIGGER ACTIVATED ---
                    audio.silence()
                    audio.speak("Ready.")
                    
                    while True:
                        temp = vision.read()
                        if temp is None: continue
                        if np.mean(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)) > BRIGHTNESS_TRIGGER:
                            break
                        time.sleep(0.05)
                    
                    audio.speak("Listening.")
                    success = record_audio_input()
                    target_frame = vision.read() 
                    audio.speak("Thinking.")
                    
                    if success and target_frame is not None:
                        user_q = context_ai.transcribe_audio(WAVE_OUTPUT_FILENAME)
                        print(f"[User Asked Raw] {user_q}")

                        if user_q:
                            # --- AGGRESSIVE TEXT CLEANING ---
                            # 1. Remove [Brackets]
                            clean_q = re.sub(r'\[.*?\]', '', user_q)
                            # 2. Remove Timestamps (00:00 or 00:00:00)
                            clean_q = re.sub(r'\b\d{2}:\d{2}\b', '', clean_q)
                            clean_q = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '', clean_q)
                            # 3. Remove "0000" artifacts
                            clean_q = re.sub(r'\b0+\b', '', clean_q)
                            # 4. Standard clean
                            clean_q = re.sub(r'[^\w\s]', '', clean_q).strip().lower()
                            # 5. Fix spaces
                            clean_q = re.sub(r'\s+', ' ', clean_q)

                            print(f"[Cleaned Command] {clean_q}")

                            if len(clean_q) > 2:
                                # --- ROUTER LOGIC ---
                                if "take me to" in clean_q or "navigate to" in clean_q:
                                    dest = clean_q.replace("take me to", "").replace("navigate to", "").strip()
                                    
                                    if len(dest) > 2:
                                        audio.speak(f"Calculating route to {dest}")
                                        # Use "Current Location" string, Map Engine handles the coordinates
                                        msg = nav_engine.calculate_route("Current Location", dest)
                                        audio.speak(msg)
                                        
                                        # Speak first instruction immediately
                                        time.sleep(3.0)
                                        first_step = nav_engine.get_next_instruction()
                                        if first_step: audio.speak(first_step)
                                    else:
                                        audio.speak("Destination not understood.")
                                else:
                                    context_ai.answer_question(target_frame, user_q)
                            else:
                                context_ai.describe_scene(target_frame)
                        else:
                            context_ai.describe_scene(target_frame)
                    elif target_frame is not None:
                        context_ai.describe_scene(target_frame)
                    
                    is_dark_state = False
                    darkness_start_time = 0
                    if os.path.exists(WAVE_OUTPUT_FILENAME):
                        os.remove(WAVE_OUTPUT_FILENAME)
                    time.sleep(2.0) 
                    continue
            else:
                is_dark_state = False
                darkness_start_time = 0

            # ==================================================
            # 2. PAUSE LOGIC
            # ==================================================
            if context_ai.is_busy:
                cv2.putText(inf_frame, "PAUSED: AI Thinking...", (50, height - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("SixthSense Brain", inf_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                time.sleep(0.05) 
                continue 

            # ==================================================
            # 3. MULTI-LAYER SAFETY & GUIDANCE
            # ==================================================
            
            # LAYER A: YOLO (Critical)
            is_danger, danger_name, closest_obj = danger_ai.analyze(inf_frame)
            current_time = time.time()
            if is_danger: last_danger_time = current_time
            in_danger_mode = (current_time - last_danger_time) < DANGER_HOLD_DURATION

            if in_danger_mode:
                pan = 0
                coverage = 0
                if closest_obj:
                    coverage = closest_obj['area'] / (width * height)
                    pan = (closest_obj['center_x'] - (width/2)) / (width/2)
                
                label = danger_name if is_danger else "DANGER"
                
                if coverage > 0.35:
                    audio.set_danger_critical(pan)
                    cv2.putText(inf_frame, f"CRITICAL: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                elif coverage > 0.15:
                    audio.set_danger_approaching(pan, label)
                    cv2.putText(inf_frame, f"WARNING: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                elif coverage > 0.05:
                    audio.set_danger_far(pan)
                    cv2.putText(inf_frame, f"DETECTED: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    audio.silence()

            # LAYER B: NAVIGATION (Only if Safe)
            elif nav_engine.is_navigating:
                nav_msg = nav_engine.get_next_instruction()
                if nav_msg:
                    audio.speak(nav_msg)
                else:
                    # Visual Path Guidance
                    deviation = nav_engine.get_path_deviation(inf_frame)
                    if deviation == 'left':
                        audio.set_danger_far(0.8) 
                        cv2.putText(inf_frame, ">> VEER RIGHT >>", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    elif deviation == 'right':
                        audio.set_danger_far(-0.8)
                        cv2.putText(inf_frame, "<< VEER LEFT <<", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        audio.silence()
            
            # LAYER C: PROXIMITY (Fallback)
            elif closest_obj:
                pan = (closest_obj['center_x'] - (width/2)) / (width/2)
                coverage = closest_obj['area'] / (width * height)
                x1,y1,x2,y2 = closest_obj['box']
                cv2.rectangle(inf_frame, (x1,y1), (x2,y2), (0,255,0), 2)

                if coverage > 0.35:
                    audio.announce_proximity(closest_obj['label'], pan)
                    cv2.putText(inf_frame, f"CLOSE: {closest_obj['label']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                else:
                    audio.silence()
            else:
                audio.silence()

            cv2.imshow("SixthSense Brain", inf_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        vision.stop()
        audio.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
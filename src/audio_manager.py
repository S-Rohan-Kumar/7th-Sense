import sounddevice as sd
import numpy as np
import threading
import pyttsx3
import time
import subprocess

class AudioManager:
    def __init__(self):
        # --- CONFIGURATION ---
        self.volume = 0.5
        
        # --- ROBUST DEVICE & SAMPLERATE DETECTION ---
        # We query the OS for the 'default' output device (e.g., your Bluetooth buds)
        # and strictly use its preferred Sample Rate.
        try:
            device_info = sd.query_devices(kind='output')
            self.sample_rate = int(device_info['default_samplerate'])
            print(f"[Audio] System Default Device: '{device_info['name']}' at {self.sample_rate}Hz")
        except Exception as e:
            print(f"[Audio Warning] Could not query device: {e}")
            # 48000 is safer for modern Bluetooth than 44100
            self.sample_rate = 48000 
        
        # --- STATE VARIABLES ---
        self.target_freq = 440.0
        self.current_pan = 0.0 
        self.mode = "silence" # silence, beep, siren
        
        # Beep Logic
        self.beep_interval = 0.0
        self.last_beep_time = 0.0
        self.beep_duration = 0.1
        self.is_beeping = False
        self.phase = 0

        # TTS & Haptics
        self.speaking_lock = False
        self.last_tts_time = 0
        self.last_haptic_time = 0
        self.last_spoken_obj = "" 
        
        # Init Stream
        # We do NOT pass a specific device ID. We let the OS route to the default.
        # We ONLY enforce the correct sample rate.
        self.stream = sd.OutputStream(
            channels=2, 
            blocksize=512, 
            samplerate=self.sample_rate, 
            callback=self.audio_callback
        )

    def start(self):
        self.stream.start()
        return self

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def _trigger_haptic(self, intensity="light"):
        now = time.time()
        if now - self.last_haptic_time < 0.2: return
        duration = "50" if intensity == "light" else "200"
        try:
            subprocess.Popen(
                ["adb", "shell", "cmd", "vibrator", "vibrate", duration],
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except: pass
        self.last_haptic_time = now

    def audio_callback(self, outdata, frames, time_info, status):
        if status: print(f"[Audio Status] {status}")
        current_time = time.time()
        
        # Calculate Stereo Pan
        norm_pan = (self.current_pan + 1) / 2
        norm_pan = max(0.0, min(1.0, norm_pan))

        # --- SIREN GENERATOR (Critical Danger) ---
        if self.mode == "siren":
            t = (np.arange(frames) + self.phase) / self.sample_rate
            # Sweep 800Hz - 1200Hz
            freq_sweep = 800 + 400 * np.abs(np.sin(2 * np.pi * 2 * t)) 
            tone = self.volume * 0.8 * (2 * (t * freq_sweep - np.floor(0.5 + t * freq_sweep)))
            
            # Apply Pan
            outdata[:, 0] = tone * (1.0 - norm_pan) # Left
            outdata[:, 1] = tone * norm_pan         # Right
            
            self.phase += frames
            return

        # --- BEEP GENERATOR (Warning Levels) ---
        if self.beep_interval > 0 and self.mode == "beep":
            if current_time - self.last_beep_time >= self.beep_interval:
                self.last_beep_time = current_time
                self.is_beeping = True
                
            if self.is_beeping and (current_time - self.last_beep_time > self.beep_duration):
                self.is_beeping = False

            if self.is_beeping:
                t = (np.arange(frames) + self.phase) / self.sample_rate
                t = t.reshape(-1, 1)
                tone = self.volume * np.sin(2 * np.pi * self.target_freq * t)
                
                # Apply Pan
                outdata[:, 0] = (tone * (1.0 - norm_pan)).flatten()
                outdata[:, 1] = (tone * norm_pan).flatten()
                self.phase += frames
            else:
                outdata.fill(0)
                self.phase = 0
        else:
            outdata.fill(0)
            self.phase = 0

    # --- DANGER INTERFACE (LEVELS) ---

    def set_danger_far(self, pan):
        """Level 1: Far (~1m). Warning Beeps."""
        self.mode = "beep"
        self.current_pan = pan
        self.target_freq = 660  # High-ish pitch
        self.beep_interval = 0.5 # Medium speed
        self.beep_duration = 0.1

    def set_danger_approaching(self, pan, obj_name):
        """Level 2: Mid (0.5-0.7m). Fast Beeps + Voice."""
        self.mode = "beep"
        self.current_pan = pan
        self.target_freq = 880  # Higher pitch
        self.beep_interval = 0.2 # Fast speed
        self.beep_duration = 0.1
        self._trigger_haptic("light")
        
        # Voice Warning Overlay
        current_time = time.time()
        if current_time - self.last_tts_time > 4.0:
            self.speak(f"Warning {obj_name} approaching")
            self.last_tts_time = current_time

    def set_danger_critical(self, pan):
        """Level 3: Close (<0.5m). Siren + Heavy Haptic."""
        self.mode = "siren"
        self.current_pan = pan 
        self._trigger_haptic("heavy")

    # --- SAFE OBJECT INTERFACE ---

    def announce_proximity(self, obj_name, pan):
        """Safe Object: Voice Only."""
        self.silence() 
        
        direction = "in front"
        if pan < -0.3: direction = "on your left"
        elif pan > 0.3: direction = "on your right"

        current_time = time.time()
        is_new_object = (obj_name != self.last_spoken_obj)
        is_time_up = (current_time - self.last_tts_time > 5.0)

        if is_new_object or is_time_up:
            text = f"{obj_name} {direction}"
            print(f"[Audio] Speaking: {text}")
            self.speak(text)
            self.last_tts_time = current_time
            self.last_spoken_obj = obj_name

    def silence(self):
        self.mode = "silence"
        self.is_beeping = False
        self.beep_interval = 0.0

    def speak(self, text):
        if self.speaking_lock: return
        
        # [OPTIMIZATION] Set lock IMMEDIATELY before thread spawns
        self.speaking_lock = True
        
        def _run():
            try:
                eng = pyttsx3.init()
                # 150 is a good comfortable speed
                eng.setProperty('rate', 150) 
                eng.say(text)
                eng.runAndWait()
            except: pass
            finally:
                # Release lock only when audio is actually finished
                self.speaking_lock = False
                
        threading.Thread(target=_run).start()
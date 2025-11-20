import sounddevice as sd
import numpy as np
import threading
import pyttsx3
import time
import subprocess

class AudioManager:
    def __init__(self):
        # --- CONFIGURATION ---
        self.sample_rate = 44100
        self.volume = 0.5
        
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
        if status: print(status)
        current_time = time.time()
        
        # --- SIREN GENERATOR (Danger) ---
        if self.mode == "siren":
            t = (np.arange(frames) + self.phase) / self.sample_rate
            freq_sweep = 800 + 400 * np.abs(np.sin(2 * np.pi * 2 * t)) 
            tone = self.volume * 0.8 * (2 * (t * freq_sweep - np.floor(0.5 + t * freq_sweep)))
            outdata[:, 0] = tone
            outdata[:, 1] = tone
            self.phase += frames
            return

        # --- BEEP GENERATOR ---
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
                norm_pan = (self.current_pan + 1) / 2
                outdata[:, 0] = (tone * (1.0 - norm_pan)).flatten()
                outdata[:, 1] = (tone * norm_pan).flatten()
                self.phase += frames
            else:
                outdata.fill(0)
                self.phase = 0
        else:
            outdata.fill(0)
            self.phase = 0

    # --- INTERFACE ---

    def set_hazard_mode(self):
        """DANGER (General): Siren + Strong Haptic"""
        self.mode = "siren"
        self._trigger_haptic("heavy")

    def set_hazard_approaching(self, obj_name):
        """DANGER (0.5-0.7m): Siren + Strong Haptic + VOICE"""
        self.mode = "siren"
        self._trigger_haptic("heavy")
        
        # Voice Warning Overlay (with cooldown)
        current_time = time.time()
        if current_time - self.last_tts_time > 5.0:
            self.speak(f"Warning {obj_name} approaching")
            self.last_tts_time = current_time

    def announce_proximity(self, obj_name, pan):
        """SAFE OBJECTS: Voice Only (No Beep)"""
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
        def _run():
            self.speaking_lock = True
            try:
                eng = pyttsx3.init()
                eng.say(text)
                eng.runAndWait()
            except: pass
            self.speaking_lock = False
        threading.Thread(target=_run).start()   
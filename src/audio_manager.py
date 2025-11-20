import sounddevice as sd
import numpy as np
import threading
import pyttsx3
import time

class AudioManager:
    def __init__(self):
        # --- CONFIGURATION ---
        self.sample_rate = 44100
        self.volume = 0.5
        
        # --- STATE VARIABLES ---
        self.target_freq = 440.0
        self.current_pan = 0.0 # -1.0 (Left) to 1.0 (Right)
        
        # Beep Logic (Matches hand.py)
        self.beep_interval = 0.0  # 0.0 = Silence/Off
        self.last_beep_time = 0.0
        self.beep_duration = 0.1  # 100ms beep length
        self.is_beeping = False
        self.phase = 0

        # TTS Engine
        self.speaking_lock = False
        
        # Initialize Stream (Paused by default until start is called)
        self.stream = sd.OutputStream(
            channels=2, 
            blocksize=512, 
            samplerate=self.sample_rate, 
            callback=self.audio_callback
        )

    def start(self):
        """Starts the audio stream."""
        self.stream.start()
        return self

    def stop(self):
        """Stops the audio stream."""
        self.stream.stop()
        self.stream.close()

    def audio_callback(self, outdata, frames, time_info, status):
        """Real-time audio generation callback (runs on separate thread)."""
        if status:
            print(status)

        current_time = time.time()

        # 1. Check logic to START a beep
        if self.beep_interval > 0:
            time_since_last = current_time - self.last_beep_time
            if time_since_last >= self.beep_interval:
                self.last_beep_time = current_time
                self.is_beeping = True

        # 2. Check logic to STOP a beep
        if self.is_beeping:
            if current_time - self.last_beep_time > self.beep_duration:
                self.is_beeping = False

        # 3. Generate Audio
        if self.is_beeping and self.beep_interval > 0:
            # Generate Sine Wave
            t = (np.arange(frames) + self.phase) / self.sample_rate
            t = t.reshape(-1, 1)
            
            tone = self.volume * np.sin(2 * np.pi * self.target_freq * t)
            
            # Stereo Panning logic: Map -1.0/1.0 (Main) to 0.0/1.0 (Synth Logic)
            # Input Pan: -1 (Left), 0 (Center), 1 (Right)
            # Synth Pan: 0 (Left), 0.5 (Center), 1 (Right)
            norm_pan = (self.current_pan + 1) / 2
            norm_pan = max(0.0, min(1.0, norm_pan))

            left = tone * (1.0 - norm_pan)
            right = tone * norm_pan
            
            outdata[:, 0] = left.flatten()
            outdata[:, 1] = right.flatten()
            
            self.phase += frames
        else:
            # Silence
            outdata.fill(0)
            self.phase = 0

    def update_sonar(self, pan, interval, freq=880):
        """
        Updates the continuous audio state.
        pan: -1.0 (Left) to 1.0 (Right)
        interval: Time in seconds between beeps (Lower = Faster)
        freq: Tone pitch in Hz
        """
        self.current_pan = pan
        self.beep_interval = interval
        self.target_freq = freq

    def play_danger(self):
        """Sets audio to 'Danger Mode' (High pitch, very fast/continuous)."""
        self.current_pan = 0.0 # Center
        self.target_freq = 1200 # High pitch
        self.beep_interval = 0.05 # Extremely fast (almost continuous)

    def silence(self):
        """Turns off the beep."""
        self.beep_interval = 0.0
        self.is_beeping = False

    def speak(self, text):
        """Threaded TTS (unchanged)."""
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
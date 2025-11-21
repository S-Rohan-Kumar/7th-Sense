from google import genai
from google.genai.errors import APIError
from google.genai.types import Part
import cv2
import threading
import os
# Assuming config.py is in the same directory
from config import GEMINI_API_KEY 

class ContextEngine:
    
    # Using 2.0 Flash for best speed/latency balance
    MODEL_NAME = 'gemini-2.5-flash' 

    def __init__(self, tts_callback):
        self.api_key = GEMINI_API_KEY
        self.client = None 
        self.is_busy = False
        self.tts = tts_callback # Function to call when text is ready
        self._setup_gemini()

    def _setup_gemini(self):
        """Initializes the Gemini Client."""
        try:
            # Client is created and API key is passed explicitly
            self.client = genai.Client(api_key=self.api_key)
            print(f"[System] Gemini client initialized with {self.MODEL_NAME}")
        except Exception as e:
            print(f"[System] Gemini Initialization Error: {e}")
            self.client = None 

    def _gemini_worker(self, frame, prompt):
        """
        Worker function for all threaded Gemini calls (Vision QA).
        Uses in-memory byte encoding to bypass file path I/O latency.
        """
        if not self.client:
            self.tts("I am not connected to the AI service.")
            return

        self.is_busy = True
        
        try:
            # 1. Encode the frame (NumPy array) directly to JPEG bytes in memory
            # (Optimization: No disk I/O)
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                raise ValueError("Could not encode frame to JPEG bytes.")

            image_bytes = buffer.tobytes()
            
            # 2. Create the image part directly from the in-memory bytes
            image_part = Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
            
            # 3. Call the model via the client (Streaming for faster feedback)
            response_stream = self.client.models.generate_content_stream(
                model=self.MODEL_NAME, 
                contents=[prompt, image_part]
            )
            
            # 4. Process streaming response
            text = ""
            for chunk in response_stream:
                text += chunk.text
            
            text = text.strip()
            print(f"[Gemini] {text}")
            self.tts(text) # Speak the full result
            
        except APIError as e:
            print(f"[Gemini API Error] {e}")
            self.tts("Service error.")
        except Exception as e:
            print(f"[Gemini General Error] {e}")
            self.tts("I encountered an issue processing the image.")
        finally:
            self.is_busy = False

    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Synchronous call to transcribe audio using Gemini.
        """
        if not self.client or not os.path.exists(audio_file_path):
            return ""

        print("[System] Transcribing audio...")
        try:
            # 1. Read the audio file into memory
            with open(audio_file_path, 'rb') as f:
                audio_bytes = f.read()

            # 2. Create the Part object directly from bytes
            audio_part = Part.from_bytes(data=audio_bytes, mime_type='audio/wav')
            
            # 3. Call the model
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=[audio_part, "Transcribe the audio in its entirety."]
            )

            return response.text.strip()
            
        except Exception as e:
            print(f"[STT Error] {e}")
            return ""

    def describe_scene(self, frame):
        """Threaded call for immediate scene description."""
        if self.is_busy or not self.client: return

        prompt = "I am blind. In one very short sentence, tell me what is directly in front of me and if it is safe."
        # Daemon thread ensures main program doesn't hang waiting for this
        threading.Thread(target=self._gemini_worker, args=(frame, prompt), daemon=True).start()

    def answer_question(self, frame, question: str):
        """
        Threaded call to answer a specific user question.
        """
        if self.is_busy: 
            self.tts("I am currently busy, please wait.")
            return
            
        if not self.client:
            self.tts("I am not connected to the language service.")
            return

        # [OPTIMIZATION] Shortened system instruction for faster generation
        prompt = f"Answer concisely. Question: \"{question}\""
        
        print(f"[System] Triggering QA: {question}")
        threading.Thread(target=self._gemini_worker, args=(frame, prompt), daemon=True).start()
import google.generativeai as genai
import cv2
import threading
import time
# FIXED: Removed '.' before config
from config import GEMINI_API_KEY

class ContextEngine:
    def __init__(self, tts_callback):
        self.api_key = GEMINI_API_KEY
        self.model = None
        self.is_busy = False
        self.tts = tts_callback # Function to call when text is ready
        self._setup_gemini()

    def _setup_gemini(self):
        try:
            genai.configure(api_key=self.api_key)
            # Try robust model selection
            for m in ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro-vision']:
                try:
                    self.model = genai.GenerativeModel(m)
                    print(f"[System] Gemini connected via {m}")
                    break
                except: continue
        except Exception as e:
            print(f"[System] Gemini Error: {e}")

    def describe_scene(self, frame):
        """Threaded call to Gemini to prevent UI freeze."""
        if self.is_busy or not self.model: return

        def _worker():
            self.is_busy = True
            try:
                # Temp file for upload (safer than memory buffer for some API versions)
                cv2.imwrite("temp_ctx.jpg", frame)
                
                # The Prompt
                prompt = "I am blind. In one very short sentence, tell me what is directly in front of me and if it is safe."
                
                img_file = genai.upload_file(path="temp_ctx.jpg", display_name="ContextInput")
                response = self.model.generate_content([prompt, img_file])
                
                text = response.text.strip()
                print(f"[Gemini] {text}")
                self.tts(text) # Speak result
                
            except Exception as e:
                print(f"[Gemini Error] {e}")
                self.tts("I couldn't see that.")
            finally:
                self.is_busy = False

        threading.Thread(target=_worker).start()
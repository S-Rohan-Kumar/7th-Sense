import google.generativeai as genai
import cv2
import threading
import time
import os
from config import GEMINI_API_KEY

class ContextEngine:
    def __init__(self, tts_callback):
        self.api_key = GEMINI_API_KEY
        self.model = None
        self.is_busy = False
        self.tts = tts_callback 
        self._setup_gemini()

    def _setup_gemini(self):
        """
        Robust connection logic: 
        1. Configure API
        2. LIST available models
        3. Select the best vision-capable model
        """
        try:
            genai.configure(api_key=self.api_key)
            
            print("[Gemini] Scanning for available models...")
            found_model = None
            
            # 1. Get all models the user has access to
            available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
            except Exception as list_err:
                print(f"[Gemini Warning] Could not list models: {list_err}")
                # Fallback list if listing fails
                available_models = ['models/gemini-2.5-flash', 'models/gemini-pro-vision']

            # 2. Priority List (Best to Worst)
            # We look for these strings inside the available model names
            priority_keywords = [
                'gemini-2.5-flash',
                'gemini-2.5-pro',
                'gemini-pro-vision',
                'gemini-2.0-pro-vision'
            ]

            for keyword in priority_keywords:
                for m_name in available_models:
                    if keyword in m_name:
                        found_model = m_name
                        break
                if found_model: break

            # 3. Connect
            if found_model:
                self.model = genai.GenerativeModel(found_model)
                print(f"[System] ✅ Gemini connected via: {found_model}")
            else:
                print("[System] ❌ No Vision model found. AI features disabled.")
                print(f"Available models were: {available_models}")

        except Exception as e:
            print(f"[System] Gemini Critical Error: {e}")

    def describe_scene(self, frame):
        """
        Captures the frame and asks Gemini.
        """
        if self.is_busy or not self.model: return

        def _worker():
            self.is_busy = True
            filename = f"temp_ctx_{int(time.time())}.jpg"
            try:
                # 1. Save Frame temporarily
                cv2.imwrite(filename, frame)
                
                # 2. The Prompt
                prompt = "The object in front of me is what? Answer in one short sentence."
                
                # 3. Generate (Using the simpler Pillow interface if file upload fails)
                # We try the file API first as it is standard for 1.5
                try:
                    print("[Gemini] Uploading image...")
                    img_file = genai.upload_file(path=filename, display_name="ContextInput")
                    
                    print("[Gemini] Thinking...")
                    response = self.model.generate_content([prompt, img_file])
                    
                    # Cleanup cloud file (good practice)
                    # genai.delete_file(img_file.name)
                    
                except Exception as upload_err:
                    print(f"[Gemini Upload Error] {upload_err}. Trying legacy byte stream...")
                    # Fallback: Send bytes directly (Works better for some older keys)
                    import PIL.Image
                    img = PIL.Image.open(filename)
                    response = self.model.generate_content([prompt, img])

                # 4. Output
                text = response.text.strip()
                print(f"[Gemini] Answer: {text}")
                self.tts(text) 
                
            except Exception as e:
                print(f"[Gemini Error] {e}")
                self.tts("I could not identify that.")
            finally:
                # Cleanup local file
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except: pass
                self.is_busy = False

        threading.Thread(target=_worker).start()
# 7th Sense: AI-Powered Assistive Vision & Navigation

7th Sense is an advanced real-time assistive technology designed to work as a digital guide for visually impaired individuals. It combines **Computer Vision (YOLO)**, **Generative AI (Google Gemini)**, **Spatial Audio**, and **Voice Navigation** to provide intelligent environmental awareness and obstacle detection.

---

## 🚀 Key Features

### 👁️ Intelligent Perception

* **Real-Time Object Detection:** Powered by YOLOv8 to detect objects such as people, cars, trucks, and more.
* **Generative Scene Description:** Google Gemini 2.5 Flash describes complex scenes and answers context-aware questions.
* **Smart Danger Analysis:** Computes danger scores using:

  * Object proximity (bounding box size)
  * Class priority (vehicles > bicycles > objects)
  * Position relative to user's center of view

---

### 🎧 Spatial & Haptic Feedback

* **3D Spatial Audio:** Stereo panning indicates object direction.
* **Dynamic Warning Levels:**

  * **Level 1 (Far):** Slow beeps
  * **Level 2 (Approaching):** Fast beeps + Voice alert + Light haptic
  * **Level 3 (Critical):** Siren + Strong haptic vibration
* **Offline TTS:** PyTTSx3 for low-latency voice feedback.

---

### 🗺️ Voice-Activated Navigation

* **Step-by-Step Guidance:** Generated using OpenRouteService.
* Converts GPS distances into **human-readable steps**.
* **Path Correction:** Detects left/right deviation with audio cues.
* **Natural Language Commands:** e.g., “Take me to the library”.

---

### 👆 Intuitive Control (Darkness Trigger)

* Cover the camera lens for **2 seconds** to activate voice-listening mode.
* No physical buttons required.

---

## 🛠️ System Architecture

The system uses a multi-threaded pipeline for ultra-low latency.

| Module                   | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| **main.py**              | System orchestrator, darkness trigger, audio handling |
| **vision_stream.py**     | Camera feed in a daemon thread                        |
| **danger_engine.py**     | YOLOv8 inference and risk scoring                     |
| **context_engine.py**    | Handles Google Gemini API for VQA                     |
| **audio_manager.py**     | Tone generation, TTS, spatial audio                   |
| **navigation_engine.py** | Geocoding, routing, turn-by-turn navigation           |

---

## 📦 Installation

### **Prerequisites**

* Python 3.9+
* Webcam
* Stable internet connection (for Gemini & Navigation APIs)

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/7th-sense.git
cd 7th-sense
```

### **2. Install Dependencies**

```bash
pip install opencv-python google-genai ultralytics pyaudio pyttsx3 openrouteservice numpy sounddevice
```

> Ensure PyTorch with CUDA is installed if using an NVIDIA GPU.

### **3. Configuration**

Create a `config.py` file:

```python
# config.py

# Keys
GEMINI_API_KEY = "your_google_gemini_key"
ORS_API_KEY = "your_openrouteservice_key"

# Hardware
CAMERA_SOURCE = 0   # webcam
USE_GPU = True

# Thresholds
CONFIDENCE_THRESHOLD = 0.5
BRIGHTNESS_TRIGGER = 30
DANGER_CLASSES = [0, 1, 2, 3, 5, 7]

# Navigation (Demo)
DEMO_ORIGIN_COORDS = [77.5946, 12.9716]
```

---

## 🎮 Usage

Run the system:

```bash
python main.py
```

### **Interaction Flow**

* **Startup:** System says *“System Online”*
* **Passive Mode:** Walk normally; system alerts when obstacles appear.
* **Active Voice Mode:**

  1. Cover camera for 2 seconds
  2. System says *“Listening”*
  3. Speak a command:

     * “What is in front of me?”
     * “Navigate to the nearest coffee shop”
     * “Read this text”

---

## 🧩 Tech Stack

| Component  | Technology              | Purpose                      |
| ---------- | ----------------------- | ---------------------------- |
| Core Logic | Python 3                | Orchestration                |
| Vision AI  | YOLOv8                  | Object detection             |
| LLM/VQA    | Google Gemini 2.5 Flash | Scene understanding          |
| Audio      | PyAudio, NumPy          | Synthetic tone generation    |
| TTS        | PyTTSx3                 | Offline voice output         |
| Navigation | OpenRouteService        | Routing engine               |
| Haptics    | ADB Shell               | Vibration control on Android |

---

## 📌 Future Improvements

* Add SLAM-based indoor navigation
* Integrate lightweight on-device LLM model
* Add depth estimation using MiDaS or ZED camera

---

## 🤝 Contributions

Feel free to submit PRs or raise issues.

---

## 📜 License

MIT License — Open for personal and research use.

---

**7th Sense – Giving Vision Beyond Sight.**

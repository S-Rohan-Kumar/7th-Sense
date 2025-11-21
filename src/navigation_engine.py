import openrouteservice
from openrouteservice.directions import directions
import time
import cv2
import numpy as np
from config import ORS_API_KEY, DEMO_ORIGIN_COORDS

class NavigationEngine:
    def __init__(self, api_key=None):
        key_to_use = api_key if api_key else ORS_API_KEY
        
        self.client = None
        if key_to_use and len(key_to_use) > 10:
            try:
                self.client = openrouteservice.Client(key=key_to_use)
                print("[Nav] OpenRouteService Client Loaded.")
            except:
                print("[Nav] ORS Key invalid. Using Mock Mode.")
        
        self.steps = []
        self.current_step_index = 0
        self.is_navigating = False
        self.last_update_time = 0
        self.step_duration = 6.0  
        
        # [CHANGED] Updated Mock Route to use Steps instead of Meters
        self.mock_route = [
            "Route started. Head north.",
            "In 15 steps, turn left.",
            "Continue straight for 25 steps.",
            "In 10 steps, turn right at the corridor.",
            "You have arrived."
        ]

    def calculate_route(self, start_text, end_text):
        """
        Fetches REAL directions and converts distances to STEPS.
        """
        print(f"[Nav] Calculating: {start_text} -> {end_text}")
        self.steps = []
        
        if self.client:
            try:
                # 1. Geocode destination
                geocode = self.client.pelias_search(text=end_text, focus_point=DEMO_ORIGIN_COORDS)
                
                if not geocode['features']:
                    return f"Could not find location: {end_text}"
                
                dest_coords = geocode['features'][0]['geometry']['coordinates']
                
                # 2. Get Walking Directions
                route = directions(
                    self.client, 
                    coordinates=[DEMO_ORIGIN_COORDS, dest_coords],
                    profile='foot-walking', 
                    format='geojson'
                )
                
                # 3. Parse & Convert to Steps
                segments = route['features'][0]['properties']['segments']
                self.steps.append(f"Route calculated. Walking to {end_text}.")
                
                for segment in segments:
                    for step in segment['steps']:
                        instr = step['instruction']
                        dist_meters = int(step['distance'])
                        
                        if dist_meters > 0:
                            # [LOGIC] 1 Step approx 0.75 meters
                            steps_count = int(dist_meters / 0.75)
                            self.steps.append(f"In {steps_count} steps, {instr}")
                        else:
                            self.steps.append(instr)
                            
                self.steps.append("You have arrived.")

            except Exception as e:
                print(f"[Nav] API Error: {e}")
        
        # Fallback
        if not self.steps:
            print("[Nav] Using Mock Route.")
            self.steps = self.mock_route

        self.current_step_index = 0
        self.is_navigating = True
        self.last_update_time = time.time()
        return f"Navigating to {end_text}."

    def get_next_instruction(self):
        """Returns next instruction based on timer."""
        if not self.is_navigating: return None

        if time.time() - self.last_update_time > self.step_duration:
            if self.current_step_index < len(self.steps):
                instruction = self.steps[self.current_step_index]
                self.current_step_index += 1
                self.last_update_time = time.time()
                return instruction
            else:
                self.is_navigating = False
                return "Navigation ended."
        return None

    def get_path_deviation(self, frame):
        """Visual Path Logic (Vanishing Point)"""
        if not self.is_navigating: return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            if lines is None: return None
            left_slopes = []
            right_slopes = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                if 0.4 < slope < 2.0: right_slopes.append(slope)
                elif -2.0 < slope < -0.4: left_slopes.append(slope)
            if len(left_slopes) > len(right_slopes) + 2: return "right" 
            if len(right_slopes) > len(left_slopes) + 2: return "left"  
        except: pass
        return None
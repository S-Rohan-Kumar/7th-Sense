import openrouteservice
from openrouteservice.directions import directions
import time
import cv2
import numpy as np
# [NEW] Import config
from config import ORS_API_KEY, DEMO_ORIGIN_COORDS

class NavigationEngine:
    def __init__(self, api_key=None):
        # Use key from config if not passed
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
        self.step_duration = 6.0  # Advance 1 step every 6s for demo
        
        # Mock Route (Fallback)
        self.mock_route = [
            "Route started. Head north.",
            "In 10 meters, turn left.",
            "Continue straight for 20 meters.",
            "Turn right at the corridor.",
            "You have arrived."
        ]

    def calculate_route(self, start_text, end_text):
        """
        Fetches REAL directions using OpenStreetMap data.
        Note: ORS prefers coordinates, so for a hackathon demo, 
        we will Geocode the destination text first.
        """
        print(f"[Nav] Calculating: {start_text} -> {end_text}")
        self.steps = []
        
        if self.client:
            try:
                # 1. Geocode the destination (Convert 'Library' to Lat/Lon)
                # We assume start is fixed (DEMO_ORIGIN_COORDS) to save time/complexity
                geocode = self.client.pelias_search(text=end_text, focus_point=DEMO_ORIGIN_COORDS)
                
                if not geocode['features']:
                    return f"Could not find location: {end_text}"
                
                # Get coords of first result
                dest_coords = geocode['features'][0]['geometry']['coordinates']
                
                # 2. Get Walking Directions
                # ORS format is [[Lon, Lat], [Lon, Lat]]
                route = directions(
                    self.client, 
                    coordinates=[DEMO_ORIGIN_COORDS, dest_coords],
                    profile='foot-walking', 
                    format='geojson'
                )
                
                # 3. Parse Steps
                # ORS returns segments -> steps -> instruction
                segments = route['features'][0]['properties']['segments']
                self.steps.append(f"Route calculated. Walking to {end_text}.")
                
                for segment in segments:
                    for step in segment['steps']:
                        instr = step['instruction']
                        dist = int(step['distance'])
                        if dist > 0:
                            self.steps.append(f"In {dist} meters, {instr}")
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
        """Visual Path Logic (Same as before)"""
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
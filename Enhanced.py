#!/usr/bin/env python3
"""
Score-Based Rally Detection for Continuous Badminton Video
Detects rallies by identifying when points are scored
"""

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from ultralytics import YOLO
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class ScoreBasedRallyDetector:
    def __init__(self):
        print("🚀 Loading models...")
        
        # YOLO for pose detection
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        # VLM for scene understanding
        self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="cuda:0"
        )
        self.vlm_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        
        print("✅ Models loaded!")
        
        self.rallies = []
        self.score_a = 0
        self.score_b = 0
    
    def detect_rally_end_indicators(self, frame, prev_frame):
        """
        Detect if a rally just ended by looking for:
        1. Sudden player movement changes (celebration/disappointment)
        2. Shuttlecock hitting ground
        3. Large motion changes
        """
        # Convert to grayscale for motion analysis
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Motion intensity
        motion_intensity = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
        
        return motion_intensity
    
    def detect_shuttlecock_landing(self, frames_window):
        """
        Analyze last few frames to detect shuttlecock landing
        Returns: (landed, position)
        """
        if len(frames_window) < 5:
            return False, None
        
        shuttlecock_positions = []
        
        for frame in frames_window:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
            
            # Add yellow detection
            mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
            mask = cv2.bitwise_or(mask, mask_yellow)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if 10 < cv2.contourArea(contour) < 500:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cy = int(M["m01"] / M["m00"])
                        shuttlecock_positions.append(cy)
                        break
        
        if len(shuttlecock_positions) >= 3:
            # Check if shuttlecock is moving downward and stops
            y_positions = shuttlecock_positions[-3:]
            
            # Downward movement
            if y_positions[-1] > y_positions[0]:
                # Check if it stopped moving (landed)
                if abs(y_positions[-1] - y_positions[-2]) < 5:
                    return True, y_positions[-1]
        
        return False, None
    
    def analyze_point_with_vlm(self, frame):
        """
        Use VLM to understand what happened at this point
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        prompt = """<image>
USER: This is a badminton match. Look at the players' body language and positions. 
Did someone just score a point? Who looks like they won - the player on the left or right?
What type of shot was played? Answer briefly.
ASSISTANT:"""
        
        inputs = self.vlm_processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = self.vlm_model.generate(**inputs, max_new_tokens=80, do_sample=False)
        
        response = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
        
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        # Parse response
        winner = None
        if "left" in response.lower():
            winner = "left"
        elif "right" in response.lower():
            winner = "right"
        
        # Extract shot type
        shot_types = ["smash", "clear", "drop", "net shot", "drive", "lob"]
        shot_type = "push"
        for shot in shot_types:
            if shot in response.lower():
                shot_type = shot.replace(" ", "_")
                break
        
        return winner, shot_type, response
    
    def process_video(self, video_path, player_a="Player A", player_b="Player B"):
        """
        Process video by detecting rally end points
        """
        print(f"\n{'='*80}")
        print(f"🎾 SCORE-BASED RALLY DETECTION")
        print(f"{'='*80}")
        print(f"Players: {player_a} vs {player_b}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {fps} FPS, {total_frames} frames, {total_frames/fps:.1f}s")
        
        frame_count = 0
        prev_frame = None
        frames_window = []
        motion_history = []
        
        rally_start_time = 0
        rally_start_frame = 0
        in_rally = True  # Assume video starts mid-rally
        
        # Thresholds
        motion_spike_threshold = 0.15  # Sudden motion increase
        landing_check_window = 10
        cooldown_frames = int(fps * 2)  # 2 second cooldown between rallies
        frames_since_last_rally = cooldown_frames
        
        print("\n🔍 Analyzing video for rally end points...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Build frame window
            frames_window.append(frame.copy())
            if len(frames_window) > landing_check_window:
                frames_window.pop(0)
            
            # Motion analysis
            if prev_frame is not None:
                motion = self.detect_rally_end_indicators(frame, prev_frame)
                motion_history.append(motion)
                
                if len(motion_history) > 30:
                    motion_history.pop(0)
                
                # Detect motion spikes (possible rally end)
                if (len(motion_history) >= 10 and 
                    motion > motion_spike_threshold and
                    motion > np.mean(motion_history[-10:]) * 2 and
                    frames_since_last_rally >= cooldown_frames):
                    
                    # Check for shuttlecock landing
                    landed, y_pos = self.detect_shuttlecock_landing(frames_window)
                    
                    if landed or motion > motion_spike_threshold * 1.5:
                        # Rally ended! Analyze with VLM
                        print(f"\n🎾 Possible rally end at {current_time:.1f}s (motion: {motion:.3f})")
                        
                        winner, shot_type, analysis = self.analyze_point_with_vlm(frame)
                        
                        print(f"   Analysis: {analysis[:100]}...")
                        
                        # Determine winner
                        if winner == "left":
                            winner_name = player_a
                            self.score_a += 1
                        elif winner == "right":
                            winner_name = player_b
                            self.score_b += 1
                        else:
                            # Default to alternating
                            if len(self.rallies) % 2 == 0:
                                winner_name = player_a
                                self.score_a += 1
                            else:
                                winner_name = player_b
                                self.score_b += 1
                        
                        rally = {
                            'start_time': self.format_time(rally_start_time),
                            'end_time': self.format_time(current_time),
                            'win_point_player': winner_name,
                            'win_reason': 'wins by landing',
                            'ball_types': shot_type,
                            'lose_reason': 'unable to return',
                            'roundscore_A': self.score_a,
                            'roundscore_B': self.score_b
                        }
                        
                        self.rallies.append(rally)
                        
                        print(f"✅ Rally {len(self.rallies)}: {rally['start_time']}-{rally['end_time']} | "
                              f"{winner_name} | {shot_type} | {self.score_a}-{self.score_b}")
                        
                        # Reset for next rally
                        rally_start_time = current_time
                        rally_start_frame = frame_count
                        frames_since_last_rally = 0
                        motion_history = []
            
            prev_frame = frame.copy()
            frame_count += 1
            frames_since_last_rally += 1
            
            # Progress
            if frame_count % 300 == 0:
                print(f"⏳ Processed {frame_count}/{total_frames} frames "
                      f"({frame_count/total_frames*100:.1f}%) | Rallies: {len(self.rallies)}")
        
        cap.release()
        
        # Save results
        df = pd.DataFrame(self.rallies)
        df.to_csv('badminton_score_based_results.csv', index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ COMPLETE!")
        print(f"📊 Total rallies: {len(self.rallies)}")
        print(f"🏆 Final Score: {self.score_a}-{self.score_b}")
        print(f"💾 Saved to: badminton_score_based_results.csv")
        print(f"{'='*80}\n")
        
        if len(self.rallies) > 0:
            print(df.to_string(index=False))
        
        return self.rallies
    
    def format_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    detector = ScoreBasedRallyDetector()
    
    detector.process_video(
        video_path="Sample.mp4",
        player_a="An Se Young",
        player_b="Ratchanok Intanon"
    )

#!/usr/bin/env python3
"""
Hybrid Badminton Rally Extractor
Uses YOLO for tracking + LLaVA for shot classification and reasoning
"""

import torch
import cv2
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from collections import deque

class HybridBadmintonExtractor:
    def __init__(self, vlm_model="llava-hf/llava-1.5-7b-hf", device="cuda:1"):
        """
        Hybrid approach: YOLO for detection + VLM for reasoning
        """
        self.device = device
        
        print("🚀 Loading YOLO detector...")
        self.yolo_pose = YOLO('yolov8n-pose.pt')
        
        print(f"🚀 Loading VLM on {device}...")
        self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model)
        
        print("✅ All models loaded!")
        
        self.rallies = []
        self.score_a = 0
        self.score_b = 0
        self.shuttlecock_buffer = deque(maxlen=30)
    
    def detect_shuttlecock_yolo(self, frame):
        """Fast shuttlecock detection with YOLO + color filtering"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White shuttlecock
        mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_detection = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    
                    score = circularity * area / 100
                    
                    if score > best_score:
                        best_score = score
                        best_detection = (cx, cy)
        
        if best_detection:
            self.shuttlecock_buffer.append(best_detection)
        
        return best_detection
    
    def detect_players_yolo(self, frame):
        """Detect players with YOLO pose"""
        results = self.yolo_pose(frame, verbose=False)
        players = []
        
        for result in results:
            if result.keypoints is not None:
                for kp in result.keypoints:
                    kp_data = kp.xy[0].cpu().numpy()
                    conf = kp.conf[0].cpu().numpy()
                    
                    valid = kp_data[conf > 0.5]
                    if len(valid) > 5:
                        x_coords = valid[:, 0]
                        y_coords = valid[:, 1]
                        
                        players.append({
                            'center_x': int(x_coords.mean()),
                            'center_y': int(y_coords.mean()),
                            'bbox': [
                                int(x_coords.min()), int(y_coords.min()),
                                int(x_coords.max()), int(y_coords.max())
                            ]
                        })
        
        players.sort(key=lambda p: p['center_x'])
        return players[:2]
    
    def analyze_rally_with_vlm(self, frame, shuttlecock_trajectory, players):
        """
        Use VLM only for high-level reasoning and shot classification
        Called only when rally ends
        """
        # Prepare frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Draw trajectory on frame for context
        vis_frame = frame_rgb.copy()
        if len(shuttlecock_trajectory) > 1:
            pts = np.array(shuttlecock_trajectory, dtype=np.int32)
            cv2.polylines(vis_frame, [pts], False, (0, 255, 0), 2)
        
        vis_image = Image.fromarray(vis_frame)
        
        prompt = f"""<image>
Analyze this badminton rally frame with the shuttlecock trajectory shown in green.

Answer these questions:
1. What type of shot was just played? (smash/clear/drop/net_shot/drive/lob/push/lift)
2. Who won the point? (left_player/right_player)
3. Why did they win? (powerful_shot/placement/opponent_error/net/out_of_bounds)

Respond in this exact format:
Shot: <shot_type>
Winner: <left_player or right_player>
Reason: <reason>
"""

        inputs = self.vlm_processor(text=prompt, images=vis_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vlm_model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        response = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse response
        shot_type = "push"
        winner = "left_player"
        reason = "opponent error"
        
        lines = response.lower().split('\n')
        for line in lines:
            if 'shot:' in line:
                for s in ['smash', 'clear', 'drop', 'net_shot', 'drive', 'lob', 'push', 'lift']:
                    if s in line:
                        shot_type = s
                        break
            if 'winner:' in line:
                winner = 'left_player' if 'left' in line else 'right_player'
            if 'reason:' in line:
                reason = line.split('reason:')[1].strip()
        
        return shot_type, winner, reason
    
    def process_video(self, video_path, player_a="Player A", player_b="Player B"):
        """
        Main processing: YOLO for tracking, VLM for analysis
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*80}")
        print(f"🎾 HYBRID BADMINTON EXTRACTION")
        print(f"{'='*80}")
        print(f"Players: {player_a} vs {player_b}")
        print(f"Video: {fps:.1f} FPS, {total_frames} frames")
        
        frame_count = 0
        rally_active = False
        rally_start_time = 0
        rally_trajectory = []
        frames_without_shuttle = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # Fast YOLO detection
            shuttlecock = self.detect_shuttlecock_yolo(frame)
            players = self.detect_players_yolo(frame)
            
            # Rally state machine
            if shuttlecock:
                frames_without_shuttle = 0
                
                if not rally_active:
                    rally_active = True
                    rally_start_time = current_time
                    rally_trajectory = []
                    print(f"\n🎾 Rally started at {self.format_time(current_time)}")
                
                rally_trajectory.append(shuttlecock)
            else:
                frames_without_shuttle += 1
            
            # Rally ended
            if rally_active and frames_without_shuttle > int(fps * 1.5):
                rally_end_time = current_time
                
                # Now use VLM for analysis
                print(f"🤖 Analyzing rally with VLM...")
                shot_type, winner, reason = self.analyze_rally_with_vlm(
                    frame, rally_trajectory, players
                )
                
                # Create rally record
                if winner == 'left_player':
                    winner_name = player_a
                    self.score_a += 1
                else:
                    winner_name = player_b
                    self.score_b += 1
                
                rally = {
                    'start_time': self.format_time(rally_start_time),
                    'end_time': self.format_time(rally_end_time),
                    'win_point_player': winner_name,
                    'win_reason': reason,
                    'ball_types': shot_type,
                    'lose_reason': "opponent " + reason,
                    'roundscore_A': self.score_a,
                    'roundscore_B': self.score_b
                }
                
                self.rallies.append(rally)
                print(f"✅ Rally {len(self.rallies)}: {rally['start_time']}-{rally['end_time']} | "
                      f"{winner_name} | {shot_type} | {self.score_a}-{self.score_b}")
                
                rally_active = False
                rally_trajectory = []
            
            frame_count += 1
            
            if frame_count % 600 == 0:
                print(f"⏳ Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        
        # Save results
        df = pd.DataFrame(self.rallies)
        df.to_csv('badminton_hybrid_results.csv', index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ COMPLETE! Found {len(self.rallies)} rallies")
        print(f"🏆 Final: {self.score_a}-{self.score_b}")
        print(f"💾 Saved to badminton_hybrid_results.csv")
        print(f"{'='*80}\n")
        
        print(df.to_string(index=False))
        
        return self.rallies
    
    def format_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    # Use GPU 1 (the idle one with only 6MB used)
    extractor = HybridBadmintonExtractor(device="cuda:1")
    
    extractor.process_video(
        video_path="./Sample.mp4",
        player_a="An Se Young",
        player_b="Ratchanok Intanon"
    )

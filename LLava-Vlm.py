#!/usr/bin/env python3
"""
Badminton Rally Extractor using LLaVA Vision-Language Model
Optimized for GTX 1080 Ti (11GB VRAM)
"""

import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datetime import timedelta
import os
from tqdm import tqdm
import json
import re

class BadmintonVLMExtractor:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", device="cuda:1"):
        """
        Initialize VLM-based badminton rally extractor
        
        Args:
            model_name: LLaVA model to use
            device: GPU device (cuda:0 or cuda:1)
        """
        self.device = device
        print(f"🚀 Loading {model_name} on {device}...")
        
        # Load model with 8-bit quantization to fit in 11GB VRAM
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device,
            low_cpu_mem_usage=True
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"✅ Model loaded successfully on {device}")
        
        self.rallies = []
        self.score_a = 0
        self.score_b = 0
        self.player_a_name = "Player A"
        self.player_b_name = "Player B"
    
    def format_time(self, seconds):
        """Convert seconds to MM:SS format"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    
    def extract_frames_from_video(self, video_path, fps_sample=2):
        """
        Extract frames from video at specified sampling rate
        
        Args:
            video_path: Path to video file
            fps_sample: Extract 1 frame every N seconds
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        print(f"📹 Video Info: {video_fps:.1f} FPS, {duration:.1f}s duration")
        
        frames = []
        timestamps = []
        frame_interval = int(video_fps * fps_sample)
        
        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                timestamps.append(frame_count / video_fps)
            
            frame_count += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        print(f"✅ Extracted {len(frames)} frames")
        return frames, timestamps
    
    def analyze_frame(self, image, timestamp):
        """
        Analyze a single frame using VLM
        
        Returns structured information about the rally state
        """
        prompt = f"""<image>
You are analyzing a badminton match. Look at this frame and answer:

1. Is there an active rally happening? (yes/no)
2. Can you see the shuttlecock? Where is it? (court_left/court_right/net/out/not_visible)
3. Which player just hit? (left_player/right_player/unknown)
4. What type of shot was it? (smash/clear/drop/net_shot/drive/lob/push/lift/unknown)
5. Did the rally just end? (yes/no)
6. If ended, what was the outcome? (winner_left/winner_right/net/out_of_bounds/ongoing)

Format your answer as JSON:
{{
    "rally_active": true/false,
    "shuttlecock_position": "...",
    "last_hitter": "...",
    "shot_type": "...",
    "rally_ended": true/false,
    "outcome": "..."
}}"""

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = self._parse_text_response(response)
        except:
            analysis = {
                "rally_active": False,
                "shuttlecock_position": "not_visible",
                "last_hitter": "unknown",
                "shot_type": "unknown",
                "rally_ended": False,
                "outcome": "ongoing"
            }
        
        return analysis
    
    def _parse_text_response(self, text):
        """Fallback parser for non-JSON responses"""
        analysis = {
            "rally_active": "yes" in text.lower() or "active" in text.lower(),
            "shuttlecock_position": "not_visible",
            "last_hitter": "unknown",
            "shot_type": "push",
            "rally_ended": "end" in text.lower(),
            "outcome": "ongoing"
        }
        
        # Extract shot type
        shot_types = ["smash", "clear", "drop", "net_shot", "drive", "lob", "push", "lift"]
        for shot in shot_types:
            if shot in text.lower():
                analysis["shot_type"] = shot
                break
        
        return analysis
    
    def detect_rallies(self, frames, timestamps):
        """
        Process all frames to detect and extract rally information
        """
        print("\n🎾 Analyzing frames with VLM...")
        
        rally_states = []
        current_rally = None
        
        for i, (frame, timestamp) in enumerate(tqdm(zip(frames, timestamps), total=len(frames))):
            analysis = self.analyze_frame(frame, timestamp)
            rally_states.append({
                'timestamp': timestamp,
                'analysis': analysis
            })
            
            # Rally state machine
            if analysis['rally_active'] and current_rally is None:
                # Start new rally
                current_rally = {
                    'start_time': timestamp,
                    'end_time': None,
                    'shots': [],
                    'last_hitter': None
                }
            
            if current_rally is not None:
                # Track shots
                if analysis['shot_type'] != 'unknown':
                    current_rally['shots'].append({
                        'time': timestamp,
                        'type': analysis['shot_type'],
                        'hitter': analysis['last_hitter']
                    })
                    current_rally['last_hitter'] = analysis['last_hitter']
                
                # Rally ended
                if analysis['rally_ended'] or not analysis['rally_active']:
                    current_rally['end_time'] = timestamp
                    current_rally['outcome'] = analysis['outcome']
                    self._finalize_rally(current_rally)
                    current_rally = None
        
        return rally_states
    
    def _finalize_rally(self, rally_data):
        """Convert rally data to final format"""
        if rally_data['end_time'] is None:
            return
        
        # Determine winner and reasons
        outcome = rally_data['outcome']
        last_shot = rally_data['shots'][-1] if rally_data['shots'] else None
        shot_type = last_shot['type'] if last_shot else 'push'
        
        if 'winner_left' in outcome:
            winner = self.player_a_name
            loser = self.player_b_name
            win_reason = "opponent error"
            lose_reason = "error"
            self.score_a += 1
        elif 'winner_right' in outcome:
            winner = self.player_b_name
            loser = self.player_a_name
            win_reason = "opponent error"
            lose_reason = "error"
            self.score_b += 1
        elif 'net' in outcome:
            winner = self.player_b_name if rally_data['last_hitter'] == 'left_player' else self.player_a_name
            win_reason = "opponent hits the net"
            lose_reason = "hits the net"
            if winner == self.player_a_name:
                self.score_a += 1
            else:
                self.score_b += 1
        elif 'out' in outcome:
            winner = self.player_b_name if rally_data['last_hitter'] == 'left_player' else self.player_a_name
            win_reason = "opponent goes out of bounds"
            lose_reason = "goes out of bounds"
            if winner == self.player_a_name:
                self.score_a += 1
            else:
                self.score_b += 1
        else:
            winner = self.player_a_name
            win_reason = "wins by landing"
            lose_reason = "opponent wins by landing"
            self.score_a += 1
        
        rally = {
            'start_time': self.format_time(rally_data['start_time']),
            'end_time': self.format_time(rally_data['end_time']),
            'win_point_player': winner,
            'win_reason': win_reason,
            'ball_types': shot_type,
            'lose_reason': lose_reason,
            'roundscore_A': self.score_a,
            'roundscore_B': self.score_b
        }
        
        self.rallies.append(rally)
        print(f"✅ Rally {len(self.rallies)}: {rally['start_time']}-{rally['end_time']} | "
              f"Winner: {winner} | Shot: {shot_type} | Score: {self.score_a}-{self.score_b}")
    
    def process_video(self, video_path, player_a="Player A", player_b="Player B", fps_sample=2):
        """
        Main processing pipeline
        
        Args:
            video_path: Path to badminton video
            player_a: Name of left player
            player_b: Name of right player
            fps_sample: Sample 1 frame every N seconds
        """
        self.player_a_name = player_a
        self.player_b_name = player_b
        self.rallies = []
        self.score_a = 0
        self.score_b = 0
        
        print(f"\n{'='*80}")
        print(f"🎾 BADMINTON RALLY EXTRACTION WITH VLM")
        print(f"{'='*80}")
        print(f"Players: {player_a} vs {player_b}")
        print(f"Video: {video_path}")
        
        # Extract frames
        frames, timestamps = self.extract_frames_from_video(video_path, fps_sample)
        
        # Detect rallies
        rally_states = self.detect_rallies(frames, timestamps)
        
        # Save results
        output_file = self.save_results()
        
        print(f"\n{'='*80}")
        print(f"✅ PROCESSING COMPLETE!")
        print(f"📊 Total rallies: {len(self.rallies)}")
        print(f"🏆 Final Score: {self.score_a} - {self.score_b}")
        print(f"💾 Results saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return self.rallies
    
    def save_results(self, output_path='badminton_vlm_results.csv'):
        """Save results to CSV"""
        df = pd.DataFrame(self.rallies)
        df.to_csv(output_path, index=False)
        
        # Print preview
        print(f"\n📋 Results Preview:")
        print(df.to_string(index=False))
        
        return output_path


def main():
    """Main execution function"""
    
    # Configuration
    VIDEO_PATH = "Sample.mp4"  # Change to your video path
    PLAYER_A = "An Se Young"
    PLAYER_B = "Ratchanok Intanon"
    FPS_SAMPLE = 2  # Sample 1 frame every 2 seconds
    
    # Verify GPU
    print(f"🔍 Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize extractor (cuda:0 because we filtered with CUDA_VISIBLE_DEVICES)
    extractor = BadmintonVLMExtractor(
        model_name="llava-hf/llava-1.5-7b-hf",
        device="cuda:0"  # This is actually GPU 1 from your system!
    )
    
    # Process video
    results = extractor.process_video(
        video_path=VIDEO_PATH,
        player_a=PLAYER_A,
        player_b=PLAYER_B,
        fps_sample=FPS_SAMPLE
    )
    
    print(f"\n🎉 Extraction complete! Found {len(results)} rallies")


if __name__ == "__main__":
    # IMPORTANT: Set this to use only GPU 1 (your idle GPU)
    # After this line, GPU 1 becomes "cuda:0" in the program
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()

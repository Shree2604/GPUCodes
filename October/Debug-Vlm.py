#!/usr/bin/env python3
"""
Diagnostic script to debug why rallies aren't being detected
"""

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from ultralytics import YOLO
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def test_shuttlecock_detection(video_path, num_frames=10):
    """Test if shuttlecock is being detected"""
    print("\n" + "="*80)
    print("🔍 TESTING SHUTTLECOCK DETECTION")
    print("="*80)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {fps} FPS, {total_frames} frames")
    
    detections = []
    frame_interval = total_frames // num_frames
    
    for i in range(num_frames):
        frame_num = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Simple color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White shuttlecock
        mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:
                detected = True
                break
        
        detections.append(detected)
        status = "✅ DETECTED" if detected else "❌ NOT FOUND"
        print(f"Frame {frame_num:4d} ({frame_num/fps:.1f}s): {status}")
    
    cap.release()
    
    detection_rate = sum(detections) / len(detections) * 100
    print(f"\n📊 Detection Rate: {detection_rate:.1f}% ({sum(detections)}/{len(detections)} frames)")
    
    if detection_rate < 30:
        print("⚠️  WARNING: Low detection rate! Shuttlecock might be:")
        print("   - Yellow/neon colored (not white)")
        print("   - Too small in frame")
        print("   - Moving too fast (motion blur)")
    
    return detection_rate

def test_vlm_understanding(video_path, device="cuda:0"):
    """Test if VLM understands badminton scenes"""
    print("\n" + "="*80)
    print("🤖 TESTING VLM UNDERSTANDING")
    print("="*80)
    
    print("Loading VLM...")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print("✅ Model loaded!\n")
    
    # Extract 3 test frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    test_frames = [
        total_frames // 4,    # 25% into video
        total_frames // 2,    # 50% into video
        3 * total_frames // 4 # 75% into video
    ]
    
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Simple prompt
        prompt = """<image>
USER: What sport is being played in this image? Describe what you see in 2-3 sentences.
ASSISTANT:"""
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        print(f"\n📍 Frame {frame_num} ({frame_num/cap.get(cv2.CAP_PROP_FPS):.1f}s):")
        print(f"   Response: {response[:200]}")
        
        # Check if it recognizes badminton
        keywords = ['badminton', 'shuttlecock', 'racket', 'court', 'net']
        found_keywords = [kw for kw in keywords if kw.lower() in response.lower()]
        
        if found_keywords:
            print(f"   ✅ Recognized: {', '.join(found_keywords)}")
        else:
            print(f"   ❌ No badminton keywords found!")
    
    cap.release()

def test_rally_logic(video_path):
    """Test rally detection logic"""
    print("\n" + "="*80)
    print("🎾 TESTING RALLY LOGIC")
    print("="*80)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    frames_with_shuttle = 0
    frames_without_shuttle = 0
    max_consecutive_detections = 0
    current_consecutive = 0
    
    rally_threshold = int(fps * 0.2)  # 0.2 seconds = 6 frames
    end_threshold = int(fps * 1.5)     # 1.5 seconds = 45 frames
    
    print(f"Rally start threshold: {rally_threshold} frames")
    print(f"Rally end threshold: {end_threshold} frames")
    print()
    
    potential_rallies = []
    rally_start = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = False
        for contour in contours:
            if 10 < cv2.contourArea(contour) < 500:
                detected = True
                break
        
        if detected:
            frames_with_shuttle += 1
            frames_without_shuttle = 0
            current_consecutive += 1
            max_consecutive_detections = max(max_consecutive_detections, current_consecutive)
            
            if rally_start is None and current_consecutive >= rally_threshold:
                rally_start = frame_count
        else:
            frames_without_shuttle += 1
            current_consecutive = 0
            
            if rally_start is not None and frames_without_shuttle >= end_threshold:
                potential_rallies.append({
                    'start': rally_start,
                    'end': frame_count,
                    'duration': (frame_count - rally_start) / fps
                })
                print(f"✅ Potential rally: {rally_start/fps:.1f}s - {frame_count/fps:.1f}s "
                      f"(duration: {(frame_count - rally_start)/fps:.1f}s)")
                rally_start = None
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n📊 Statistics:")
    print(f"   Total frames: {total_frames}")
    print(f"   Frames with shuttlecock: {frames_with_shuttle} ({frames_with_shuttle/total_frames*100:.1f}%)")
    print(f"   Max consecutive detections: {max_consecutive_detections} frames ({max_consecutive_detections/fps:.2f}s)")
    print(f"   Potential rallies found: {len(potential_rallies)}")
    
    if len(potential_rallies) == 0:
        print("\n❌ NO RALLIES DETECTED!")
        print("   Possible reasons:")
        if max_consecutive_detections < rally_threshold:
            print(f"   ⚠️  Max consecutive detections ({max_consecutive_detections}) < threshold ({rally_threshold})")
            print(f"      Suggestion: Lower rally_threshold to {max(1, max_consecutive_detections // 2)}")
        if frames_with_shuttle < total_frames * 0.1:
            print(f"   ⚠️  Very low detection rate ({frames_with_shuttle/total_frames*100:.1f}%)")
            print(f"      Suggestion: Adjust color thresholds or use YOLO object detection")

def main():
    VIDEO_PATH = "Sample.mp4"
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        return
    
    print("\n" + "="*80)
    print("🔬 BADMINTON DETECTION DIAGNOSTIC TOOL")
    print("="*80)
    print(f"Video: {VIDEO_PATH}\n")
    
    # Test 1: Shuttlecock detection
    detection_rate = test_shuttlecock_detection(VIDEO_PATH)
    
    # Test 2: Rally logic
    test_rally_logic(VIDEO_PATH)
    
    # Test 3: VLM understanding (only if detection rate is reasonable)
    if detection_rate > 20:
        response = input("\n🤖 Test VLM understanding? This will load the model. (y/n): ")
        if response.lower() == 'y':
            test_vlm_understanding(VIDEO_PATH)
    else:
        print("\n⚠️  Skipping VLM test due to low detection rate")
        print("   Fix shuttlecock detection first!")
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDATIONS:")
    print("="*80)
    print("1. If shuttlecock detection is low (<30%):")
    print("   - Try YOLO-based detection instead of color")
    print("   - Adjust HSV color ranges for your shuttlecock color")
    print()
    print("2. If no rallies detected despite good detection:")
    print("   - Lower rally start threshold")
    print("   - Lower rally end threshold")
    print()
    print("3. If VLM doesn't recognize badminton:")
    print("   - Simplify prompts")
    print("   - Use multi-frame context")
    print("="*80)

if __name__ == "__main__":
    main()

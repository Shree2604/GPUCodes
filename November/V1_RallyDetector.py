"""
Complete Rally Detection System for Badminton Videos
Phase 1: Rally Time Detection Only
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
from enum import Enum
import argparse
import os
from datetime import datetime


class RallyState(Enum):
    """States for rally detection state machine"""
    IDLE = 0           # No rally happening
    RALLY_ACTIVE = 1   # Rally in progress
    RALLY_ENDING = 2   # Potential end detected


class ShuttlecockDetector:
    """
    Detects shuttlecock in frames using YOLO + color filtering
    
    SETUP OPTIONS:
    1. Roboflow Model (DEFAULT - Pre-configured):
       - Uses shuttlecock-cqzy3/1 model automatically
       - No configuration needed!
    
    2. Custom Roboflow Model:
       - Provide your own roboflow_config with API key and model details
       
    3. Local YOLO Model:
       - Download from Roboflow and provide model_path
       
    4. Fallback Color Detection:
       - Always used as backup if YOLO fails
    """
    
    def __init__(self, use_yolo=True, confidence_threshold=0.4, 
                 use_roboflow=True, roboflow_config=None, model_path=None):
        self.use_yolo = use_yolo
        self.confidence_threshold = confidence_threshold
        self.use_roboflow = use_roboflow
        self.model = None
        
        # Default Roboflow configuration (pre-configured shuttlecock model)
        if roboflow_config is None and use_roboflow:
            roboflow_config = {
                'api_key': 'dgMdKMJNrUwGlaZ9MB0h',
                'model_id': 'shuttlecock-cqzy3',
                'version': 1
            }
            print("✓ Using default Roboflow shuttlecock model (shuttlecock-cqzy3/1)")
        
        if use_roboflow and roboflow_config:
            # Use Roboflow API
            try:
                from roboflow import Roboflow
                rf = Roboflow(api_key=roboflow_config['api_key'])
                project = rf.workspace().project(roboflow_config['model_id'])
                self.model = project.version(roboflow_config['version']).model
                self.roboflow_mode = True
                print(f"✓ Loaded Roboflow shuttlecock model: {roboflow_config['model_id']}/v{roboflow_config['version']}")
                print("  (API-based inference - requires internet)")
            except ImportError:
                print("⚠ Roboflow package not installed. Install with: pip install roboflow")
                print("  Falling back to color detection")
                self.use_yolo = False
                self.roboflow_mode = False
            except Exception as e:
                print(f"⚠ Roboflow API failed: {e}")
                print("  Falling back to color detection")
                self.use_yolo = False
                self.roboflow_mode = False
        
        elif use_yolo:
            try:
                from ultralytics import YOLO
                # Try to load custom shuttlecock model
                if model_path and os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    print(f"✓ Loaded custom shuttlecock YOLO model: {model_path}")
                else:
                    self.model = YOLO('shuttlecock.pt')
                    print("✓ Loaded custom shuttlecock YOLO model")
                self.roboflow_mode = False
            except:
                # Fallback to YOLOv8n (detect sports ball class)
                try:
                    from ultralytics import YOLO
                    self.model = YOLO('yolov8n.pt')
                    print("✓ Using YOLOv8n (will detect as sports ball)")
                    self.roboflow_mode = False
                except:
                    print("⚠ No YOLO model available, using color detection only")
                    self.use_yolo = False
                    self.roboflow_mode = False
        else:
            self.roboflow_mode = False
            print("✓ Using color-based detection only")
        
        # HSV range for white shuttlecock
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        
    def detect_yolo(self, frame):
        """Detect using YOLO model (local or Roboflow)"""
        if self.roboflow_mode:
            # Roboflow API detection
            try:
                # Save frame temporarily
                temp_path = 'temp_frame.jpg'
                cv2.imwrite(temp_path, frame)
                
                # Get predictions from Roboflow
                # Convert confidence to percentage (Roboflow expects 0-100)
                confidence_pct = int(self.confidence_threshold * 100)
                response = self.model.predict(temp_path, confidence=confidence_pct, overlap=50)
                predictions = response.json()
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                detections = []
                if 'predictions' in predictions and predictions['predictions']:
                    for pred in predictions['predictions']:
                        x, y = pred['x'], pred['y']
                        w, h = pred['width'], pred['height']
                        
                        detections.append({
                            'bbox': [int(x - w/2), int(y - h/2), int(w), int(h)],
                            'center': (int(x), int(y)),
                            'confidence': pred['confidence']
                        })
                
                return detections if detections else None
                
            except Exception as e:
                print(f"Roboflow API error: {e}")
                return None
        
        else:
            # Local YOLO detection
            from ultralytics import YOLO
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Look for sports ball (class 32) or shuttlecock (if custom model)
        detections = []
        for result in results:
            for box in result.boxes:
                # Filter for small fast-moving objects in upper part of frame
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                
                # Shuttlecock characteristics: small, aspect ratio ~1
                if w < 100 and h < 100 and 0.5 < w/h < 2.0:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(w), int(h)],
                        'center': (int((x1+x2)/2), int((y1+y2)/2)),
                        'confidence': float(box.conf)
                    })
        
        return detections if detections else None
    
    def detect_color(self, frame):
        """Detect white shuttlecock using HSV color filtering"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to reduce noise
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and circularity
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 500:  # Shuttlecock size range
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.3:  # Reasonably circular
                        x, y, w, h = cv2.boundingRect(cnt)
                        detections.append({
                            'bbox': [x, y, w, h],
                            'center': (x + w//2, y + h//2),
                            'confidence': circularity
                        })
        
        return detections if detections else None
    
    def detect(self, frame):
        """
        Main detection method - tries both approaches
        Returns: dict with detection info or None
        """
        detections = []
        
        # Try YOLO first
        if self.use_yolo:
            yolo_detections = self.detect_yolo(frame)
            if yolo_detections:
                detections.extend(yolo_detections)
        
        # Add color-based detections
        color_detections = self.detect_color(frame)
        if color_detections:
            detections.extend(color_detections)
        
        if not detections:
            return None
        
        # Return detection with highest confidence
        best_detection = max(detections, key=lambda x: x['confidence'])
        return {
            'detected': True,
            'bbox': best_detection['bbox'],
            'center': best_detection['center'],
            'confidence': best_detection['confidence']
        }


class RallyDetector:
    """
    Main rally detection class using state machine logic
    
    Logic:
    - START: Shuttlecock appears AND is moving
    - END: Shuttlecock stationary >1s OR disappears >1s
    """
    
    def __init__(self, 
                 min_rally_duration=2.0,
                 end_timeout=1.5,
                 movement_threshold=10,
                 fps=30,
                 detector_config=None):
        
        # Initialize shuttlecock detector with config
        if detector_config:
            self.shuttle_detector = ShuttlecockDetector(**detector_config)
        else:
            self.shuttle_detector = ShuttlecockDetector()
        
        self.state = RallyState.IDLE
        
        # Configuration
        self.min_rally_duration = min_rally_duration  # seconds
        self.end_timeout = end_timeout                 # seconds
        self.movement_threshold = movement_threshold   # pixels
        self.fps = fps
        
        # Tracking variables
        self.rally_start_time = None
        self.rally_start_frame = None
        self.last_shuttle_seen = None
        self.shuttle_positions = deque(maxlen=5)  # Keep last 5 positions
        
        # Statistics
        self.rally_count = 0
        
    def is_shuttlecock_moving(self):
        """Check if shuttlecock is moving based on position history"""
        if len(self.shuttle_positions) < 3:
            return True  # Assume moving if not enough data
        
        # Calculate movement between consecutive positions
        movements = []
        positions = list(self.shuttle_positions)
        
        for i in range(len(positions) - 1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            distance = np.sqrt(dx**2 + dy**2)
            movements.append(distance)
        
        avg_movement = np.mean(movements)
        return avg_movement > self.movement_threshold
    
    def process_frame(self, frame, frame_number, timestamp):
        """
        Process single frame and return rally events
        
        Returns: 
            ('start', timestamp, frame_number) or 
            ('end', timestamp, frame_number) or 
            None
        """
        # Detect shuttlecock
        shuttle = self.shuttle_detector.detect(frame)
        
        # State machine logic
        if self.state == RallyState.IDLE:
            # Looking for rally start
            if shuttle and shuttle['detected']:
                self.state = RallyState.RALLY_ACTIVE
                self.rally_start_time = timestamp
                self.rally_start_frame = frame_number
                self.last_shuttle_seen = timestamp
                self.shuttle_positions.clear()
                self.shuttle_positions.append(shuttle['center'])
                
                self.rally_count += 1
                return ('start', timestamp, frame_number)
        
        elif self.state == RallyState.RALLY_ACTIVE:
            # Rally in progress
            if shuttle and shuttle['detected']:
                self.last_shuttle_seen = timestamp
                self.shuttle_positions.append(shuttle['center'])
                
                # Check if shuttlecock stopped moving
                if len(self.shuttle_positions) >= 3 and not self.is_shuttlecock_moving():
                    self.state = RallyState.RALLY_ENDING
                    
            else:
                # Shuttlecock disappeared
                time_since_last = timestamp - self.last_shuttle_seen
                if time_since_last > self.end_timeout:
                    return self._end_rally(timestamp, frame_number)
        
        elif self.state == RallyState.RALLY_ENDING:
            # Confirming rally end
            if shuttle and shuttle['detected'] and self.is_shuttlecock_moving():
                # False alarm - rally continues
                self.state = RallyState.RALLY_ACTIVE
                self.shuttle_positions.append(shuttle['center'])
            else:
                # Confirmed end
                time_since_last = timestamp - self.last_shuttle_seen
                if time_since_last > 0.5:  # Give 0.5s grace period
                    return self._end_rally(timestamp, frame_number)
        
        return None
    
    def _end_rally(self, timestamp, frame_number):
        """End current rally and reset state"""
        if self.rally_start_time is None:
            self._reset()
            return None
        
        duration = timestamp - self.rally_start_time
        
        # Filter out too-short rallies (likely false positives)
        if duration < self.min_rally_duration:
            self._reset()
            return None
        
        result = ('end', timestamp, frame_number)
        self._reset()
        return result
    
    def _reset(self):
        """Reset state for next rally"""
        self.state = RallyState.IDLE
        self.rally_start_time = None
        self.rally_start_frame = None
        self.last_shuttle_seen = None
        self.shuttle_positions.clear()


def format_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def process_video(video_path, output_csv='rallies_output.csv', visualize=False, 
                  output_video=None, detector_config=None):
    """
    Main video processing function
    
    Args:
        video_path: Path to input video
        output_csv: Path to output CSV file
        visualize: If True, show live detection
        output_video: If provided, save annotated video
        detector_config: Configuration dict for ShuttlecockDetector
    
    Returns:
        DataFrame with rally timings
    """
    
    print("\n" + "="*60)
    print("RALLY DETECTION PIPELINE - Phase 1")
    print("="*60)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    print(f"\nVideo Info:")
    print(f"  Path: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {format_time(duration)} ({duration:.1f}s)")
    print(f"  Total Frames: {total_frames}")
    
    # Initialize detector
    detector = RallyDetector(fps=fps, detector_config=detector_config)
    
    # Output video writer
    out_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Rally tracking
    rallies = []
    current_rally = None
    
    frame_count = 0
    print(f"\nProcessing video...")
    print(f"[Press 'q' to stop if visualizing]\n")
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / fps
        
        # Process frame
        event = detector.process_frame(frame, frame_count, timestamp)
        
        # Handle events
        if event:
            event_type, event_time, event_frame = event
            
            if event_type == 'start':
                current_rally = {
                    'start_time': event_time,
                    'start_frame': event_frame
                }
                print(f"✓ Rally #{detector.rally_count} START at {format_time(event_time)}")
            
            elif event_type == 'end' and current_rally:
                current_rally['end_time'] = event_time
                current_rally['end_frame'] = event_frame
                current_rally['duration'] = event_time - current_rally['start_time']
                rallies.append(current_rally)
                print(f"  Rally #{len(rallies)} END at {format_time(event_time)} (duration: {current_rally['duration']:.1f}s)")
                current_rally = None
        
        # Visualization
        if visualize or output_video:
            vis_frame = frame.copy()
            
            # Draw status
            status_color = (0, 255, 0) if detector.state == RallyState.RALLY_ACTIVE else (128, 128, 128)
            status_text = f"State: {detector.state.name} | Rally: {detector.rally_count}"
            cv2.putText(vis_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.putText(vis_frame, format_time(timestamp), (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw shuttlecock positions
            for pos in detector.shuttle_positions:
                cv2.circle(vis_frame, pos, 3, (0, 255, 255), -1)
            
            if output_video:
                out_writer.write(vis_frame)
            
            if visualize:
                cv2.imshow('Rally Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Progress update
        if frame_count % int(fps * 10) == 0:  # Every 10 seconds
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({format_time(timestamp)}/{format_time(duration)})")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    if out_writer:
        out_writer.release()
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"\n" + "="*60)
    print(f"DETECTION COMPLETE")
    print(f"="*60)
    print(f"Total rallies detected: {len(rallies)}")
    
    if len(rallies) == 0:
        print("\n⚠ WARNING: No rallies detected!")
        print("  Possible issues:")
        print("  - Video quality too low")
        print("  - Shuttlecock not visible")
        print("  - Need to adjust detection thresholds")
        return None
    
    # Create DataFrame
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    df = pd.DataFrame(rallies)
    df['video_id'] = video_id
    df['rally_id'] = range(1, len(df) + 1)
    df['start_time_formatted'] = df['start_time'].apply(format_time)
    df['end_time_formatted'] = df['end_time'].apply(format_time)
    df['duration'] = df['duration'].round(1)
    
    # Reorder columns
    output_df = df[['video_id', 'rally_id', 'start_time_formatted', 
                     'end_time_formatted', 'duration']]
    output_df.columns = ['video_id', 'rally_id', 'start_time', 'end_time', 'duration']
    
    # Save CSV
    output_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    # Print summary statistics
    print(f"\nRally Statistics:")
    print(f"  Average duration: {df['duration'].mean():.1f}s")
    print(f"  Shortest rally: {df['duration'].min():.1f}s")
    print(f"  Longest rally: {df['duration'].max():.1f}s")
    
    # Print first few rallies
    print(f"\nFirst 5 rallies:")
    print(output_df.head().to_string(index=False))
    
    return output_df


def evaluate_predictions(predicted_csv, ground_truth_csv):
    """
    Evaluate rally detection against ground truth
    
    Metrics:
    - Precision: % of detected rallies that are correct
    - Recall: % of actual rallies that were detected
    - Temporal IoU: Overlap of time boundaries
    """
    pred_df = pd.read_csv(predicted_csv)
    gt_df = pd.read_csv(ground_truth_csv)
    
    # Convert time strings to seconds
    def time_to_seconds(time_str):
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    
    pred_df['start_sec'] = pred_df['start_time'].apply(time_to_seconds)
    pred_df['end_sec'] = pred_df['end_time'].apply(time_to_seconds)
    gt_df['start_sec'] = gt_df['start_time'].apply(time_to_seconds)
    gt_df['end_sec'] = gt_df['end_time'].apply(time_to_seconds)
    
    # Calculate IoU for each pair
    def calculate_iou(pred_start, pred_end, gt_start, gt_end):
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        intersection = max(0, intersection_end - intersection_start)
        
        union_start = min(pred_start, gt_start)
        union_end = max(pred_end, gt_end)
        union = union_end - union_start
        
        return intersection / union if union > 0 else 0
    
    matched_pred = set()
    matched_gt = set()
    ious = []
    
    # Match predictions to ground truth
    for i, pred_row in pred_df.iterrows():
        best_iou = 0
        best_gt_idx = None
        
        for j, gt_row in gt_df.iterrows():
            if j in matched_gt:
                continue
            
            iou = calculate_iou(
                pred_row['start_sec'], pred_row['end_sec'],
                gt_row['start_sec'], gt_row['end_sec']
            )
            
            if iou > best_iou and iou > 0.5:  # Threshold for match
                best_iou = iou
                best_gt_idx = j
        
        if best_gt_idx is not None:
            matched_pred.add(i)
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
    
    # Calculate metrics
    precision = len(matched_pred) / len(pred_df) if len(pred_df) > 0 else 0
    recall = len(matched_gt) / len(gt_df) if len(gt_df) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(ious) if ious else 0
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Ground Truth Rallies: {len(gt_df)}")
    print(f"Predicted Rallies: {len(pred_df)}")
    print(f"Matched Rallies: {len(matched_pred)}")
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Average IoU: {avg_iou:.3f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou
    }


def main():
    parser = argparse.ArgumentParser(
        description='Badminton Rally Detection - Phase 1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Easiest - Use default Roboflow model (pre-configured)
  python rally_detector_complete.py --video match.mp4
  
  # With visualization
  python rally_detector_complete.py --video match.mp4 --visualize
  
  # Use custom Roboflow model
  python rally_detector_complete.py --video match.mp4 --roboflow \\
      --api-key YOUR_KEY --model-id your-model --rf-version 1
  
  # Use local downloaded model
  python rally_detector_complete.py --video match.mp4 \\
      --model-path ./weights/best.pt
  
  # Disable YOLO, use color detection only
  python rally_detector_complete.py --video match.mp4 --no-yolo
        """
    )
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='rallies_output.csv', 
                       help='Output CSV file path')
    parser.add_argument('--visualize', action='store_true', 
                       help='Show live visualization')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save annotated video to this path')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Ground truth CSV for evaluation')
    
    # Roboflow options
    parser.add_argument('--roboflow', action='store_true',
                       help='Use Roboflow API for detection (default: True if no other model specified)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Roboflow API key (default: uses pre-configured shuttlecock-cqzy3 model)')
    parser.add_argument('--model-id', type=str, default=None,
                       help='Roboflow model ID (default: shuttlecock-cqzy3)')
    parser.add_argument('--rf-version', type=int, default=1,
                       help='Roboflow model version number (default: 1)')
    
    # Local model options
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to local YOLO model weights')
    parser.add_argument('--no-yolo', action='store_true',
                       help='Use only color detection (no YOLO)')
    
    args = parser.parse_args()
    
    # Configure detector based on arguments
    # Default to Roboflow if no other option specified
    use_roboflow = args.roboflow or (not args.model_path and not args.no_yolo)
    
    detector_config = {
        'use_yolo': not args.no_yolo,
        'use_roboflow': use_roboflow,
        'roboflow_config': None,
        'model_path': args.model_path
    }
    
    # Custom Roboflow config if provided
    if use_roboflow and (args.api_key or args.model_id):
        if not args.api_key or not args.model_id:
            print("⚠ Warning: When providing custom Roboflow model, both --api-key and --model-id are required")
            print("  Using default shuttlecock-cqzy3 model instead")
        else:
            detector_config['roboflow_config'] = {
                'api_key': args.api_key,
                'model_id': args.model_id,
                'version': args.rf_version
            }
    
    print("\n" + "="*60)
    print("RALLY DETECTION SYSTEM - Phase 1")
    print("="*60)
    print("\nConfiguration:")
    if detector_config['use_roboflow']:
        if detector_config['roboflow_config']:
            print(f"  Model: Roboflow {detector_config['roboflow_config']['model_id']}")
        else:
            print(f"  Model: Roboflow shuttlecock-cqzy3/1 (default)")
        print(f"  Mode: Cloud-based inference")
    elif detector_config['model_path']:
        print(f"  Model: Local YOLO - {detector_config['model_path']}")
        print(f"  Mode: Local inference")
    elif args.no_yolo:
        print(f"  Model: Color-based detection only")
        print(f"  Mode: HSV filtering")
    print()
    
    # Process video with custom detector config
    result_df = process_video(
        video_path=args.video,
        output_csv=args.output,
        visualize=args.visualize,
        output_video=args.save_video,
        detector_config=detector_config
    )
    
    # Evaluate if ground truth provided
    if args.ground_truth and result_df is not None:
        evaluate_predictions(args.output, args.ground_truth)


if __name__ == "__main__":
    # Quick usage examples (uncomment to use):
    
    # Example 1: Use default Roboflow model (simplest - just works!)
    # process_video(
    #     video_path='your_match.mp4',
    #     output_csv='rallies_output.csv',
    #     visualize=True
    # )
    
    # Example 2: Use custom Roboflow model
    # detector_config = {
    #     'use_roboflow': True,
    #     'roboflow_config': {
    #         'api_key': 'YOUR_API_KEY',
    #         'model_id': 'your-model-id',
    #         'version': 1
    #     }
    # }
    # process_video(
    #     video_path='your_match.mp4',
    #     detector_config=detector_config,
    #     visualize=True
    # )
    
    # Example 3: Use local YOLO model
    # detector_config = {
    #     'use_yolo': True,
    #     'use_roboflow': False,
    #     'model_path': './weights/best.pt'
    # }
    # process_video(
    #     video_path='your_match.mp4',
    #     detector_config=detector_config
    # )
    
    # Run command-line interface
    main()

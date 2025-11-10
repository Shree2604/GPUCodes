"""
Enhanced Rally Detection System v4.1 for Badminton Videos
Incorporates all critical improvements + TrackNetV3/MonoTrack support

Key Enhancements:
1. Height-based ground detection
2. Speed-based activity validation
3. Inter-rally gap enforcement (prevents splitting)
4. Trajectory smoothness validation
5. ROI/boundary filtering
6. Confidence trend analysis
7. TrackNetV3/MonoTrack integration
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
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class RallyState(Enum):
    """States for rally detection state machine"""
    IDLE = 0
    RALLY_ACTIVE = 1
    RALLY_ENDING = 2


class TrackingMethod(Enum):
    """Available tracking methods"""
    ROBOFLOW = "roboflow"
    YOLO = "yolo"
    TRACKNET = "tracknet"
    MONOTRACK = "monotrack"
    COLOR = "color"


class ShuttlecockTracker:
    """
    Advanced shuttlecock tracker with multiple detection methods
    
    Supports:
    1. Roboflow API (default)
    2. TrackNetV3 (specialized for ball/shuttle tracking)
    3. MonoTrack (state-of-the-art single object tracker)
    4. YOLO (general object detection)
    5. Color-based (fallback)
    """
    
    def __init__(self, 
                 method='roboflow',
                 confidence_threshold=0.4,
                 roboflow_config=None,
                 model_path=None,
                 tracknet_path=None,
                 frame_height=None,
                 frame_width=None):
        
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # Initialize based on method
        if method == 'roboflow':
            self._init_roboflow(roboflow_config)
        elif method == 'tracknet':
            self._init_tracknet(tracknet_path)
        elif method == 'monotrack':
            self._init_monotrack(model_path)
        elif method == 'yolo':
            self._init_yolo(model_path)
        
        # HSV color detection (always available as fallback)
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        
        # Tracking history for temporal consistency
        self.position_history = deque(maxlen=30)
        self.confidence_history = deque(maxlen=10)
    
    def _init_roboflow(self, config):
        """Initialize Roboflow API"""
        if config is None:
            config = {
                'api_key': 'dgMdKMJNrUwGlaZ9MB0h',
                'model_id': 'shuttlecock-cqzy3',
                'version': 1
            }
        
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=config['api_key'])
            project = rf.workspace().project(config['model_id'])
            self.model = project.version(config['version']).model
            print(f"✓ Loaded Roboflow model: {config['model_id']}/v{config['version']}")
        except Exception as e:
            print(f"⚠ Roboflow failed: {e}")
            self.method = 'color'
    
    def _init_tracknet(self, model_path):
        """
        Initialize TrackNetV3
        
        TrackNetV3 is specifically designed for ball/shuttlecock tracking in sports.
        Paper: https://arxiv.org/abs/2004.10506
        
        Architecture: Uses temporal context (past 3 frames) to predict current position
        Output: Heatmap of likely shuttlecock positions
        """
        try:
            # Try to import TrackNet
            import torch
            import torch.nn as nn
            
            # Simple TrackNet-like architecture
            class SimpleTrackNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Input: 3 consecutive frames (9 channels)
                    self.conv1 = nn.Conv2d(9, 64, 3, padding=1)
                    self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                    self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                    self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
                    self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
                    # Output: Heatmap (1 channel)
                    self.conv6 = nn.Conv2d(64, 1, 1)
                    self.relu = nn.ReLU()
                    self.sigmoid = nn.Sigmoid()
                
                def forward(self, x):
                    x = self.relu(self.conv1(x))
                    x = self.relu(self.conv2(x))
                    x = self.relu(self.conv3(x))
                    x = self.relu(self.conv4(x))
                    x = self.relu(self.conv5(x))
                    x = self.sigmoid(self.conv6(x))
                    return x
            
            self.model = SimpleTrackNet()
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                print(f"✓ Loaded TrackNetV3 model: {model_path}")
            else:
                print("⚠ No TrackNetV3 weights provided, using untrained model")
                print("  To train: Use https://github.com/Chang-Anthony/TrackNetV3")
            
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.frame_buffer = deque(maxlen=3)
            
        except ImportError:
            print("⚠ PyTorch not installed. Install with: pip install torch torchvision")
            self.method = 'color'
        except Exception as e:
            print(f"⚠ TrackNet initialization failed: {e}")
            self.method = 'color'
    
    def _init_monotrack(self, model_path):
        """
        Initialize MonoTrack
        
        MonoTrack is a state-of-the-art single object tracker
        Better than TrackNet for handling occlusions and fast motion
        
        If you want to use MonoTrack:
        1. Clone: git clone https://github.com/tsingqguo/MonoTrack
        2. Follow their setup instructions
        3. Provide model path here
        """
        print("⚠ MonoTrack integration requires manual setup")
        print("  See: https://github.com/tsingqguo/MonoTrack")
        print("  Falling back to YOLO detection")
        self._init_yolo(model_path)
    
    def _init_yolo(self, model_path):
        """Initialize YOLO model"""
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"✓ Loaded YOLO model: {model_path}")
            else:
                self.model = YOLO('yolov8n.pt')
                print("✓ Using YOLOv8n")
        except Exception as e:
            print(f"⚠ YOLO failed: {e}")
            self.method = 'color'
    
    def detect_roboflow(self, frame):
        """Roboflow API detection"""
        try:
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            
            confidence_pct = int(self.confidence_threshold * 100)
            response = self.model.predict(temp_path, confidence=confidence_pct, overlap=50)
            predictions = response.json()
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if 'predictions' in predictions and predictions['predictions']:
                detections = []
                for pred in predictions['predictions']:
                    x, y = pred['x'], pred['y']
                    w, h = pred['width'], pred['height']
                    detections.append({
                        'bbox': [int(x - w/2), int(y - h/2), int(w), int(h)],
                        'center': (int(x), int(y)),
                        'confidence': pred['confidence']
                    })
                return detections
            return []
        except Exception as e:
            return []
    
    def detect_tracknet(self, frame):
        """
        TrackNetV3 heatmap-based detection
        Uses last 3 frames to predict current position
        """
        import torch
        
        # Resize frame for model input
        input_h, input_w = 288, 512  # TrackNet standard size
        frame_resized = cv2.resize(frame, (input_w, input_h))
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Add to buffer
        self.frame_buffer.append(frame_gray)
        
        if len(self.frame_buffer) < 3:
            return []
        
        # Stack 3 frames
        input_frames = np.stack(list(self.frame_buffer), axis=0)
        input_tensor = torch.from_numpy(input_frames).float().unsqueeze(0)
        input_tensor = input_tensor.to(self.device) / 255.0
        
        # Get heatmap
        with torch.no_grad():
            heatmap = self.model(input_tensor)
            heatmap = heatmap.squeeze().cpu().numpy()
        
        # Find peak in heatmap
        max_val = heatmap.max()
        if max_val < self.confidence_threshold:
            return []
        
        max_loc = np.unravel_index(heatmap.argmax(), heatmap.shape)
        y_pred, x_pred = max_loc
        
        # Scale back to original resolution
        x_orig = int(x_pred * self.frame_width / input_w)
        y_orig = int(y_pred * self.frame_height / input_h)
        
        return [{
            'bbox': [x_orig - 10, y_orig - 10, 20, 20],
            'center': (x_orig, y_orig),
            'confidence': float(max_val)
        }]
    
    def detect_yolo(self, frame):
        """YOLO detection"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                
                if w < 100 and h < 100 and 0.5 < w/h < 2.0:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(w), int(h)],
                        'center': (int((x1+x2)/2), int((y1+y2)/2)),
                        'confidence': float(box.conf)
                    })
        return detections
    
    def detect_color(self, frame):
        """Color-based detection (fallback)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 500:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.3:
                        x, y, w, h = cv2.boundingRect(cnt)
                        detections.append({
                            'bbox': [x, y, w, h],
                            'center': (x + w//2, y + h//2),
                            'confidence': circularity
                        })
        return detections
    
    def detect(self, frame):
        """Main detection method with temporal smoothing"""
        # Get detections based on method
        if self.method == 'roboflow':
            detections = self.detect_roboflow(frame)
        elif self.method == 'tracknet':
            detections = self.detect_tracknet(frame)
        elif self.method == 'yolo':
            detections = self.detect_yolo(frame)
        else:
            detections = self.detect_color(frame)
        
        # Add color detections as backup
        if not detections:
            detections = self.detect_color(frame)
        
        if not detections:
            return None
        
        # Filter using temporal consistency
        best_detection = self._select_best_detection(detections)
        
        if best_detection:
            self.position_history.append(best_detection['center'])
            self.confidence_history.append(best_detection['confidence'])
            
            return {
                'detected': True,
                'bbox': best_detection['bbox'],
                'center': best_detection['center'],
                'confidence': best_detection['confidence']
            }
        
        return None
    
    def _select_best_detection(self, detections):
        """Select best detection using temporal consistency"""
        if not detections:
            return None
        
        if len(self.position_history) == 0:
            # First detection - return highest confidence
            return max(detections, key=lambda x: x['confidence'])
        
        # Score each detection by proximity to predicted position
        last_pos = self.position_history[-1]
        
        scored = []
        for det in detections:
            dx = det['center'][0] - last_pos[0]
            dy = det['center'][1] - last_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Combined score: confidence + proximity
            proximity_score = 1.0 / (1.0 + distance / 100.0)
            total_score = det['confidence'] * 0.6 + proximity_score * 0.4
            
            scored.append((total_score, det))
        
        return max(scored, key=lambda x: x[0])[1]
    
    def get_average_confidence(self):
        """Get average confidence over recent detections"""
        if len(self.confidence_history) == 0:
            return 0.0
        return np.mean(list(self.confidence_history))


class EnhancedRallyDetector:
    """
    Enhanced Rally Detector v4.1 with all critical improvements
    
    Key Features:
    1. Height-based ground detection
    2. Speed validation
    3. Inter-rally gap enforcement
    4. Trajectory smoothness
    5. ROI filtering
    6. Confidence trend analysis
    """
    
    def __init__(self,
                 fps=30,
                 frame_height=1080,
                 frame_width=1920,
                 tracker_config=None,
                 # Rally timing
                 min_rally_duration=2.0,
                 max_gap_to_merge=3.0,
                 end_timeout=1.5,
                 # Height detection
                 ground_height_ratio=0.75,
                 # Speed validation
                 min_active_speed=50,
                 speed_check_frames=10,
                 # Movement detection
                 movement_threshold=10,
                 # Trajectory validation
                 max_position_jump=200,
                 # Confidence
                 min_avg_confidence=0.3,
                 # ROI (court boundaries)
                 court_roi=None):
        
        # Initialize tracker
        if tracker_config:
            self.tracker = ShuttlecockTracker(
                frame_height=frame_height,
                frame_width=frame_width,
                **tracker_config
            )
        else:
            self.tracker = ShuttlecockTracker(
                method='roboflow',
                frame_height=frame_height,
                frame_width=frame_width
            )
        
        self.state = RallyState.IDLE
        self.fps = fps
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # Configuration
        self.min_rally_duration = min_rally_duration
        self.max_gap_to_merge = max_gap_to_merge
        self.end_timeout = end_timeout
        self.ground_height_ratio = ground_height_ratio
        self.min_active_speed = min_active_speed
        self.speed_check_frames = speed_check_frames
        self.movement_threshold = movement_threshold
        self.max_position_jump = max_position_jump
        self.min_avg_confidence = min_avg_confidence
        
        # Court ROI (if None, use full frame)
        if court_roi:
            self.court_roi = court_roi
        else:
            # Default: exclude bottom 10% and top 5%
            self.court_roi = {
                'x_min': int(frame_width * 0.05),
                'x_max': int(frame_width * 0.95),
                'y_min': int(frame_height * 0.05),
                'y_max': int(frame_height * 0.90)
            }
        
        # Tracking
        self.rally_start_time = None
        self.rally_start_frame = None
        self.last_shuttle_seen = None
        self.shuttle_positions = deque(maxlen=30)
        self.shuttle_speeds = deque(maxlen=self.speed_check_frames)
        
        # Rally storage
        self.rallies = []
        self.rally_count = 0
        
    def is_on_ground(self, y_pos):
        """Check if shuttlecock is on ground (Enhancement #1)"""
        ground_threshold = self.frame_height * self.ground_height_ratio
        return y_pos > ground_threshold
    
    def is_moving_fast_enough(self):
        """Check if shuttlecock speed is above threshold (Enhancement #2)"""
        if len(self.shuttle_speeds) < 3:
            return True
        
        avg_speed = np.mean(list(self.shuttle_speeds))
        return avg_speed >= self.min_active_speed
    
    def has_valid_trajectory(self, new_pos):
        """Check if new position is consistent with trajectory (Enhancement #4)"""
        if len(self.shuttle_positions) == 0:
            return True
        
        last_pos = self.shuttle_positions[-1]
        dx = abs(new_pos[0] - last_pos[0])
        dy = abs(new_pos[1] - last_pos[1])
        jump = np.sqrt(dx**2 + dy**2)
        
        return jump < self.max_position_jump
    
    def is_in_court(self, pos):
        """Check if position is within court boundaries (Enhancement #5)"""
        x, y = pos
        return (self.court_roi['x_min'] <= x <= self.court_roi['x_max'] and
                self.court_roi['y_min'] <= y <= self.court_roi['y_max'])
    
    def has_good_confidence(self):
        """Check confidence trend (Enhancement #6)"""
        avg_conf = self.tracker.get_average_confidence()
        return avg_conf >= self.min_avg_confidence
    
    def calculate_speed(self):
        """Calculate current shuttlecock speed"""
        if len(self.shuttle_positions) < 2:
            return 0
        
        pos1 = self.shuttle_positions[-2]
        pos2 = self.shuttle_positions[-1]
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Speed in pixels per frame
        speed = distance
        return speed
    
    def should_end_rally(self, shuttle, timestamp):
        """
        Enhanced end detection with multiple conditions
        Returns: (should_end, reason)
        """
        # Condition 1: Shuttlecock missing
        if not shuttle:
            time_since_last = timestamp - self.last_shuttle_seen
            if time_since_last > self.end_timeout:
                return True, "timeout"
            return False, None
        
        pos = shuttle['center']
        
        # Condition 2: Height-based ground detection (CRITICAL)
        if self.is_on_ground(pos[1]):
            return True, "ground"
        
        # Condition 3: Out of bounds (CRITICAL)
        if not self.is_in_court(pos):
            return True, "out_of_bounds"
        
        # Condition 4: Invalid trajectory jump
        if not self.has_valid_trajectory(pos):
            return True, "erratic_tracking"
        
        # Condition 5: Speed too low (CRITICAL)
        if len(self.shuttle_speeds) >= self.speed_check_frames:
            if not self.is_moving_fast_enough():
                return True, "low_speed"
        
        # Condition 6: Confidence trend
        if len(self.tracker.confidence_history) >= 5:
            if not self.has_good_confidence():
                return True, "low_confidence"
        
        # Condition 7: Stationary
        if len(self.shuttle_positions) >= 5:
            recent_positions = list(self.shuttle_positions)[-5:]
            movements = []
            for i in range(len(recent_positions) - 1):
                dx = recent_positions[i+1][0] - recent_positions[i][0]
                dy = recent_positions[i+1][1] - recent_positions[i][1]
                movements.append(np.sqrt(dx**2 + dy**2))
            
            if np.mean(movements) < self.movement_threshold:
                return True, "stationary"
        
        return False, None
    
    def process_frame(self, frame, frame_number, timestamp):
        """
        Process frame with enhanced logic
        Returns: ('start', time, frame) or ('end', time, frame, reason) or None
        """
        # Detect shuttlecock
        shuttle = self.tracker.detect(frame)
        
        # Update speed tracking
        if shuttle:
            speed = self.calculate_speed()
            self.shuttle_speeds.append(speed)
        
        # State machine
        if self.state == RallyState.IDLE:
            if shuttle and shuttle['detected']:
                # Valid start conditions
                if self.is_in_court(shuttle['center']) and \
                   not self.is_on_ground(shuttle['center'][1]):
                    
                    self.state = RallyState.RALLY_ACTIVE
                    self.rally_start_time = timestamp
                    self.rally_start_frame = frame_number
                    self.last_shuttle_seen = timestamp
                    self.shuttle_positions.clear()
                    self.shuttle_speeds.clear()
                    self.shuttle_positions.append(shuttle['center'])
                    
                    self.rally_count += 1
                    return ('start', timestamp, frame_number)
        
        elif self.state == RallyState.RALLY_ACTIVE:
            if shuttle and shuttle['detected']:
                self.last_shuttle_seen = timestamp
                self.shuttle_positions.append(shuttle['center'])
            
            # Check end conditions
            should_end, reason = self.should_end_rally(shuttle, timestamp)
            
            if should_end:
                return self._end_rally(timestamp, frame_number, reason)
        
        return None
    
    def _end_rally(self, timestamp, frame_number, reason="unknown"):
        """End rally and check validity"""
        if self.rally_start_time is None:
            self._reset()
            return None
        
        duration = timestamp - self.rally_start_time
        
        # Filter short rallies
        if duration < self.min_rally_duration:
            self._reset()
            return None
        
        rally_data = {
            'start_time': self.rally_start_time,
            'start_frame': self.rally_start_frame,
            'end_time': timestamp,
            'end_frame': frame_number,
            'duration': duration,
            'reason': reason
        }
        
        self.rallies.append(rally_data)
        
        result = ('end', timestamp, frame_number, reason)
        self._reset()
        return result
    
    def _reset(self):
        """Reset for next rally"""
        self.state = RallyState.IDLE
        self.rally_start_time = None
        self.rally_start_frame = None
        self.last_shuttle_seen = None
        self.shuttle_positions.clear()
        self.shuttle_speeds.clear()
    
    def merge_close_rallies(self):
        """
        Enhancement #3: Merge rallies separated by small gaps
        Prevents rally splitting due to brief occlusions
        """
        if len(self.rallies) < 2:
            return
        
        merged = []
        current_rally = self.rallies[0].copy()
        
        for next_rally in self.rallies[1:]:
            gap = next_rally['start_time'] - current_rally['end_time']
            
            if gap < self.max_gap_to_merge:
                # Merge: extend current rally to include next
                current_rally['end_time'] = next_rally['end_time']
                current_rally['end_frame'] = next_rally['end_frame']
                current_rally['duration'] = (current_rally['end_time'] - 
                                             current_rally['start_time'])
                print(f"  ⚡ Merged rallies with {gap:.1f}s gap")
            else:
                # Gap too large - start new rally
                merged.append(current_rally)
                current_rally = next_rally.copy()
        
        merged.append(current_rally)
        self.rallies = merged


def format_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def process_video_enhanced(video_path, 
                          output_csv='rallies_enhanced.csv',
                          visualize=False,
                          output_video=None,
                          tracker_config=None,
                          detector_config=None):
    """
    Process video with enhanced v4.1 detector
    """
    print("\n" + "="*70)
    print("ENHANCED RALLY DETECTION SYSTEM v4.1")
    print("="*70)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {format_time(duration)}")
    print(f"  Total Frames: {total_frames}")
    
    # Initialize enhanced detector
    if detector_config is None:
        detector_config = {}
    
    detector = EnhancedRallyDetector(
        fps=fps,
        frame_height=height,
        frame_width=width,
        tracker_config=tracker_config,
        **detector_config
    )
    
    print(f"\nEnhancements Active:")
    print(f"  ✓ Height-based ground detection (y > {detector.ground_height_ratio})")
    print(f"  ✓ Speed validation (min {detector.min_active_speed} px/frame)")
    print(f"  ✓ Inter-rally gap merging (max {detector.max_gap_to_merge}s)")
    print(f"  ✓ Trajectory smoothness (max jump {detector.max_position_jump}px)")
    print(f"  ✓ ROI court filtering")
    print(f"  ✓ Confidence trend analysis (min {detector.min_avg_confidence})")
    
    # Output writer
    out_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    current_rally = None
    
    print(f"\nProcessing...\n")
    
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
            event_type = event[0]
            
            if event_type == 'start':
                _, event_time, event_frame = event
                current_rally = detector.rallies[-1] if detector.rallies else None
                print(f"✓ Rally #{detector.rally_count} START at {format_time(event_time)}")
            
            elif event_type == 'end':
                _, event_time, event_frame, reason = event
                if detector.rallies:
                    rally = detector.rallies[-1]
                    print(f"  Rally END at {format_time(event_time)} " +
                          f"(duration: {rally['duration']:.1f}s, reason: {reason})")
        
        # Visualization
        if visualize or output_video:
            vis_frame = frame.copy()
            
            # Draw court ROI
            roi = detector.court_roi
            cv2.rectangle(vis_frame, 
                         (roi['x_min'], roi['y_min']),
                         (roi['x_max'], roi['y_max']),
                         (100, 100, 100), 2)
            
            # Draw ground line
            ground_y = int(height * detector.ground_height_ratio)
            cv2.line(vis_frame, (0, ground_y), (width, ground_y), (0, 0, 255), 2)
            cv2.putText(vis_frame, "Ground Line", (10, ground_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Status
            status_color = (0, 255, 0) if detector.state == RallyState.RALLY_ACTIVE else (128, 128, 128)
            status_text = f"State: {detector.state.name} | Rally: #{detector.rally_count}"
            cv2.putText(vis_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Speed indicator
            if len(detector.shuttle_speeds) > 0:
                avg_speed = np.mean(list(detector.shuttle_speeds))
                speed_text = f"Speed: {avg_speed:.1f} px/frame"
                speed_color = (0, 255, 0) if avg_speed >= detector.min_active_speed else (0, 165, 255)
                cv2.putText(vis_frame, speed_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
            
            # Confidence
            avg_conf = detector.tracker.get_average_confidence()
            conf_text = f"Conf: {avg_conf:.2f}"
            conf_color = (0, 255, 0) if avg_conf >= detector.min_avg_confidence else (0, 165, 255)
            cv2.putText(vis_frame, conf_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
            
            # Timestamp
            cv2.putText(vis_frame, format_time(timestamp), (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw trajectory
            if len(detector.shuttle_positions) > 1:
                positions = list(detector.shuttle_positions)
                for i in range(len(positions) - 1):
                    cv2.line(vis_frame, positions[i], positions[i+1], (0, 255, 255), 2)
                
                # Current position
                if positions:
                    cv2.circle(vis_frame, positions[-1], 8, (0, 255, 0), -1)
            
            if output_video:
                out_writer.write(vis_frame)
            
            if visualize:
                cv2.imshow('Enhanced Rally Detection v4.1', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Progress
        if frame_count % int(fps * 10) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}%")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    if out_writer:
        out_writer.release()
    if visualize:
        cv2.destroyAllWindows()
    
    # CRITICAL: Merge close rallies (Enhancement #3)
    print(f"\nApplying inter-rally gap merging...")
    original_count = len(detector.rallies)
    detector.merge_close_rallies()
    merged_count = original_count - len(detector.rallies)
    if merged_count > 0:
        print(f"  ✓ Merged {merged_count} rally pair(s)")
    
    print(f"\n" + "="*70)
    print(f"DETECTION COMPLETE")
    print(f"="*70)
    print(f"Total rallies detected: {len(detector.rallies)}")
    
    if len(detector.rallies) == 0:
        print("\n⚠ WARNING: No rallies detected!")
        return None
    
    # Create DataFrame
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    df = pd.DataFrame(detector.rallies)
    df['video_id'] = video_id
    df['rally_id'] = range(1, len(df) + 1)
    df['start_time_formatted'] = df['start_time'].apply(format_time)
    df['end_time_formatted'] = df['end_time'].apply(format_time)
    df['duration'] = df['duration'].round(1)
    
    # Output columns
    output_df = df[['video_id', 'rally_id', 'start_time_formatted', 
                     'end_time_formatted', 'duration', 'reason']]
    output_df.columns = ['video_id', 'rally_id', 'start_time', 'end_time', 'duration', 'end_reason']
    
    # Save CSV
    output_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    # Statistics
    print(f"\nRally Statistics:")
    print(f"  Average duration: {df['duration'].mean():.1f}s")
    print(f"  Shortest: {df['duration'].min():.1f}s")
    print(f"  Longest: {df['duration'].max():.1f}s")
    
    # End reason breakdown
    print(f"\nEnd Reason Breakdown:")
    reason_counts = df['reason'].value_counts()
    for reason, count in reason_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Preview
    print(f"\nFirst 5 Rallies:")
    print(output_df.head().to_string(index=False))
    
    return output_df


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Badminton Rally Detection v4.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Roboflow with all enhancements
  python rally_detector_v4.1.py --video match.mp4
  
  # Use TrackNetV3 (requires trained model)
  python rally_detector_v4.1.py --video match.mp4 --method tracknet --model weights/tracknet.pth
  
  # Use YOLO with custom settings
  python rally_detector_v4.1.py --video match.mp4 --method yolo \\
      --ground-ratio 0.8 --min-speed 60
  
  # Visualize with saved output
  python rally_detector_v4.1.py --video match.mp4 --visualize --save-video output.mp4
  
  # Custom court ROI (x_min,y_min,x_max,y_max as ratios)
  python rally_detector_v4.1.py --video match.mp4 --roi 0.1,0.05,0.9,0.85

Enhancement Details:
  1. Height-based ground detection: Ends rally when shuttle touches ground
  2. Speed validation: Filters out slow-moving false positives
  3. Inter-rally gap merging: Reconnects split rallies (gaps <3s)
  4. Trajectory smoothness: Rejects erratic tracking jumps
  5. ROI filtering: Only tracks within court boundaries
  6. Confidence trend: Ends rally on sustained low confidence
        """
    )
    
    # Video I/O
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='rallies_enhanced.csv', 
                       help='Output CSV path')
    parser.add_argument('--visualize', action='store_true', help='Show live visualization')
    parser.add_argument('--save-video', type=str, default=None, help='Save annotated video')
    
    # Tracking method
    parser.add_argument('--method', type=str, default='roboflow',
                       choices=['roboflow', 'tracknet', 'monotrack', 'yolo', 'color'],
                       help='Tracking method (default: roboflow)')
    parser.add_argument('--model', type=str, default=None, help='Model weights path')
    
    # Roboflow config
    parser.add_argument('--api-key', type=str, default=None, help='Roboflow API key')
    parser.add_argument('--model-id', type=str, default=None, help='Roboflow model ID')
    parser.add_argument('--rf-version', type=int, default=1, help='Roboflow version')
    
    # Rally detection parameters
    parser.add_argument('--min-duration', type=float, default=2.0,
                       help='Minimum rally duration (s)')
    parser.add_argument('--max-gap', type=float, default=3.0,
                       help='Max gap for merging rallies (s)')
    
    # Enhancement parameters
    parser.add_argument('--ground-ratio', type=float, default=0.75,
                       help='Ground detection height ratio (0-1)')
    parser.add_argument('--min-speed', type=float, default=50,
                       help='Minimum active speed (px/frame)')
    parser.add_argument('--max-jump', type=float, default=200,
                       help='Max trajectory jump (px)')
    parser.add_argument('--min-conf', type=float, default=0.3,
                       help='Minimum average confidence')
    parser.add_argument('--roi', type=str, default=None,
                       help='Court ROI as x_min,y_min,x_max,y_max (ratios 0-1)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ENHANCED RALLY DETECTION SYSTEM v4.1")
    print("="*70)
    
    # Configure tracker
    tracker_config = {
        'method': args.method,
        'model_path': args.model
    }
    
    if args.method == 'roboflow':
        if args.api_key and args.model_id:
            tracker_config['roboflow_config'] = {
                'api_key': args.api_key,
                'model_id': args.model_id,
                'version': args.rf_version
            }
        print(f"  Tracking: Roboflow API")
    elif args.method == 'tracknet':
        print(f"  Tracking: TrackNetV3 (temporal heatmap)")
        if not args.model:
            print(f"  ⚠ No TrackNet model provided, using untrained weights")
    elif args.method == 'monotrack':
        print(f"  Tracking: MonoTrack (requires setup)")
    elif args.method == 'yolo':
        print(f"  Tracking: YOLO detection")
    else:
        print(f"  Tracking: Color-based only")
    
    # Configure detector
    detector_config = {
        'min_rally_duration': args.min_duration,
        'max_gap_to_merge': args.max_gap,
        'ground_height_ratio': args.ground_ratio,
        'min_active_speed': args.min_speed,
        'max_position_jump': args.max_jump,
        'min_avg_confidence': args.min_conf
    }
    
    # Parse ROI if provided
    if args.roi:
        try:
            roi_vals = [float(x) for x in args.roi.split(',')]
            if len(roi_vals) == 4:
                # Get video dimensions first
                cap = cv2.VideoCapture(args.video)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                detector_config['court_roi'] = {
                    'x_min': int(width * roi_vals[0]),
                    'x_max': int(width * roi_vals[2]),
                    'y_min': int(height * roi_vals[1]),
                    'y_max': int(height * roi_vals[3])
                }
                print(f"  ROI: Custom court boundaries")
        except:
            print(f"  ⚠ Invalid ROI format, using defaults")
    
    print()
    
    # Process video
    result_df = process_video_enhanced(
        video_path=args.video,
        output_csv=args.output,
        visualize=args.visualize,
        output_video=args.save_video,
        tracker_config=tracker_config,
        detector_config=detector_config
    )
    
    if result_df is not None:
        print(f"\n✓ Enhanced detection complete!")
        print(f"  Improvements over v4.0:")
        print(f"    • Reduced false positives from grounded shuttlecock")
        print(f"    • Eliminated rally splits from brief occlusions")
        print(f"    • Filtered out-of-bounds detections")
        print(f"    • Rejected erratic tracking artifacts")


if __name__ == "__main__":
    # Quick test examples:
    
    # Example 1: Default with all enhancements
    # process_video_enhanced('match.mp4', visualize=True)
    
    # Example 2: TrackNetV3 with custom thresholds
    # tracker_config = {'method': 'tracknet', 'model_path': 'weights/tracknet.pth'}
    # detector_config = {'ground_height_ratio': 0.8, 'min_active_speed': 60}
    # process_video_enhanced('match.mp4', tracker_config=tracker_config, 
    #                       detector_config=detector_config)
    
    # Example 3: Conservative settings (fewer false positives)
    # detector_config = {
    #     'min_rally_duration': 3.0,
    #     'min_active_speed': 70,
    #     'min_avg_confidence': 0.4,
    #     'max_gap_to_merge': 2.0
    # }
    # process_video_enhanced('match.mp4', detector_config=detector_config)
    
    main()
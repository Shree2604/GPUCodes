import pandas as pd
import numpy as np

FPS = 30

# Ground truth for validation (based on manual review)
GROUND_TRUTH = [
    (0, 480),      # 00:00 - 00:16
    (570, 840),    # 00:19 - 00:28
    (930, 1290),   # 00:31 - 00:43
    (1380, 1620),  # 00:46 - 00:54
    (1680, 2010),  # 00:56 - 01:07
    (2070, 2156)   # 01:09 - 01:14
]

# Load CSV
df = pd.read_csv("Code_ball.csv")

print("\n" + "="*70)
print("ANALYZING BALL TRACKING DATA")
print("="*70)
print(f"Total frames: {len(df)}")
print(f"Visible frames: {df['Visibility'].sum()}")
print(f"Missing frames: {len(df) - df['Visibility'].sum()}")

# ============================================
# RALLY DETECTION BASED ON VISIBILITY GAPS
# ============================================

# Find all frames where ball is visible
valid_mask = (df['Visibility'] == 1) & ~((df['X'] == 0) & (df['Y'] == 0))
valid_frames = df[valid_mask]['Frame'].values

print(f"\nValid tracked frames: {len(valid_frames)}")

# Find gaps between consecutive valid frames
gaps = []
for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    if gap > 15:  # Significant gap (> 0.5 seconds)
        gaps.append({
            'before': valid_frames[i-1],
            'after': valid_frames[i],
            'gap_size': gap,
            'gap_duration': gap / FPS
        })

print(f"\n{'Detected Gaps (>15 frames):':^70}")
print("-"*70)
for i, g in enumerate(gaps, 1):
    print(f"Gap {i}: Frame {g['before']:4d} → {g['after']:4d} | "
          f"Size: {g['gap_size']:3d} frames ({g['gap_duration']:4.2f}s)")

# ============================================
# STRATEGY: GAP-BASED SEGMENTATION
# ============================================
# Use gaps >= 15 frames as rally boundaries
# This naturally segments the video into rallies

MIN_GAP_THRESHOLD = 15  # 0.5 seconds - minimum gap to split rallies
MIN_RALLY_LENGTH = 60   # 2.0 seconds - minimum rally duration

rally_segments = []
rally_start = valid_frames[0]

for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    
    if gap >= MIN_GAP_THRESHOLD:
        # End current rally
        rally_end = valid_frames[i-1]
        rally_duration = rally_end - rally_start
        
        # Only keep rallies longer than minimum duration
        if rally_duration >= MIN_RALLY_LENGTH:
            rally_segments.append((rally_start, rally_end))
        
        # Start new rally
        rally_start = valid_frames[i]

# Add final rally
rally_end = valid_frames[-1]
if rally_end - rally_start >= MIN_RALLY_LENGTH:
    rally_segments.append((rally_start, rally_end))

print(f"\nInitial segments detected: {len(rally_segments)}")

# ============================================
# POST-PROCESSING: MERGE CLOSE SEGMENTS
# ============================================
# Merge segments that are separated by very short breaks
# (to handle tracking errors/brief occlusions)

MERGE_THRESHOLD = 50  # 1.67 seconds - max gap to merge segments

if len(rally_segments) > 0:
    merged_rallies = []
    current_start, current_end = rally_segments[0]
    
    for i in range(1, len(rally_segments)):
        next_start, next_end = rally_segments[i]
        gap_between = next_start - current_end
        
        if gap_between <= MERGE_THRESHOLD:
            # Merge: extend current rally to include next segment
            current_end = next_end
        else:
            # Save current rally and start new one
            merged_rallies.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # Add final rally
    merged_rallies.append((current_start, current_end))
else:
    merged_rallies = []

# ============================================
# FINAL FILTERING
# ============================================
# Remove any very short rallies that slipped through

FINAL_MIN_DURATION = 90  # 3.0 seconds minimum

final_rallies = []
for start, end in merged_rallies:
    duration_frames = end - start
    duration_sec = duration_frames / FPS
    
    if duration_frames >= FINAL_MIN_DURATION:
        final_rallies.append((start, end))

# ============================================
# ACCURACY CALCULATION
# ============================================

def iou(detected, ground_truth):
    """Calculate Intersection over Union (IoU)"""
    inter_start = max(detected[0], ground_truth[0])
    inter_end = min(detected[1], ground_truth[1])
    intersection = max(0, inter_end - inter_start)
    
    union_start = min(detected[0], ground_truth[0])
    union_end = max(detected[1], ground_truth[1])
    union = union_end - union_start
    
    return (intersection / union * 100) if union > 0 else 0

def frame_to_time(frame):
    """Convert frame number to MM:SS format"""
    seconds = frame / FPS
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# Calculate IoU for each rally pair
iou_scores = []
rally_matches = []

# Match detected rallies to ground truth
for i, gt_rally in enumerate(GROUND_TRUTH):
    best_iou = 0
    best_match = None
    
    for j, det_rally in enumerate(final_rallies):
        iou_score = iou(det_rally, gt_rally)
        if iou_score > best_iou:
            best_iou = iou_score
            best_match = j
    
    iou_scores.append(best_iou)
    rally_matches.append(best_match)

overall_accuracy = np.mean(iou_scores) if iou_scores else 0

# ============================================
# PRINT RESULTS
# ============================================

print("\n" + "="*70)
print("DETECTED RALLIES")
print("="*70)

for i, (start, end) in enumerate(final_rallies, 1):
    duration = (end - start) / FPS
    print(f"Rally {i}: Frames {start:4d} → {end:4d} | "
          f"{frame_to_time(start)} → {frame_to_time(end)} | "
          f"Duration: {duration:5.2f} sec")

print("\n" + "="*70)
print("GROUND TRUTH RALLIES")
print("="*70)

for i, (start, end) in enumerate(GROUND_TRUTH, 1):
    duration = (end - start) / FPS
    print(f"Rally {i}: Frames {start:4d} → {end:4d} | "
          f"{frame_to_time(start)} → {frame_to_time(end)} | "
          f"Duration: {duration:5.2f} sec")

print("\n" + "="*70)
print("ACCURACY METRICS (IoU per rally)")
print("="*70)

for i, (iou_score, match_idx) in enumerate(zip(iou_scores, rally_matches), 1):
    if match_idx is not None:
        det_rally = final_rallies[match_idx]
        gt_rally = GROUND_TRUTH[i-1]
        frame_diff = abs(det_rally[0] - gt_rally[0])
        print(f"Rally {i}: IoU = {iou_score:6.2f}% | "
              f"Matched with detected rally {match_idx+1} | "
              f"Start offset: {frame_diff:3d} frames")
    else:
        print(f"Rally {i}: IoU = {iou_score:6.2f}% | No match found")

print(f"\n{'='*70}")
print(f"{'OVERALL ACCURACY:':<40} {overall_accuracy:6.2f}%")
print(f"{'Detected Rallies:':<40} {len(final_rallies)}")
print(f"{'Ground Truth Rallies:':<40} {len(GROUND_TRUTH)}")
print(f"{'Correctly Matched (IoU > 50%):':<40} {sum(1 for iou in iou_scores if iou > 50)}")
print("="*70)

# ============================================
# PARAMETER TUNING SUGGESTIONS
# ============================================

if overall_accuracy < 80:
    print("\n" + "="*70)
    print("TUNING SUGGESTIONS TO IMPROVE ACCURACY")
    print("="*70)
    
    avg_gap = np.mean([g['gap_size'] for g in gaps]) if gaps else 0
    
    if len(final_rallies) > len(GROUND_TRUTH):
        print(f"⚠ Too many rallies detected ({len(final_rallies)} vs {len(GROUND_TRUTH)} expected)")
        print(f"  → Try INCREASING MIN_GAP_THRESHOLD from {MIN_GAP_THRESHOLD} to {int(avg_gap * 0.8)}")
        print(f"  → Or INCREASE MERGE_THRESHOLD from {MERGE_THRESHOLD} to 75-100")
    
    elif len(final_rallies) < len(GROUND_TRUTH):
        print(f"⚠ Too few rallies detected ({len(final_rallies)} vs {len(GROUND_TRUTH)} expected)")
        print(f"  → Try DECREASING MIN_GAP_THRESHOLD from {MIN_GAP_THRESHOLD} to 10-12")
        print(f"  → Or DECREASE MERGE_THRESHOLD from {MERGE_THRESHOLD} to 30-40")
    
    low_iou_rallies = [(i+1, iou) for i, iou in enumerate(iou_scores) if iou < 70]
    if low_iou_rallies:
        print(f"\n⚠ Rallies with low IoU (<70%): {', '.join([f'Rally {r}' for r, _ in low_iou_rallies])}")
        print(f"  → Check if rally boundaries need fine-tuning")
    
    print("="*70)
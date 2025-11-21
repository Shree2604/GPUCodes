import pandas as pd
import numpy as np

FPS = 30

# Ground truth rallies (manually verified)
GROUND_TRUTH = [
    (0, 480),      # Rally 1: 00:00 - 00:16 (16.0s)
    (570, 840),    # Rally 2: 00:19 - 00:28 (9.0s)
    (930, 1290),   # Rally 3: 00:31 - 00:43 (12.0s)
    (1380, 1620),  # Rally 4: 00:46 - 00:54 (8.0s)
    (1680, 2010),  # Rally 5: 00:56 - 01:07 (11.0s)
    (2070, 2156)   # Rally 6: 01:09 - 01:11 (2.87s)
]

# Load ball tracking data
df = pd.read_csv("./Code_ball.csv")

print("="*80)
print("OPTIMAL RALLY DETECTION ALGORITHM (Refined)")
print("="*80)
print(f"Total frames: {len(df)}")
print(f"Video duration: {len(df)/FPS:.2f} seconds ({int(len(df)/FPS//60)}:{int(len(df)/FPS%60):02d})")

# ============================================
# STEP 1: Identify valid ball detections
# ============================================
df['valid'] = (df['Visibility'] == 1) & ~((df['X'] == 0) & (df['Y'] == 0))
valid_frames = df[df['valid']]['Frame'].values

print(f"\nValid tracked frames: {len(valid_frames)} ({len(valid_frames)/len(df)*100:.1f}%)")
print(f"Missing frames: {len(df) - len(valid_frames)}")

# ============================================
# STEP 2: Analyze all gaps to find patterns
# ============================================
all_gaps = []
for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    if gap > 1:  # Any gap
        all_gaps.append({
            'before': valid_frames[i-1],
            'after': valid_frames[i],
            'size': gap,
            'time': gap / FPS
        })

print(f"\nTotal gaps found: {len(all_gaps)}")
if all_gaps:
    gap_sizes = [g['size'] for g in all_gaps]
    print(f"Gap size range: {min(gap_sizes)} to {max(gap_sizes)} frames")
    print(f"Average gap: {np.mean(gap_sizes):.1f} frames")
    print(f"Median gap: {np.median(gap_sizes):.1f} frames")

# Show significant gaps
print(f"\nSignificant gaps (>10 frames):")
print(f"{'Before':<10} {'After':<10} {'Gap Size':<12} {'Duration':<12}")
print("-"*50)
for g in all_gaps:
    if g['size'] > 10:
        print(f"{g['before']:<10} {g['after']:<10} {g['size']:<12} {g['time']:.2f}s")

# ============================================
# STEP 3: Multi-level segmentation approach
# ============================================
# Instead of one threshold, use adaptive approach:
# 1. Start with small gaps to create fine segments
# 2. Merge segments intelligently based on duration and spacing

INITIAL_GAP = 8       # Small gap to catch all potential breaks (relaxed from 12)
MIN_SEGMENT_FRAMES = 30  # Very short segments likely noise (1 second)

initial_segments = []
segment_start = valid_frames[0]

for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    
    if gap > INITIAL_GAP:
        segment_end = valid_frames[i-1]
        segment_duration = segment_end - segment_start
        
        # Keep even very short segments initially
        if segment_duration >= MIN_SEGMENT_FRAMES:
            initial_segments.append((segment_start, segment_end))
        
        segment_start = valid_frames[i]

# Add final segment
segment_end = valid_frames[-1]
if segment_end - segment_start >= MIN_SEGMENT_FRAMES:
    initial_segments.append((segment_start, segment_end))

print(f"\n{len(initial_segments)} initial segments detected (gap threshold: {INITIAL_GAP} frames)")

# ============================================
# STEP 4: Smart merging with adaptive threshold
# ============================================
# Key insight: Look at gaps BETWEEN segments, not just a fixed threshold
# Merge segments that are very close (likely same rally with brief occlusion)

if len(initial_segments) == 0:
    merged_rallies = []
else:
    # Calculate gaps between segments
    segment_gaps = []
    for i in range(len(initial_segments) - 1):
        gap = initial_segments[i+1][0] - initial_segments[i][1]
        segment_gaps.append(gap)
    
    if segment_gaps:
        print(f"\nGaps between segments: {segment_gaps}")
        print(f"Gap statistics - Min: {min(segment_gaps)}, Max: {max(segment_gaps)}, Mean: {np.mean(segment_gaps):.1f}")
    
    # Adaptive merging: merge gaps smaller than threshold
    # Use a more conservative threshold to avoid over-merging
    MERGE_GAP_THRESHOLD = 35  # Frames (about 1.17 seconds) - more conservative
    
    merged_rallies = []
    current_start, current_end = initial_segments[0]
    
    for i in range(1, len(initial_segments)):
        next_start, next_end = initial_segments[i]
        gap_between = next_start - current_end
        
        # Only merge if gap is small AND resulting rally would still be reasonable
        if gap_between <= MERGE_GAP_THRESHOLD:
            # Merge: extend current rally
            current_end = next_end
        else:
            # Save current rally and start new one
            merged_rallies.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # Add final rally
    merged_rallies.append((current_start, current_end))

print(f"After merging (threshold: {MERGE_GAP_THRESHOLD} frames): {len(merged_rallies)} rallies")

# ============================================
# STEP 5: Final filtering - keep meaningful rallies
# ============================================
# Keep rallies that are at least 2 seconds long
MIN_RALLY_DURATION = 60  # frames (2 seconds) - more lenient

final_rallies = []
for s, e in merged_rallies:
    duration = e - s
    if duration >= MIN_RALLY_DURATION:
        final_rallies.append((s, e))
    else:
        print(f"  Filtered out short segment: frames {s} → {e} ({duration/FPS:.2f}s)")

print(f"After filtering (min {MIN_RALLY_DURATION/FPS:.1f}s): {len(final_rallies)} rallies")

# ============================================
# ACCURACY CALCULATION (IoU)
# ============================================

def calculate_iou(detected, ground_truth):
    """Calculate Intersection over Union (IoU) metric"""
    inter_start = max(detected[0], ground_truth[0])
    inter_end = min(detected[1], ground_truth[1])
    intersection = max(0, inter_end - inter_start)
    
    union_start = min(detected[0], ground_truth[0])
    union_end = max(detected[1], ground_truth[1])
    union = union_end - union_start
    
    return (intersection / union * 100) if union > 0 else 0

def frame_to_timestamp(frame):
    """Convert frame to MM:SS format"""
    seconds = frame / FPS
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

# Match detected rallies to ground truth
iou_scores = []
rally_matches = []

for i, gt_rally in enumerate(GROUND_TRUTH):
    best_iou = 0
    best_match_idx = None
    
    for j, det_rally in enumerate(final_rallies):
        iou_score = calculate_iou(det_rally, gt_rally)
        if iou_score > best_iou:
            best_iou = iou_score
            best_match_idx = j
    
    iou_scores.append(best_iou)
    rally_matches.append(best_match_idx)

overall_accuracy = np.mean(iou_scores) if iou_scores else 0

# ============================================
# RESULTS OUTPUT
# ============================================

print("\n" + "="*80)
print("DETECTED RALLIES")
print("="*80)
print(f"{'#':<5} {'Frames':<20} {'Time Range':<20} {'Duration':<15}")
print("-"*80)

for i, (start, end) in enumerate(final_rallies, 1):
    duration = (end - start) / FPS
    print(f"{i:<5} {start:4d} → {end:4d}{'':<8} "
          f"{frame_to_timestamp(start)} → {frame_to_timestamp(end)}{'':<5} "
          f"{duration:5.2f} sec")

print("\n" + "="*80)
print("GROUND TRUTH RALLIES")
print("="*80)
print(f"{'#':<5} {'Frames':<20} {'Time Range':<20} {'Duration':<15}")
print("-"*80)

for i, (start, end) in enumerate(GROUND_TRUTH, 1):
    duration = (end - start) / FPS
    print(f"{i:<5} {start:4d} → {end:4d}{'':<8} "
          f"{frame_to_timestamp(start)} → {frame_to_timestamp(end)}{'':<5} "
          f"{duration:5.2f} sec")

print("\n" + "="*80)
print("IoU ACCURACY ANALYSIS")
print("="*80)
print(f"{'GT Rally':<12} {'Detected':<12} {'IoU':<10} {'Start Δ':<12} {'End Δ':<12}")
print("-"*80)

for i, (iou, match_idx) in enumerate(zip(iou_scores, rally_matches), 1):
    if match_idx is not None:
        det_rally = final_rallies[match_idx]
        gt_rally = GROUND_TRUTH[i-1]
        start_diff = det_rally[0] - gt_rally[0]
        end_diff = det_rally[1] - gt_rally[1]
        
        status = "✓" if iou >= 70 else "✗"
        print(f"Rally {i:<6} Rally {match_idx+1:<6} {iou:6.2f}% {status}  "
              f"{start_diff:+5d} frames   {end_diff:+5d} frames")
    else:
        print(f"Rally {i:<6} {'NONE':<6} {iou:6.2f}% ✗  {'N/A':<12} {'N/A':<12}")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Overall IoU Accuracy:          {overall_accuracy:6.2f}%")
print(f"Detected Rallies:              {len(final_rallies)}")
print(f"Ground Truth Rallies:          {len(GROUND_TRUTH)}")
print(f"Perfect Detection:             {'✓ Yes' if len(final_rallies) == len(GROUND_TRUTH) else '✗ No'}")
print(f"High Quality Matches (>70%):   {sum(1 for iou in iou_scores if iou >= 70)}")
print(f"Acceptable Matches (>50%):     {sum(1 for iou in iou_scores if iou >= 50)}")
print(f"Poor Matches (<50%):           {sum(1 for iou in iou_scores if iou < 50)}")

# Calculate precision and recall
true_positives = sum(1 for iou in iou_scores if iou >= 50)
precision = true_positives / len(final_rallies) if final_rallies else 0
recall = true_positives / len(GROUND_TRUTH) if GROUND_TRUTH else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPrecision (TP/Detected):       {precision*100:6.2f}%")
print(f"Recall (TP/Ground Truth):      {recall*100:6.2f}%")
print(f"F1 Score:                      {f1_score*100:6.2f}%")

print("="*80)

# ============================================
# ANALYSIS & RECOMMENDATIONS
# ============================================

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

rally_diff = len(final_rallies) - len(GROUND_TRUTH)

if rally_diff == 0:
    print("✓ Perfect count! Detected exactly the right number of rallies.")
    if overall_accuracy < 70:
        print("  However, boundary alignment needs improvement.")
        print(f"  Current parameters: INITIAL_GAP={INITIAL_GAP}, MERGE_GAP={MERGE_GAP_THRESHOLD}")
elif rally_diff > 0:
    print(f"⚠ Over-segmentation: {rally_diff} extra rally(ies) detected")
    print(f"  → Try INCREASING MERGE_GAP_THRESHOLD from {MERGE_GAP_THRESHOLD} to {MERGE_GAP_THRESHOLD + 15}")
    print(f"  → Or INCREASING INITIAL_GAP from {INITIAL_GAP} to {INITIAL_GAP + 2}")
else:
    print(f"⚠ Under-segmentation: Missing {abs(rally_diff)} rally(ies)")
    print(f"  → Try DECREASING MERGE_GAP_THRESHOLD from {MERGE_GAP_THRESHOLD} to {max(20, MERGE_GAP_THRESHOLD - 10)}")
    print(f"  → Or DECREASING INITIAL_GAP from {INITIAL_GAP} to {max(5, INITIAL_GAP - 2)}")

if overall_accuracy >= 70:
    print(f"\n✓ Good accuracy achieved: {overall_accuracy:.2f}%")
elif overall_accuracy >= 50:
    print(f"\n⚠ Moderate accuracy: {overall_accuracy:.2f}%")
else:
    print(f"\n✗ Low accuracy: {overall_accuracy:.2f}%")

print("="*80)
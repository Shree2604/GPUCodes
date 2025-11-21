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
print("OPTIMAL RALLY DETECTION ALGORITHM (Maximum Sensitivity)")
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
# STEP 2: Comprehensive gap analysis
# ============================================
all_gaps = []
for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    if gap > 1:  # Any gap
        all_gaps.append({
            'index': i,
            'before': valid_frames[i-1],
            'after': valid_frames[i],
            'size': gap,
            'time': gap / FPS
        })

print(f"\nTotal gaps found: {len(all_gaps)}")
if all_gaps:
    gap_sizes = [g['size'] for g in all_gaps]
    print(f"Gap size range: {min(gap_sizes)} to {max(gap_sizes)} frames")
    print(f"Average gap: {np.mean(gap_sizes):.1f} frames ({np.mean(gap_sizes)/FPS:.2f}s)")
    print(f"Median gap: {np.median(gap_sizes):.1f} frames ({np.median(gap_sizes)/FPS:.2f}s)")

# Show ALL significant gaps
print(f"\nALL gaps (>5 frames):")
print(f"{'#':<4} {'Before':<10} {'After':<10} {'Gap Size':<12} {'Duration':<12} {'Gap Time'}")
print("-"*70)
for idx, g in enumerate(all_gaps, 1):
    if g['size'] > 5:
        start_time = g['before'] / FPS
        print(f"{idx:<4} {g['before']:<10} {g['after']:<10} {g['size']:<12} "
              f"{g['time']:.2f}s{'':<8} {int(start_time//60):02d}:{int(start_time%60):02d}")

# ============================================
# STEP 3: Very aggressive initial segmentation
# ============================================
# Use a very small gap threshold to catch all possible breaks
INITIAL_GAP = 5       # Very small - catch almost all gaps (was 8)
MIN_SEGMENT_FRAMES = 20  # Keep very short segments (was 30)

initial_segments = []
segment_start = valid_frames[0]

for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    
    if gap > INITIAL_GAP:
        segment_end = valid_frames[i-1]
        segment_duration = segment_end - segment_start
        
        if segment_duration >= MIN_SEGMENT_FRAMES:
            initial_segments.append((segment_start, segment_end))
        
        segment_start = valid_frames[i]

# Add final segment
segment_end = valid_frames[-1]
if segment_end - segment_start >= MIN_SEGMENT_FRAMES:
    initial_segments.append((segment_start, segment_end))

print(f"\n{len(initial_segments)} initial segments detected (gap threshold: {INITIAL_GAP} frames)")

# Show all initial segments
print(f"\nInitial segments:")
for idx, (s, e) in enumerate(initial_segments, 1):
    print(f"  Segment {idx}: frames {s:4d} → {e:4d} ({(e-s)/FPS:5.2f}s)")

# ============================================
# STEP 4: Very conservative merging
# ============================================
# Only merge segments that are VERY close together (tracking errors)
# DO NOT merge segments with significant gaps between them

if len(initial_segments) == 0:
    merged_rallies = []
else:
    # Calculate gaps between segments
    segment_gaps = []
    for i in range(len(initial_segments) - 1):
        gap = initial_segments[i+1][0] - initial_segments[i][1]
        segment_gaps.append({
            'after_segment': i+1,
            'gap': gap,
            'time': gap / FPS
        })
    
    if segment_gaps:
        print(f"\nGaps between segments:")
        for i, sg in enumerate(segment_gaps):
            print(f"  After segment {sg['after_segment']}: {sg['gap']} frames ({sg['time']:.2f}s)")
    
    # Very conservative merge threshold - only merge tracking errors
    MERGE_GAP_THRESHOLD = 15  # Only ~0.5 seconds (was 35)
    
    merged_rallies = []
    current_start, current_end = initial_segments[0]
    
    for i in range(1, len(initial_segments)):
        next_start, next_end = initial_segments[i]
        gap_between = next_start - current_end
        
        # Only merge very small gaps (brief tracking loss)
        if gap_between <= MERGE_GAP_THRESHOLD:
            current_end = next_end
            print(f"  → Merged segment {i} with previous (gap: {gap_between} frames)")
        else:
            merged_rallies.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    merged_rallies.append((current_start, current_end))

print(f"\nAfter merging (threshold: {MERGE_GAP_THRESHOLD} frames): {len(merged_rallies)} rallies")

# ============================================
# STEP 5: Minimal filtering
# ============================================
# Only remove extremely short segments (< 1.5 seconds)
MIN_RALLY_DURATION = 45  # frames (1.5 seconds) - very lenient (was 60)

final_rallies = []
filtered_count = 0
for s, e in merged_rallies:
    duration = e - s
    if duration >= MIN_RALLY_DURATION:
        final_rallies.append((s, e))
    else:
        filtered_count += 1
        print(f"  Filtered out very short segment: frames {s} → {e} ({duration/FPS:.2f}s)")

print(f"After filtering (min {MIN_RALLY_DURATION/FPS:.1f}s): {len(final_rallies)} rallies")
if filtered_count > 0:
    print(f"  ({filtered_count} segments removed)")

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
        
        status = "✓" if iou >= 70 else ("~" if iou >= 50 else "✗")
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
print(f"Rally Count Match:             {'✓ Yes' if len(final_rallies) == len(GROUND_TRUTH) else '✗ No'}")
print(f"High Quality Matches (≥70%):   {sum(1 for iou in iou_scores if iou >= 70)}")
print(f"Acceptable Matches (≥50%):     {sum(1 for iou in iou_scores if iou >= 50)}")
print(f"Poor Matches (<50%):           {sum(1 for iou in iou_scores if iou < 50)}")

# Calculate precision and recall
true_positives_strict = sum(1 for iou in iou_scores if iou >= 70)
true_positives_loose = sum(1 for iou in iou_scores if iou >= 50)

precision_strict = true_positives_strict / len(final_rallies) if final_rallies else 0
recall_strict = true_positives_strict / len(GROUND_TRUTH) if GROUND_TRUTH else 0
f1_strict = 2 * precision_strict * recall_strict / (precision_strict + recall_strict) if (precision_strict + recall_strict) > 0 else 0

precision_loose = true_positives_loose / len(final_rallies) if final_rallies else 0
recall_loose = true_positives_loose / len(GROUND_TRUTH) if GROUND_TRUTH else 0
f1_loose = 2 * precision_loose * recall_loose / (precision_loose + recall_loose) if (precision_loose + recall_loose) > 0 else 0

print(f"\n--- Strict Metrics (IoU ≥ 70%) ---")
print(f"Precision:                     {precision_strict*100:6.2f}%")
print(f"Recall:                        {recall_strict*100:6.2f}%")
print(f"F1 Score:                      {f1_strict*100:6.2f}%")

print(f"\n--- Loose Metrics (IoU ≥ 50%) ---")
print(f"Precision:                     {precision_loose*100:6.2f}%")
print(f"Recall:                        {recall_loose*100:6.2f}%")
print(f"F1 Score:                      {f1_loose*100:6.2f}%")

print("="*80)

# ============================================
# FINAL ANALYSIS
# ============================================

print("\n" + "="*80)
print("FINAL ANALYSIS")
print("="*80)

rally_diff = len(final_rallies) - len(GROUND_TRUTH)

if rally_diff == 0:
    print("✓ PERFECT COUNT! Detected exactly 6 rallies!")
    if overall_accuracy >= 70:
        print("✓ EXCELLENT ACCURACY! IoU > 70%")
    elif overall_accuracy >= 60:
        print("~ Good accuracy achieved, minor boundary adjustments needed")
    else:
        print("⚠ Rally count is correct but boundaries need refinement")
        print("  Consider fine-tuning the merge threshold slightly")
elif rally_diff > 0:
    print(f"⚠ Over-segmentation: {rally_diff} extra rally(ies) detected")
    print(f"  → INCREASE MERGE_GAP_THRESHOLD from {MERGE_GAP_THRESHOLD} to {MERGE_GAP_THRESHOLD + 10}")
else:
    print(f"⚠ Under-segmentation: Missing {abs(rally_diff)} rally(ies)")
    print(f"  → DECREASE MERGE_GAP_THRESHOLD from {MERGE_GAP_THRESHOLD} to {max(10, MERGE_GAP_THRESHOLD - 3)}")
    print(f"  → Or DECREASE INITIAL_GAP from {INITIAL_GAP} to {max(3, INITIAL_GAP - 1)}")

print(f"\nCurrent parameters:")
print(f"  INITIAL_GAP = {INITIAL_GAP} frames ({INITIAL_GAP/FPS:.2f}s)")
print(f"  MERGE_GAP_THRESHOLD = {MERGE_GAP_THRESHOLD} frames ({MERGE_GAP_THRESHOLD/FPS:.2f}s)")
print(f"  MIN_RALLY_DURATION = {MIN_RALLY_DURATION} frames ({MIN_RALLY_DURATION/FPS:.2f}s)")

print("="*80)
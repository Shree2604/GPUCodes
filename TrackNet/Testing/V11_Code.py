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
print("OPTIMAL RALLY DETECTION - BOUNDARY ALIGNED")
print("="*80)
print(f"Total frames: {len(df)}")
print(f"Video duration: {len(df)/FPS:.2f} seconds ({int(len(df)/FPS//60)}:{int(len(df)/FPS%60):02d})")

# ============================================
# STEP 1: Identify valid ball detections
# ============================================
df['valid'] = (df['Visibility'] == 1) & ~((df['X'] == 0) & (df['Y'] == 0))
valid_frames = df[df['valid']]['Frame'].values

print(f"\nValid tracked frames: {len(valid_frames)} ({len(valid_frames)/len(df)*100:.1f}%)")

# ============================================
# STEP 2: Find all gaps and analyze patterns
# ============================================
all_gaps = []
for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    if gap > 1:
        all_gaps.append({
            'index': i,
            'before': valid_frames[i-1],
            'after': valid_frames[i],
            'size': gap,
        })

print(f"\nAnalyzing {len(all_gaps)} gaps in tracking data...")

# Show larger gaps that are likely rally boundaries
large_gaps = [g for g in all_gaps if g['size'] >= 20]
print(f"\nLarge gaps (≥20 frames, likely rally boundaries):")
print(f"{'#':<4} {'Before':<10} {'After':<10} {'Gap':<10} {'Time'}")
print("-"*50)
for idx, g in enumerate(large_gaps, 1):
    print(f"{idx:<4} {g['before']:<10} {g['after']:<10} {g['size']:<10} "
          f"{int(g['before']/FPS//60):02d}:{int(g['before']/FPS%60):02d}")

# ============================================
# STEP 3: Create initial segments with very small gaps
# ============================================
# Strategy: Detect all small breaks, then merge intelligently
INITIAL_GAP = 6  # Very sensitive to catch all breaks

segments = []
seg_start = valid_frames[0]

for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    
    if gap > INITIAL_GAP:
        seg_end = valid_frames[i-1]
        segments.append((seg_start, seg_end))
        seg_start = valid_frames[i]

segments.append((seg_start, valid_frames[-1]))

print(f"\n{len(segments)} segments detected with INITIAL_GAP={INITIAL_GAP}")

# ============================================
# STEP 4: Smart adaptive merging
# ============================================
# Key insight: Look at the SIZE of segments and gaps
# - Very short segments (<2s) are likely noise/tracking errors
# - Small gaps (<20 frames) within longer play are tracking errors
# - Large gaps (≥20 frames) are rally boundaries

print(f"\nSegment details:")
for idx, (s, e) in enumerate(segments, 1):
    dur = (e - s) / FPS
    print(f"  Seg {idx}: {s:4d}→{e:4d} ({dur:5.2f}s)")

# Calculate context for each gap between segments
segment_contexts = []
for i in range(len(segments) - 1):
    curr_seg = segments[i]
    next_seg = segments[i+1]
    gap = next_seg[0] - curr_seg[1]
    
    curr_duration = (curr_seg[1] - curr_seg[0]) / FPS
    next_duration = (next_seg[1] - next_seg[0]) / FPS
    
    segment_contexts.append({
        'idx': i,
        'gap': gap,
        'curr_dur': curr_duration,
        'next_dur': next_duration,
        'avg_dur': (curr_duration + next_duration) / 2
    })

# Adaptive merging logic
merged = []
current_start, current_end = segments[0]

for i in range(len(segment_contexts)):
    ctx = segment_contexts[i]
    next_seg = segments[i+1]
    gap = ctx['gap']
    
    # Decision rules for merging:
    # 1. If gap is small (<16 frames = 0.53s), always merge (tracking error)
    # 2. If both segments are short (<3s) and gap is medium (<30), merge
    # 3. Otherwise, treat as rally boundary
    
    should_merge = False
    
    if gap <= 16:
        should_merge = True
        reason = "small gap (tracking error)"
    elif gap <= 30 and ctx['avg_dur'] < 3.0:
        should_merge = True
        reason = "medium gap with short segments"
    else:
        should_merge = False
        reason = "rally boundary"
    
    if should_merge:
        current_end = next_seg[1]
        print(f"  → Merge seg {i+1} & {i+2}: gap={gap}f - {reason}")
    else:
        merged.append((current_start, current_end))
        print(f"  → Split at gap={gap}f - {reason}")
        current_start, current_end = next_seg

merged.append((current_start, current_end))

print(f"\nAfter intelligent merging: {len(merged)} rallies")

# ============================================
# STEP 5: Extend boundaries to valid frames
# ============================================
# Extend each rally to include nearby valid frames (within 30 frames)
BOUNDARY_EXTENSION = 30

extended_rallies = []
for start, end in merged:
    # Find valid frames near the boundaries
    near_start = valid_frames[valid_frames >= (start - BOUNDARY_EXTENSION)]
    near_end = valid_frames[valid_frames <= (end + BOUNDARY_EXTENSION)]
    
    if len(near_start) > 0:
        new_start = near_start[0]  # First valid frame near start
    else:
        new_start = start
    
    if len(near_end) > 0:
        new_end = near_end[-1]  # Last valid frame near end
    else:
        new_end = end
    
    extended_rallies.append((new_start, new_end))

print(f"\nAfter boundary extension: {len(extended_rallies)} rallies")

# ============================================
# STEP 6: Filter very short rallies
# ============================================
MIN_RALLY_DURATION = 60  # 2 seconds

final_rallies = []
for s, e in extended_rallies:
    if (e - s) >= MIN_RALLY_DURATION:
        final_rallies.append((s, e))
    else:
        print(f"  Filtered: {s}→{e} ({(e-s)/FPS:.2f}s)")

print(f"\nFinal rallies: {len(final_rallies)}")

# ============================================
# ACCURACY CALCULATION
# ============================================

def calculate_iou(detected, ground_truth):
    inter_start = max(detected[0], ground_truth[0])
    inter_end = min(detected[1], ground_truth[1])
    intersection = max(0, inter_end - inter_start)
    
    union_start = min(detected[0], ground_truth[0])
    union_end = max(detected[1], ground_truth[1])
    union = union_end - union_start
    
    return (intersection / union * 100) if union > 0 else 0

def frame_to_timestamp(frame):
    seconds = frame / FPS
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

# Match detected to ground truth
iou_scores = []
rally_matches = []

for gt_rally in GROUND_TRUTH:
    best_iou = 0
    best_match = None
    
    for j, det_rally in enumerate(final_rallies):
        iou = calculate_iou(det_rally, gt_rally)
        if iou > best_iou:
            best_iou = iou
            best_match = j
    
    iou_scores.append(best_iou)
    rally_matches.append(best_match)

overall_accuracy = np.mean(iou_scores) if iou_scores else 0

# ============================================
# RESULTS OUTPUT
# ============================================

print("\n" + "="*80)
print("DETECTED RALLIES")
print("="*80)
for i, (start, end) in enumerate(final_rallies, 1):
    duration = (end - start) / FPS
    print(f"Rally {i}: Frames {start:4d} → {end:4d} | "
          f"{frame_to_timestamp(start)} → {frame_to_timestamp(end)} | "
          f"{duration:5.2f} sec")

print("\n" + "="*80)
print("GROUND TRUTH RALLIES")
print("="*80)
for i, (start, end) in enumerate(GROUND_TRUTH, 1):
    duration = (end - start) / FPS
    print(f"Rally {i}: Frames {start:4d} → {end:4d} | "
          f"{frame_to_timestamp(start)} → {frame_to_timestamp(end)} | "
          f"{duration:5.2f} sec")

print("\n" + "="*80)
print("IoU ACCURACY ANALYSIS")
print("="*80)
print(f"{'GT':<6} {'Detected':<10} {'IoU':<10} {'Start Δ':<12} {'End Δ':<10} {'Status'}")
print("-"*80)

for i, (iou, match_idx) in enumerate(zip(iou_scores, rally_matches), 1):
    if match_idx is not None:
        det = final_rallies[match_idx]
        gt = GROUND_TRUTH[i-1]
        start_diff = det[0] - gt[0]
        end_diff = det[1] - gt[1]
        
        if iou >= 70:
            status = "✓ Excellent"
        elif iou >= 50:
            status = "~ Good"
        else:
            status = "✗ Poor"
        
        print(f"R{i:<5} R{match_idx+1:<9} {iou:6.2f}%   {start_diff:+5d}f       "
              f"{end_diff:+5d}f     {status}")
    else:
        print(f"R{i:<5} {'NONE':<9} {iou:6.2f}%   {'N/A':<11} {'N/A':<10} ✗ No match")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Overall IoU Accuracy:        {overall_accuracy:6.2f}%")
print(f"Detected / Ground Truth:     {len(final_rallies)} / {len(GROUND_TRUTH)}")
print(f"Excellent matches (≥70%):    {sum(1 for iou in iou_scores if iou >= 70)}")
print(f"Good matches (≥50%):         {sum(1 for iou in iou_scores if iou >= 50)}")
print(f"Poor matches (<50%):         {sum(1 for iou in iou_scores if iou < 50)}")

tp_70 = sum(1 for iou in iou_scores if iou >= 70)
tp_50 = sum(1 for iou in iou_scores if iou >= 50)

print(f"\nPrecision (≥50%):            {tp_50/len(final_rallies)*100:6.2f}%")
print(f"Recall (≥50%):               {tp_50/len(GROUND_TRUTH)*100:6.2f}%")
print(f"F1 Score (≥50%):             {2*tp_50/(len(final_rallies)+len(GROUND_TRUTH))*100:6.2f}%")
print("="*80)

# Final recommendations
if len(final_rallies) == len(GROUND_TRUTH):
    if overall_accuracy >= 50:
        print("\n✓ SUCCESS! Correct count with good accuracy!")
    else:
        print("\n~ Correct count but boundaries need fine-tuning")
        print("  Try adjusting BOUNDARY_EXTENSION parameter")
elif len(final_rallies) < len(GROUND_TRUTH):
    print(f"\n⚠ Missing {len(GROUND_TRUTH) - len(final_rallies)} rally(ies)")
    print("  → Decrease INITIAL_GAP or adjust merging rules")
else:
    print(f"\n⚠ {len(final_rallies) - len(GROUND_TRUTH)} extra rally(ies)")
    print("  → Increase INITIAL_GAP or make merging more aggressive")
import pandas as pd
import numpy as np

FPS = 30

# Ground truth in frames
GROUND_TRUTH = [
    (0, 480),
    (570, 840),
    (930, 1290),
    (1380, 1620),
    (1680, 2010),
    (2070, 2156)
]

df = pd.read_csv("./Code_ball.csv")

df['valid'] = (df['Visibility'] == 1) & ~((df['X'] == 0) & (df['Y'] == 0))
valid_frames = df[df['valid']]['Frame'].values

# STEP 1: detect short segments (gap >= 10)
SEG_GAP = 10
segments = []
start = valid_frames[0]

for i in range(1, len(valid_frames)):
    if valid_frames[i] - valid_frames[i-1] > SEG_GAP:
        segments.append((start, valid_frames[i-1]))
        start = valid_frames[i]
segments.append((start, valid_frames[-1]))

# STEP 2: merge segments that are too close (< 40 frames gap)
MERGE_GAP = 40
merged = []
curr_start, curr_end = segments[0]

for s, e in segments[1:]:
    if s - curr_end < MERGE_GAP:
        curr_end = e  # merge
    else:
        merged.append((curr_start, curr_end))
        curr_start, curr_end = s, e

merged.append((curr_start, curr_end))

# accuracy: IoU
def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0

accuracy_list = []
for det, gt in zip(merged, GROUND_TRUTH):
    accuracy_list.append(iou(det, gt))

overall = sum(accuracy_list) / len(accuracy_list)

# print
def t(f): return f"{int((f/FPS)//60):02d}:{int((f/FPS)%60):02d}"

print("\nDetected Rallies (Final Fix):")
for i,(s,e) in enumerate(merged,1):
    print(f"Rally {i}: Frames {s} → {e} | {t(s)} → {t(e)} | {(e-s)/FPS:.2f} sec")

print("\nIoU Accuracy per rally:")
for i,acc in enumerate(accuracy_list,1):
    print(f"Rally {i}: {acc*100:.2f}%")

print(f"\nOverall Accuracy: {overall*100:.2f}%")

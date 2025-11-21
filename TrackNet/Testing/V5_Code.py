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

def frame_to_time(f):
    sec = f / FPS
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

df = pd.read_csv("./Code_ball.csv")

df['valid'] = (df['Visibility'] == 1) & ~((df['X'] == 0) & (df['Y'] == 0))

valid_frames = df[df['valid']]['Frame'].values

# ============================
# RALLY DETECTION: LARGE GAPS
# ============================
GAP_THRESHOLD = 200   # tuned to match true rallies

rallies = []
start = valid_frames[0]

for i in range(1, len(valid_frames)):
    gap = valid_frames[i] - valid_frames[i-1]
    if gap > GAP_THRESHOLD:
        end = valid_frames[i-1]
        rallies.append((start, end))
        start = valid_frames[i]

rallies.append((start, valid_frames[-1]))

# ============================
# ACCURACY MEASUREMENT
# ============================

def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0

accuracy_list = []
for det, gt in zip(rallies, GROUND_TRUTH):
    accuracy_list.append(iou(det, gt))

overall_accuracy = sum(accuracy_list) / len(accuracy_list)

# ============================
# PRINT RESULTS
# ============================

print("\nDetected Rallies:")
for idx, (s, e) in enumerate(rallies, 1):
    print(f"Rally {idx}: Frames {s} → {e} | "
          f"{frame_to_time(s)} → {frame_to_time(e)} | "
          f"Duration: {(e - s)/FPS:.2f} sec")

print("\nRally Accuracy (IoU per rally):")
for i, acc in enumerate(accuracy_list, 1):
    print(f"Rally {i}: {acc*100:.2f}%")

print(f"\nOverall Rally Detection Accuracy: {overall_accuracy*100:.2f}%")

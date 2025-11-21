import pandas as pd

FPS = 30
GAP_THRESHOLD = 20    # break if no ball movement for > 20 frames

df = pd.read_csv("./Code_ball.csv")

# valid detection
df['valid'] = (df['Visibility'] == 1) & ~((df['X'] == 0) & (df['Y'] == 0))

# Get all frames where ball is visible
valid_frames = df[df['valid']]['Frame'].values

rallies = []
start = valid_frames[0]

for i in range(1, len(valid_frames)):
    if valid_frames[i] - valid_frames[i-1] > GAP_THRESHOLD:
        # break detected
        end = valid_frames[i-1]
        rallies.append((start, end))
        start = valid_frames[i]

# add last rally
rallies.append((start, valid_frames[-1]))


def frame_to_time(f):
    sec = f / FPS
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

print("Detected Rallies:")
for idx, (s, e) in enumerate(rallies, 1):
    print(f"Rally {idx}: Frames {s} → {e} | {frame_to_time(s)} → {frame_to_time(e)} | Duration: {(e - s)/FPS:.2f} sec")

import pandas as pd
import numpy as np

FPS = 30
STOP_THRESHOLD = 2        # speed below this = ball is stopped
BREAK_TIME = 0.5          # rally break if stop > 0.5 sec
MAX_MISSING = 5           # missing frames allowed inside rally

df = pd.read_csv("./Code_ball.csv")

# Identify valid positions (ball tracked)
valid = (df["Visibility"] == 1) & ~((df["X"] == 0) & (df["Y"] == 0))

# Compute speed
df["speed"] = np.sqrt((df["X"].diff() ** 2) + (df["Y"].diff() ** 2))

rallies = []
start = None
stop_counter = 0
missing_counter = 0

for i in range(1, len(df)):

    if not valid[i]:
        missing_counter += 1
        if missing_counter > MAX_MISSING and start is not None:
            end = df.loc[i - missing_counter, "Frame"]
            rallies.append((start, end))
            start = None
        continue

    # Speed of ball
    spd = df.loc[i, "speed"]

    if spd > STOP_THRESHOLD:
        if start is None:
            start = df.loc[i, "Frame"]
        stop_counter = 0
        missing_counter = 0
    else:
        stop_counter += 1
        if stop_counter > BREAK_TIME * FPS and start is not None:
            end = df.loc[i - stop_counter, "Frame"]
            rallies.append((start, end))
            start = None
            stop_counter = 0

# If ending on rally
if start is not None:
    rallies.append((start, df.loc[len(df)-1, "Frame"]))

def frame_to_time(f):
    sec = f / FPS
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

print("Detected Rallies:")
for idx, (s, e) in enumerate(rallies, 1):
    print(f"Rally {idx}: Frames {s} → {e} | {frame_to_time(s)} → {frame_to_time(e)} | Duration: {(e - s)/FPS:.2f} sec")

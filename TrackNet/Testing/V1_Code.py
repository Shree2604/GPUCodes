import pandas as pd

FPS = 30
MISS_TOLERANCE = 3     # Allowed gap of missing ball frames inside a rally

df = pd.read_csv("./Code_ball.csv")

# Mark ball missing if visibility == 0 or coordinates are (0,0)
df['ball_present'] = (df['Visibility'] == 1) & ~((df['X'] == 0) & (df['Y'] == 0))

rallies = []
start = None
miss_count = 0

for i in range(len(df)):
    if df.loc[i, 'ball_present']:
        if start is None:
            start = df.loc[i, 'Frame']
        miss_count = 0
    else:
        if start is not None:
            miss_count += 1
            if miss_count > MISS_TOLERANCE:
                end = df.loc[i - miss_count, 'Frame']
                rallies.append((start, end))
                start = None
                miss_count = 0

# If rally continues till last frame
if start is not None:
    rallies.append((start, df.loc[len(df)-1, 'Frame']))

# Convert to timestamp
def frame_to_time(f):
    sec = f / FPS
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

print("Detected Rallies:")
for idx, (s, e) in enumerate(rallies, 1):
    print(f"Rally {idx}: Frames {s} → {e} | {frame_to_time(s)} → {frame_to_time(e)} | Duration: {(e - s)/FPS:.2f} sec")

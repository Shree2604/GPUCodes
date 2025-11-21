import pandas as pd
import numpy as np

FPS = 30

df = pd.read_csv("Code_ball.csv")

# -------------------------------------------
# STEP 1: Compute speed
# -------------------------------------------
df['dx'] = df['X'].diff().fillna(0)
df['dy'] = df['Y'].diff().fillna(0)
df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
df['speed'] = df['speed'].rolling(7, min_periods=1).mean()

# -------------------------------------------
# STEP 2: Compute direction (left-right)
# -------------------------------------------
median_x = df['X'].median()
df['side'] = (df['X'] > median_x).astype(int)
df['side_change'] = df['side'].diff().abs().fillna(0)

# -------------------------------------------
# STEP 3: Combine signals
# -------------------------------------------
# If ball is moving OR switching sides => active rally
ACTIVE_SPEED = 3.5
ACTIVE_SIDE_CHANGE = 1

df['active'] = ((df['speed'] > ACTIVE_SPEED) | (df['side_change'] == ACTIVE_SIDE_CHANGE)).astype(int)

# Smooth activity
df['active_smooth'] = df['active'].rolling(10, min_periods=1).mean()
df['active_smooth'] = (df['active_smooth'] > 0.3).astype(int)

# -------------------------------------------
# STEP 4: Extract rally segments
# -------------------------------------------
segments = []
started = None

for i in range(len(df)):
    if df['active_smooth'][i] == 1 and started is None:
        started = i
    if df['active_smooth'][i] == 0 and started is not None:
        segments.append((started, i-1))
        started = None

if started is not None:
    segments.append((started, len(df)-1))

# -------------------------------------------
# STEP 5: Merge segments that are too close
# -------------------------------------------
MERGE_GAP = 40     # < 1.3 sec break = same rally
merged = []

cs, ce = segments[0]
for s, e in segments[1:]:
    if s - ce <= MERGE_GAP:
        ce = e  # extend
    else:
        merged.append((cs, ce))
        cs, ce = s, e
merged.append((cs, ce))

# -------------------------------------------
# STEP 6: Remove very short false rallies
# -------------------------------------------
FINAL_MIN_DURATION = 60   # 2 seconds
final_rallies = [(s, e) for (s, e) in merged if (e - s) >= FINAL_MIN_DURATION]

# -------------------------------------------
# PRINT RESULTS
# -------------------------------------------
def t(f):
    return f"{int((f/FPS)//60):02d}:{int((f/FPS)%60):02d}"

print("\nDetected Rallies:")
for i, (s, e) in enumerate(final_rallies, 1):
    print(f"Rally {i}: Frames {s} → {e} | {t(s)} → {t(e)} | {(e-s)/FPS:.2f} sec")

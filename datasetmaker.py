import random
import pandas as pd
from unifiedairfoilgenerator import unifiedairfoilgenerator

data = []
airfoil_gen = unifiedairfoilgenerator()
attempts = 0
while len(data) < 10000:
    max_camber = round(random.uniform(0, 0.05), 3)
    camber_pos = round(random.uniform(0.2, 0.7),3)
    thickness = round(random.uniform(0.08, 0.15),3)
    reynolds = random.randint(300000, 1000000)
    aoa = round(random.uniform(-4, 8), 1)
    attempts += 1
    try:
        x, y = airfoil_gen.generate_airfoil(max_camber, camber_pos, thickness)
        cl, cd = airfoil_gen.run_xfoil_automated(reynolds, aoa)
        if cl is not None and cd is not None:
            data.append((max_camber, camber_pos, thickness, reynolds, aoa, cl, cd))
            print("added data point:", data[-1])
        else:
            print("Failed to get cl/cd for parameters:", (max_camber, camber_pos, thickness, reynolds, aoa))
    except Exception as e:
        print(f"Error generating data point: {e}")
    print(f"Attempts: {attempts}, Data points collected: {len(data)}")
df = pd.DataFrame(data, columns=["max_camber", "camber_pos", "thickness", "reynolds", "aoa", "cl", "cd"])
df.to_csv("airfoil_dataset.csv", index=False)
df.head()
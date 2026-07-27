import random
import pandas as pd
from unifiedairfoilgenerator import unifiedairfoilgenerator
import numpy as np
from optimizer import AirfoilOptimizer
import joblib


model_cl = joblib.load("airfoil_surrogate_model_cl.pkl")
model_cd = joblib.load("airfoil_surrogate_model_cd.pkl")
scaler = joblib.load("airfoil_scaler.pkl")
airfoil_gen = unifiedairfoilgenerator()

test_cases = []
while len(test_cases) < 10:
    aoa = round(random.uniform(-4, 8), 1)
    reynolds = round(random.uniform(300000, 1000000), 0)
    test_cases.append((aoa, reynolds))
print("Test Cases (AoA, Reynolds):", test_cases)


random_airfoils = []
for i in range (100):
    max_camber = round(random.uniform(0, 0.05), 4)
    camber_pos = round(random.uniform(0.2, 0.7), 4)
    thickness = round(random.uniform(0.08, 0.15), 4)
    random_airfoils.append((max_camber, camber_pos, thickness))
print("Generated 100 random airfoils for testing.")

best_gen_LD = -1000
cl_high =0
cd_low = 1000

best_gen_airfoil = []
ai_optimized_LD = 0
ai_optimized_airfoil = []

for case in test_cases:
    for airfoil in random_airfoils:
        cl = model_cl.predict(scaler.transform([[airfoil[0], airfoil[1], airfoil[2], case[1], case[0]]]))[0]
        cd = model_cd.predict(scaler.transform([[airfoil[0], airfoil[1], airfoil[2], case[1], case[0]]]))[0]
        if cl / (10**cd) > best_gen_LD:
            best_gen_LD = cl / (10**cd)
            cl_high = cl
            cd_low = 10**cd
            best_gen_airfoil = airfoil
    print("Random Generated Airfoil - AoA:", case[0], "Reynolds:", case[1], "Best L/D:", best_gen_LD, "cl:", cl_high, "cd:", cd_low, "Best Airfoil Params:", best_gen_airfoil)
    optimizer = AirfoilOptimizer(case[0], case[1])
    res = optimizer.optimize_airfoil()
    ai_optimized_airfoil = res.x
    ai_optimized_LD = -res.fun
    print("AI Optimized Airfoil - AoA:", case[0], "Reynolds:", case[1], "Best L/D:", ai_optimized_LD, "Best Airfoil Params:", ai_optimized_airfoil)
    best_gen_LD = -1000
    cl_high =0
    cd_low = 1000
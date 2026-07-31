import random
import pandas as pd
import numpy as np
import joblib
from unifiedairfoilgenerator import unifiedairfoilgenerator
from optimizer import AirfoilOptimizer

# 1. Load ML models & generator
model_cl = joblib.load("airfoil_surrogate_model_cl.pkl")
model_cd = joblib.load("airfoil_surrogate_model_cd.pkl")
scaler = joblib.load("airfoil_scaler.pkl")
airfoil_gen = unifiedairfoilgenerator()

# 2. Generate 10 test conditions
test_cases = []
while len(test_cases) < 10:
    aoa = round(random.uniform(-4, 8), 1)
    reynolds = round(random.uniform(300000, 1000000), 0)
    test_cases.append((aoa, reynolds))
print("Test Cases (AoA, Reynolds):", test_cases)

# 3. Generate 100 random airfoils
random_airfoils = []
for i in range(100):
    max_camber = round(random.uniform(0, 0.05), 4)
    camber_pos = round(random.uniform(0.2, 0.7), 4)
    thickness = round(random.uniform(0.08, 0.15), 4)
    random_airfoils.append((max_camber, camber_pos, thickness))
print("Generated 100 random airfoils for testing.")

results_list = []

for case in test_cases:
    aoa = case[0]
    reynolds = case[1]
    reynolds_log = np.log10(reynolds)
    
    # Reset tracking variables per case
    best_gen_LD = -1000
    best_rnd_ai_cl = 0
    best_rnd_ai_cd = 1000
    best_gen_airfoil = []

    # EVALUATE RANDOM AIRFOILS AI
    for airfoil in random_airfoils:
        features = np.array([[airfoil[0], airfoil[1], airfoil[2], reynolds_log, aoa]])
        scaled_feat = scaler.transform(features)
        
        cl = model_cl.predict(scaled_feat)[0]
        cd_log = model_cd.predict(scaled_feat)[0]
        cd = 10**cd_log
        cd = max(cd, 0.005) #to avoid small drag values that are unrealistic
        
        ld = cl / cd
        if ld > best_gen_LD:
            best_gen_LD = ld
            best_rnd_ai_cl = cl
            best_rnd_ai_cd = cd
            best_gen_airfoil = airfoil

    print("Evaluating Case -> AoA:", aoa, "Re:", int(reynolds))
    print("Best Random (AI) -> Params:", best_gen_airfoil, "CL:", round(best_rnd_ai_cl, 4), "CD:", round(best_rnd_ai_cd, 5), "L/D:", round(best_gen_LD, 2))

    #  XFOIL GROUND-TRUTH FOR BEST RANDOM
    airfoil_gen.generate_airfoil(best_gen_airfoil[0], best_gen_airfoil[1], best_gen_airfoil[2])
    rnd_xfoil_cl, rnd_xfoil_cd = airfoil_gen.run_xfoil_automated(reynolds, aoa)
    
    if rnd_xfoil_cl is not None and rnd_xfoil_cd is not None and rnd_xfoil_cd > 0:
        rnd_xfoil_ld = rnd_xfoil_cl / rnd_xfoil_cd
        print("Best Random (XFOIL) L/D:", round(rnd_xfoil_ld, 2))
    else:
        rnd_xfoil_cl, rnd_xfoil_cd, rnd_xfoil_ld = np.nan, np.nan, np.nan

    # Percent error between AI prediction and XFOIL for random best
    if not np.isnan(rnd_xfoil_ld) and rnd_xfoil_ld != 0:
        rnd_pct_error = abs((best_gen_LD - rnd_xfoil_ld) / rnd_xfoil_ld) * 100
    else:
        rnd_pct_error = np.nan

    #  AI OPTIMIZER RUN 
    optimizer = AirfoilOptimizer(aoa, reynolds)
    res = optimizer.optimize_airfoil()
    ai_opt_airfoil = res.x
    
    # Calculate AI prediction for optimized shape directly from surrogate pipeline
    opt_feat = scaler.transform(np.array([[ai_opt_airfoil[0], ai_opt_airfoil[1], ai_opt_airfoil[2], reynolds_log, aoa]]))
    opt_ai_cl = model_cl.predict(opt_feat)[0]
    opt_ai_cd_log = model_cd.predict(opt_feat)[0]
    opt_ai_cd = max(10**opt_ai_cd_log, 0.005)
    opt_ai_ld = opt_ai_cl / opt_ai_cd

    print("AI Optimized     -> Params:", [round(p,4) for p in ai_opt_airfoil],  "CL:", round(opt_ai_cl, 4), "CD:", round(opt_ai_cd, 5), "L/D:", round(opt_ai_ld, 2))

    # XFOIL GROUND-TRUTH FOR OPTIMIZED AIRFOIL 
    airfoil_gen.generate_airfoil(ai_opt_airfoil[0], ai_opt_airfoil[1], ai_opt_airfoil[2])
    opt_xfoil_cl, opt_xfoil_cd = airfoil_gen.run_xfoil_automated(reynolds, aoa)
    
    if opt_xfoil_cl is not None and opt_xfoil_cd is not None and opt_xfoil_cd > 0:
        opt_xfoil_ld = opt_xfoil_cl / opt_xfoil_cd
        print("AI Optimized (XFOIL) L/D:", round(opt_xfoil_ld, 2))
    else:
        opt_xfoil_cl, opt_xfoil_cd, opt_xfoil_ld = np.nan, np.nan, np.nan

    # Percent error between AI prediction and XFOIL for optimized
    if not np.isnan(opt_xfoil_ld) and opt_xfoil_ld != 0:
        opt_pct_error = abs((opt_ai_ld - opt_xfoil_ld) / opt_xfoil_ld) * 100
    else:
        opt_pct_error = np.nan

    #  PERFORMANCE ADVANTAGE (XFOIL Ground Truth Baseline) 
    if not np.isnan(opt_xfoil_ld) and not np.isnan(rnd_xfoil_ld) and rnd_xfoil_ld != 0:
        opt_vs_rnd_pct_diff = ((opt_xfoil_ld - rnd_xfoil_ld) / rnd_xfoil_ld) * 100
    else:
        opt_vs_rnd_pct_diff = np.nan

    #  APPEND ROW TO DATASET 
    results_list.append({
        "AoA": aoa,
        "Reynolds": int(reynolds),
        
        # Best Random Columns
        "best_random_camber": round(best_gen_airfoil[0], 4),
        "best_random_camber_pos": round(best_gen_airfoil[1], 4),
        "best_random_thickness": round(best_gen_airfoil[2], 4),
        "random_ai_cl": round(best_rnd_ai_cl, 4),
        "random_ai_cd": round(best_rnd_ai_cd, 5),
        "random_ai_clcd": round(best_gen_LD, 2),
        "random_xfoil_cl": round(rnd_xfoil_cl, 4) if not np.isnan(rnd_xfoil_cl) else np.nan,
        "random_xfoil_cd": round(rnd_xfoil_cd, 5) if not np.isnan(rnd_xfoil_cd) else np.nan,
        "random_xfoil_clcd": round(rnd_xfoil_ld, 2) if not np.isnan(rnd_xfoil_ld) else np.nan,
        "random_ai_vs_xfoil_pct_error": round(rnd_pct_error, 2) if not np.isnan(rnd_pct_error) else np.nan,
        
        # Optimized Columns
        "optimized_camber": round(ai_opt_airfoil[0], 4),
        "optimized_camber_pos": round(ai_opt_airfoil[1], 4),
        "optimized_thickness": round(ai_opt_airfoil[2], 4),
        "optimized_ai_cl": round(opt_ai_cl, 4),
        "optimized_ai_cd": round(opt_ai_cd, 5),
        "optimized_ai_clcd": round(opt_ai_ld, 2),
        "optimized_xfoil_cl": round(opt_xfoil_cl, 4) if not np.isnan(opt_xfoil_cl) else np.nan,
        "optimized_xfoil_cd": round(opt_xfoil_cd, 5) if not np.isnan(opt_xfoil_cd) else np.nan,
        "optimized_xfoil_clcd": round(opt_xfoil_ld, 2) if not np.isnan(opt_xfoil_ld) else np.nan,
        "optimized_ai_vs_xfoil_pct_error": round(opt_pct_error, 2) if not np.isnan(opt_pct_error) else np.nan,
        
        # Final Advantage
        "opt_vs_rnd_xfoil_pct_diff": round(opt_vs_rnd_pct_diff, 2) if not np.isnan(opt_vs_rnd_pct_diff) else np.nan
    })

#  SAVE TO CSV ---
df_results = pd.DataFrame(results_list)
df_results.to_csv("validation_results.csv", index=False)
print("\n CSV validation report saved as 'validation_results.csv'!")
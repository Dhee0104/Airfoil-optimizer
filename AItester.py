import numpy as np
import joblib
from unifiedairfoilgenerator import unifiedairfoilgenerator

model_cl = joblib.load("airfoil_surrogate_model_cl.pkl")
model_cd = joblib.load("airfoil_surrogate_model_cd.pkl")
scaler = joblib.load("airfoil_scaler.pkl")
airfoil_gen = unifiedairfoilgenerator()
target_aoa = 5.0
target_reynolds = 300000
max_camber = .02
camber_pos = .4
thickness = .12
ai_cl = model_cl.predict(scaler.transform([[max_camber, camber_pos, thickness, target_reynolds, target_aoa]]))[0]
ai_cd = model_cd.predict(scaler.transform([[max_camber, camber_pos, thickness, target_reynolds, target_aoa]]))[0]
ai_cd = 10**ai_cd
airfoil_gen.generate_airfoil(max_camber, camber_pos, thickness)
xfoil_cl, xfoil_cd = airfoil_gen.run_xfoil_automated(target_reynolds, target_aoa)
print("AI Predicted CL:", ai_cl, "AI Predicted CD:", ai_cd)
print("XFOIL Predicted CL:", xfoil_cl, "XFOIL Predicted CD:", xfoil_cd)

import numpy as np
import joblib
from unifiedairfoilgenerator import unifiedairfoilgenerator

model_cl = joblib.load("airfoil_surrogate_model_cl.pkl")
model_cd = joblib.load("airfoil_surrogate_model_cd.pkl")
scaler = joblib.load("airfoil_scaler.pkl")
airfoil_gen = unifiedairfoilgenerator()
target_aoa = float(input("Enter target angle of attack (AOA) in degrees: "))
target_reynolds = float(input("Enter target Reynolds number: "))
max_camber = float(input("Enter max camber (0 to 0.1): "))
camber_pos = float(input("Enter camber position (0.1 to 0.8): "))
thickness = float(input("Enter thickness (0.07 to 0.2): "))
log_reynolds = np.log10(target_reynolds)
ai_cl = model_cl.predict(scaler.transform([[max_camber, camber_pos, thickness, log_reynolds, target_aoa]]))[0]
ai_cd = model_cd.predict(scaler.transform([[max_camber, camber_pos, thickness, log_reynolds, target_aoa]]))[0]
ai_cd = 10**ai_cd
airfoil_gen.generate_airfoil(max_camber, camber_pos, thickness)
xfoil_cl, xfoil_cd = airfoil_gen.run_xfoil_automated(target_reynolds, target_aoa)
print("AI Predicted CL:", ai_cl, "AI Predicted CD:", ai_cd)
print("XFOIL Predicted CL:", xfoil_cl, "XFOIL Predicted CD:", xfoil_cd)

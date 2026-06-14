import numpy as np
import joblib
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

model_cl = joblib.load("airfoil_surrogate_model_cl.pkl")
model_cd = joblib.load("airfoil_surrogate_model_cd.pkl")
scaler = joblib.load("airfoil_scaler.pkl")

target_aoa = float(input("Enter target angle of attack (AOA) in degrees: "))
target_reynolds = float(input("Enter target Reynolds number: "))
space = [
    Real(0, 0.1, name='max_camber'),
    Real(0.1, 0.8, name='camber_pos'),
    Real(0.07, 0.2, name='thickness')
]
log_reynolds = np.log10(target_reynolds)

@use_named_args(space)
def get_airfoil_performance(max_camber, camber_pos, thickness):
    input_features = np.array([[max_camber, camber_pos, thickness, log_reynolds, target_aoa]])
    input_scaled = scaler.transform(input_features)
    predicted_cl = model_cl.predict(input_scaled)[0]
    predicted_cd_log = model_cd.predict(input_scaled)[0]
    predicted_cd = 10**predicted_cd_log
    if predicted_cd <= 0.005:
        return 0.005
    print("Predicted CL:", predicted_cl, "Predicted CD:", predicted_cd)
    return -predicted_cl / predicted_cd

@use_named_args(space)
def get_airfoil_performance_cl_only(max_camber, camber_pos, thickness):
    input_features = np.array([[max_camber, camber_pos, thickness, log_reynolds, target_aoa]])
    input_scaled = scaler.transform(input_features)
    predicted_cl = model_cl.predict(input_scaled)[0]
    print("Predicted CL:", predicted_cl)
    return -predicted_cl

print("optimizing")
res = gp_minimize(func=get_airfoil_performance, dimensions=space, n_calls=100, n_random_starts=10, random_state=42)

optimal_max_camber, optimal_camber_pos, optimal_thickness = res.x
max_cl_cd = -res.fun
print("optimal max camber:", optimal_max_camber)
print("optimal camber position:", optimal_camber_pos)
print("optimal thickness:", optimal_thickness)
print("maximum CL/CD ratio:", max_cl_cd)
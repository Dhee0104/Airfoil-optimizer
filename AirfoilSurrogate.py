import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor  
import joblib
import numpy as np

df = pd.read_csv("airfoil_dataset.csv")
X = df[["max_camber", "camber_pos", "thickness", "reynolds", "aoa"]].values
y_cl = df["cl"].values
y_cd = df["cd"].values
y_cd_log = np.log(y_cd)
X_train, X_test, Y_train_cl, Y_test_cl = train_test_split(X, y_cl, test_size=0.2, random_state=42)
X_train, X_test, Y_train_cd_log, Y_test_cd_log = train_test_split(X, y_cd_log, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_cl = MLPRegressor(hidden_layer_sizes=(128, 128, 128, 64, 32), activation='relu', solver='adam', max_iter=1000, early_stopping=True, random_state=42)
print("Training cl model")
model_cl.fit(X_train_scaled, Y_train_cl)
train_score_cl = model_cl.score(X_train_scaled, Y_train_cl)
test_score_cl = model_cl.score(X_test_scaled, Y_test_cl)
print("Training R^2 Score (CL):", train_score_cl)
print("Testing R^2 Score (CL):", test_score_cl)

print("Training cd model")
model_cd = MLPRegressor(hidden_layer_sizes=(128, 128, 128, 64, 32), activation='relu', solver='adam', max_iter=1000, early_stopping=True, random_state=42)
model_cd.fit(X_train_scaled, Y_train_cd_log)
train_score_cd = model_cd.score(X_train_scaled, Y_train_cd_log)
test_score_cd = model_cd.score(X_test_scaled, Y_test_cd_log)
print("Training R^2 Score (CD):", train_score_cd)
print("Testing R^2 Score (CD):", test_score_cd)
joblib.dump(model_cl, "airfoil_surrogate_model_cl.pkl")
joblib.dump(model_cd, "airfoil_surrogate_model_cd.pkl")
joblib.dump(scaler, "airfoil_scaler.pkl")


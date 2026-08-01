import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor  
import joblib

df = pd.read_csv("airfoil_dataset.csv")



X = df[["max_camber", "camber_pos", "thickness", "reynolds", "aoa"]].values
y_cl = df["cl"].values
y_cd_log = np.log(df["cd"].values) 

Y_combined = np.column_stack((y_cl, y_cd_log))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_combined, test_size=0.2, random_state=42)

Y_train_cl, Y_train_cd_log = Y_train[:, 0], Y_train[:, 1]
Y_test_cl, Y_test_cd_log = Y_test[:, 0], Y_test[:, 1]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Training CL model")
model_cl = MLPRegressor(hidden_layer_sizes=(128, 128, 128, 64, 32), activation='relu', solver='adam', max_iter=1000, early_stopping=True, random_state=42)
model_cl.fit(X_train_scaled, Y_train_cl)
print("Training R^2 Score (CL):", model_cl.score(X_train_scaled, Y_train_cl))
print("Testing R^2 Score (CL):", model_cl.score(X_test_scaled, Y_test_cl))

print("Training CD model")
model_cd = MLPRegressor(hidden_layer_sizes=(128, 128, 128, 64, 32), activation='relu', solver='adam', max_iter=1000, early_stopping=True, random_state=42)
model_cd.fit(X_train_scaled, Y_train_cd_log)
print("Training R^2 Score (CD):", model_cd.score(X_train_scaled, Y_train_cd_log))
print("Testing R^2 Score (CD):", model_cd.score(X_test_scaled, Y_test_cd_log))


joblib.dump(model_cl, "airfoil_surrogate_model_cl.pkl")
joblib.dump(model_cd, "airfoil_surrogate_model_cd.pkl")
joblib.dump(scaler, "airfoil_scaler.pkl") 

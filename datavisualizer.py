import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("airfoil_dataset.csv")
inputs = df[["max_camber", "camber_pos", "thickness", "reynolds", "aoa"]]
for col in inputs.columns:
    plt.figure()
    plt.hist(inputs[col], bins=20, edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()
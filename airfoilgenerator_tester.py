import os
import numpy as np
from pyxfoil import Xfoil,set_workdir
import matplotlib.pyplot as plt

set_workdir(os.getcwd()) # Set the working directory to the current directory

def generate_airfoil(max_camber, camber_pos, thickness, filepath= "temp_airfoil.dat", num_points=100):
    #https://en.wikipedia.org/wiki/NACA_airfoil use formulas to generate the airfoil coordinates
    x = np.linspace(0, 1, num_points) # chord line
    yt = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4) # thickness distribution
    yc = np.zeros_like(x) 
    dycdx = np.zeros_like(x)
    if camber_pos > 0: 
        idx_fwd = x <= camber_pos # idx_fwd are the indexes of the points before the camber position
        yc[idx_fwd] = (max_camber / camber_pos**2) * (2 * camber_pos * x[idx_fwd] - x[idx_fwd]**2) #camber line
        dycdx[idx_fwd] = (2 * max_camber / camber_pos**2) * (camber_pos - x[idx_fwd])#camber slope
        idx_aft = x > camber_pos # idx_aft are the indexes of the points after the camber position
        yc[idx_aft] = (max_camber / (1 - camber_pos)**2) * ((1 - 2 * camber_pos) + 2 * camber_pos * x[idx_aft] - x[idx_aft]**2)#camber line
        dycdx[idx_aft] = (2 * max_camber / (1 - camber_pos)**2) * (camber_pos - x[idx_aft])#camber slope
    theta = np.arctan(dycdx) #angle from the slope of camber line
    
    #use the camber line and thickness distribution to calculate the upper and lower surface coordinates
    x_upper  = x - yt * np.sin(theta)
    y_upper = yc + yt * np.cos(theta)
    x_lower = x + yt * np.sin(theta)
    y_lower = yc - yt * np.cos(theta)
    
    #writes cords in selig format
    x_coords = np.concatenate([x_upper[::-1], x_lower[1:]])
    y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
    with open(filepath, "w") as f:
        f.write(f"NACA_{int(max_camber*100)}{int(camber_pos*10)}{int(thickness*100):02d}\n")
        for xc, y_val in zip(x_coords, y_coords):
            f.write(f" {xc:.6f}   {y_val:.6f}\n")
            
    print(f"Airfoil coordinates written to dat file: {filepath}")
    return x_coords, y_coords


def plot_airfoil(x, y, title="Airfoil Profile"):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, color='black', linewidth=2, label='Airfoil Surface')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.25, 0.25)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()


x, y = generate_airfoil(0.02, 0.4, 0.12, filepath="NACA_2412.dat")
plot_airfoil(x, y)

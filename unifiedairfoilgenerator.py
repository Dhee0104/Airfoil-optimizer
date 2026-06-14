import os
import subprocess
from matplotlib import lines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class unifiedairfoilgenerator:
    def __init__(self, xfoil_path=r"C:\Users\Dheem\Documents\XFOIL6.99\xfoil.exe"):
        self.xfoil_exe = xfoil_path
        self.dat_file = "temp_airfoil.dat"
        self.input_file = "xfoil_input.txt"
        self.res_file = "simulation_output.res"
    
    def generate_airfoil(self, max_camber, camber_pos, thickness, filepath="temp_airfoil.dat", num_points=100):
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
    
    def run_xfoil_automated(self, reynolds, aoa, dat_file = "temp_airfoil.dat"):
        

        # remove old files 
        if os.path.exists(self.res_file):
            os.remove(self.res_file)
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
            
        # ppar, 160 panels, operations, viscous, reynolds, iterations, polar accumulation, aoa, pacc off, quit
        commands = (
            f"load {dat_file}\n"
            f"ppar\n"
            f"n 160\n\n\n"
            f"oper\n"
            f"v\n{reynolds}\n"
            f"iter 100\n"
            f"pacc\n"
            f"{self.res_file}\n\n"
            f"alfa {aoa}\n"
            f"pacc\n"
            f"quit\n")
        
        # puit commands in a text file
        with open(self.input_file, "w") as f:
            f.write(commands)
            
        # run xfoil
        try:
            with open(self.input_file, "r") as f:
                process = subprocess.run([self.xfoil_exe],stdin=f,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=5)
        except subprocess.TimeoutExpired:
            print("Error: xfoil timed out.")
            return None



        # remove instructions
        if os.path.exists(self.input_file):
            os.remove(self.input_file)
            
        # make sure res file created
        if os.path.exists(self.res_file) and os.path.getsize(self.res_file) > 0:
            print("Success! res file was generated.")
            with open(self.res_file, "r") as f:
                lines = f.readlines()
                print("".join(lines[:20])) # Print results
                
                if len(lines) > 12:
                    data_line = lines[-1].split()
                    cl = float(data_line[1])
                    cd = float(data_line[2])
                    return cl, cd
        else:
            print("Error: res file was not generated or is empty.")

    def plot_airfoil(self, x, y, title="Airfoil Profile"):
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


airfoil_gen = unifiedairfoilgenerator()
x, y = airfoil_gen.generate_airfoil(0.02, 0.4, 0.12)
airfoil_gen.plot_airfoil(x, y, title="NACA 2412 Airfoil Profile")
cl, cd = airfoil_gen.run_xfoil_automated(500000, 0.0)
print("cl", cl)
print("cd", cd) 
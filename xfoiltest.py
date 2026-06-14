import os
import subprocess
import time

def run_xfoil_automated( reynolds, aoa, xfoil_exe = r"C:\Users\Dheem\Documents\XFOIL6.99\xfoil.exe", dat_file = "temp_airfoil.dat"):
    res_file = "simulation_output.res"
    input_file = "xfoil_input.txt"
    
    # remove old files 
    if os.path.exists(res_file):
        os.remove(res_file)
    if os.path.exists(input_file):
        os.remove(input_file)
        
    # ppar, 160 panels, operations, viscous, reynolds, iterations, polar accumulation, aoa, pacc off, quit
    commands = f"""load {dat_file}
        ppar
        n 160


        oper
        v
        {reynolds}
        iter 100
        pacc
        {res_file}
 
        alfa {aoa} 
        pacc
        quit
"""
    
    # puit commands in a text file
    with open(input_file, "w") as f:
        f.write(commands)
        
    # run xfoil
    try:
        with open(input_file, "r") as f:
            process = subprocess.run([xfoil_exe],stdin=f,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=5)
    except subprocess.TimeoutExpired:
        print("Error: xfoil timed out.")
        return None



    # remove instructions
    if os.path.exists(input_file):
        os.remove(input_file)
        
    # make sure res file created
    if os.path.exists(res_file) and os.path.getsize(res_file) > 0:
        print("Success! res file was generated.")
        with open(res_file, "r") as f:
            print("".join(f.readlines()[:20])) # Print results
    else:
        print("Error: res file was not generated or is empty.")

# Run a test calculation matching your manual parameters
run_xfoil_automated(500000, 0.0, dat_file="NACA_2412.dat")
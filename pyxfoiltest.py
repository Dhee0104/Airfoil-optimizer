from pyxfoil import Xfoil, set_workdir
import os
from pyxfoil import set_xfoilexe

set_workdir(os.getcwd()) # Set the working directory to the current directory
set_xfoilexe(r"C:\Users\Dheem\Documents\XFOIL6.99\xfoil.exe")
xfoil = Xfoil('NACA_2412')
xfoil.points_from_dat('NACA_2412.dat')
xfoil.set_ppar(100)
xfoil.run_result(alfa=5.0, mach=0.0, Re=100000.0)
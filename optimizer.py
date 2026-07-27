import numpy as np
import joblib
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

class AirfoilOptimizer:
    space = [
            Real(0, 0.05, name='max_camber'),
            Real(0.2, 0.7, name='camber_pos'),
            Real(0.08, 0.15, name='thickness')
    ]
    
    def __init__(self, target_aoa, target_reynolds):
        self.model_cl = joblib.load("airfoil_surrogate_model_cl.pkl")
        self.model_cd = joblib.load("airfoil_surrogate_model_cd.pkl")
        self.scaler = joblib.load("airfoil_scaler.pkl")
        self.target_aoa = target_aoa
        self.target_reynolds = target_reynolds
        


    def optimize_airfoil(self):
        @use_named_args(self.space)
        def get_airfoil_performance(max_camber, camber_pos, thickness):
            input_features = np.array([[max_camber, camber_pos, thickness, self.target_reynolds, self.target_aoa]])
            input_scaled = self.scaler.transform(input_features)
            predicted_cl = self.model_cl.predict(input_scaled)[0]
            predicted_cd_log = self.model_cd.predict(input_scaled)[0]
            predicted_cd = 10**predicted_cd_log
            predicted_cd = max(predicted_cd, .005)
            # print("Predicted CL:", predicted_cl, "Predicted CD:", predicted_cd)
            return -predicted_cl / predicted_cd

        @use_named_args(self.space)
        def get_airfoil_performance_cl_only(self, max_camber, camber_pos, thickness):
            input_features = np.array([[max_camber, camber_pos, thickness, self.target_reynolds, self.target_aoa]])
            input_scaled = self.scaler.transform(input_features)
            predicted_cl = self.model_cl.predict(input_scaled)[0]
            print("Predicted CL:", predicted_cl)
            return -predicted_cl


        print("optimizing")
        res = gp_minimize(func=get_airfoil_performance, dimensions=self.space, n_calls=100, n_random_starts=50, kappa =5.0, random_state=42)

        optimal_max_camber, optimal_camber_pos, optimal_thickness = res.x
        max_cl_cd = -res.fun
        # print("optimal max camber:", optimal_max_camber)
        # print("optimal camber position:", optimal_camber_pos)
        # print("optimal thickness:", optimal_thickness)
        # print("maximum CL/CD ratio:", max_cl_cd)
        return res
    
    def plot_optimization_history(self, res):
    
        trials = np.arange(1, len(res.func_vals) + 1)
        ld_ratios = -res.func_vals 
        
        configs = np.array(res.x_iters)
        cambers = configs[:, 0]
        positions = configs[:, 1]
        thicknesses = configs[:, 2]
        
        # 1. Unpack the 2x2 grid correctly as a 2D array
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex='col')
        ax1, ax2 = axs[0, 0], axs[0, 1]
        ax3, ax4 = axs[1, 0], axs[1, 1]
        
        best_idx = np.argmin(res.func_vals)
        
        # --- 1. Thickness Plot (Top Left) ---
        ax1.plot(trials, ld_ratios, 'gray', alpha=0.3, linestyle='--')
        scatter1 = ax1.scatter(trials, ld_ratios, c=thicknesses, cmap='viridis', s=40, zorder=3)
        fig.colorbar(scatter1, ax=ax1, label='Airfoil Thickness (%)')
        ax1.scatter(trials[best_idx], ld_ratios[best_idx], color='red', marker='*', s=250, 
                    edgecolor='black', label=f'Champion ({ld_ratios[best_idx]:.2f})', zorder=5) 
        ax1.set_ylabel("Efficiency ($C_L/C_D$)", fontsize=11)
        ax1.set_title("Thickness Evolution", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # --- 2. Parameter Paths Plot (Top Right) ---
        ax2.plot(trials, thicknesses, label='Thickness', color='purple', alpha=0.7)
        ax2.plot(trials, cambers * 2, label='Camber (scaled x2)', color='teal', alpha=0.7) 
        ax2.plot(trials, positions, label='Camber Position', color='orange', alpha=0.7)
        ax2.axvline(x=trials[best_idx], color='red', linestyle=':', alpha=0.8)
        ax2.set_ylabel("Geometric Values", fontsize=11)
        ax2.set_title("Parameter Convergence Paths", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # --- 3. Camber Plot (Bottom Left) ---
        ax3.plot(trials, ld_ratios, 'gray', alpha=0.3, linestyle='--')
        scatter3 = ax3.scatter(trials, ld_ratios, c=cambers, cmap='plasma', s=40, zorder=3)
        fig.colorbar(scatter3, ax=ax3, label='Camber (%)')
        ax3.scatter(trials[best_idx], ld_ratios[best_idx], color='red', marker='*', s=250, 
                    edgecolor='black', zorder=5) 
        ax3.set_xlabel("Iteration / Trial Number", fontsize=11)
        ax3.set_ylabel("Efficiency ($C_L/C_D$)", fontsize=11)
        ax3.set_title("Camber Evolution", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # --- 4. Camber Position Plot (Bottom Right) ---
        ax4.plot(trials, ld_ratios, 'gray', alpha=0.3, linestyle='--')
        scatter4 = ax4.scatter(trials, ld_ratios, c=positions, cmap='inferno', s=40, zorder=3)
        fig.colorbar(scatter4, ax=ax4, label='Camber Position (%)')
        ax4.scatter(trials[best_idx], ld_ratios[best_idx], color='red', marker='*', s=250, 
                    edgecolor='black', zorder=5) 
        ax4.set_xlabel("Iteration / Trial Number", fontsize=11)
        ax4.set_ylabel("Efficiency ($C_L/C_D$)", fontsize=11)
        ax4.set_title("Camber Position Evolution", fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("optimization_convergence_proof.png", dpi=300)
        plt.show()
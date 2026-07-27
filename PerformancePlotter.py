from optimizer import AirfoilOptimizer

optimizer = AirfoilOptimizer(target_aoa=0, target_reynolds=500000)
res = optimizer.optimize_airfoil()
optimizer.plot_optimization_history(res)
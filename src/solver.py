import numpy as np
from typing import Tuple, Callable
from src.physics import windshield_ode, WindshieldThermalModel

def runge_kutta_4(t0: float, T0: float, dt: float, t_final: float,
                  model: WindshieldThermalModel, 
                  duty_cycle_func: Callable[[float], float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    4th order Runge-Kutta solver.
    """
    t_vals = [t0]
    T_vals = [T0]
    
    t = t0
    T = T0
    
    while t < t_final:
        d = duty_cycle_func(t)
        
        k1 = dt * windshield_ode(t, T, model, d)
        k2 = dt * windshield_ode(t + dt/2, T + k1/2, model, d)
        k3 = dt * windshield_ode(t + dt/2, T + k2/2, model, d)
        k4 = dt * windshield_ode(t + dt, T + k3, model, d)
        
        T = T + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += dt
        
        t_vals.append(t)
        T_vals.append(T)
    
    return np.array(t_vals), np.array(T_vals)
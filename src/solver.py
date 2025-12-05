import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable
from src.physics import windshield_ode, WindshieldThermalModel

def solve_windshield_temperature(
    t_span: Tuple[float, float],
    T0: float,
    model: WindshieldThermalModel,
    duty_cycle_func: Callable[[float], float],
    max_step: float = 10.0,      #if max step is very small(ex:1) it takes longer to run thr program so adjust it accordingly
    **kwargs  # <--- NEW: Allow passing extra SciPy arguments
) -> Tuple[np.ndarray, np.ndarray]:
    
    def ode_wrapper(t, y):
        T_s = y[0] 
        d = duty_cycle_func(t)
        return [windshield_ode(t, T_s, model, d)]

    # Only create t_eval if not using events (events conflict with fixed t_eval sometimes)
    t_eval = None
    if 'events' not in kwargs:
        t_eval = np.arange(t_span[0], t_span[1], max_step)

    sol = solve_ivp(
        fun=ode_wrapper,
        t_span=t_span,
        y0=[T0],
        method='RK45',
        t_eval=t_eval,
        max_step=max_step,
        **kwargs  # <--- NEW: Pass them to solve_ivp
    )

    return sol.t, sol.y[0]
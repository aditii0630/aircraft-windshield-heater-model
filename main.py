import matplotlib.pyplot as plt
from typing import List

from src.config import WindshieldParams, TAXI, CLIMB, CRUISE, FlightCondition
from src.physics import WindshieldThermalModel
from src.solver import runge_kutta_4

def simulate_duty_cycle_comparison(condition: FlightCondition, 
                                   duty_cycles: List[float]):
    params = WindshieldParams()
    model = WindshieldThermalModel(params, condition)
    
    t_heating = 1000
    t_cooling = 1000
    T0 = condition.T_ambient
    
    plt.figure(figsize=(12, 8))
    
    for duty in duty_cycles:
        def duty_cycle_func(t):
            return duty if t < t_heating else 0.0
        
        t_vals, T_vals = runge_kutta_4(
            t0=0, T0=T0, dt=1.0, t_final=t_heating + t_cooling,
            model=model, duty_cycle_func=duty_cycle_func
        )
        
        T_celsius = T_vals - 273.15
        plt.plot(t_vals, T_celsius, linewidth=2, label=f'Duty Cycle = {duty}')
        
        # Stats
        T_ss = T_vals[int(t_heating)]
        print(f"Duty {duty} | Max Temp: {T_ss-273.15:.1f}°C")

    plt.axvline(x=t_heating, color='red', linestyle='--', alpha=0.5, label='Heater Off')
    plt.xlabel('Time (s)')
    plt.ylabel('Surface Temp (°C)')
    plt.title(f'Response - {condition.name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def simulate_flight_profile():
    params = WindshieldParams()
    phases = [(TAXI, 300, 0.5), (CLIMB, 600, 0.75), (CRUISE, 1800, 0.6)]
    
    t_cumulative = 0
    T_current = TAXI.T_ambient
    all_times, all_temps = [], []
    
    for condition, duration, duty in phases:
        model = WindshieldThermalModel(params, condition)
        t_vals, T_vals = runge_kutta_4(
            t0=0, T0=T_current, dt=1.0, t_final=duration,
            model=model, duty_cycle_func=lambda t: duty
        )
        all_times.extend(t_vals + t_cumulative)
        all_temps.extend(T_vals - 273.15)
        t_cumulative += t_vals[-1]
        T_current = T_vals[-1]
        
    plt.figure(figsize=(10, 6))
    plt.plot(all_times, all_temps)
    plt.title("Full Flight Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Temp (°C)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("Running Simulations...")
    simulate_duty_cycle_comparison(TAXI, [0.25, 0.5, 0.75])
    simulate_flight_profile()
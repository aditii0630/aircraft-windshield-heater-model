import matplotlib.pyplot as plt
from typing import List
import numpy as np

from src.config import WindshieldParams, TAXI, CLIMB, CRUISE, FlightCondition
from src.physics import WindshieldThermalModel
from src.solver import solve_windshield_temperature


def simulate_duty_cycle_comparison(condition: FlightCondition, duty_cycles: List[float]):
    params = WindshieldParams()
    model = WindshieldThermalModel(params, condition)
    
    t_heating = 1000
    t_cooling = 1000
    T0 = condition.T_ambient
    
    plt.figure(figsize=(12, 8))
    
    for duty in duty_cycles:
        def duty_cycle_func(t):
            return duty if t < t_heating else 0.0
        
        t_vals, T_vals = solve_windshield_temperature(
                t_span=(0, t_heating + t_cooling),
                T0=T0,
                model=model,
                duty_cycle_func=duty_cycle_func,
                max_step=10.0
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
    phases = [(TAXI, 300, 1), (CLIMB, 600, 1), (CRUISE, 1800, 1)]
    
    t_cumulative = 0
    T_current = TAXI.T_ambient
    all_times, all_temps = [], []
    
    for condition, duration, duty in phases:
        model = WindshieldThermalModel(params, condition)
        t_vals, T_vals = solve_windshield_temperature(
                t_span=(0, duration),
                T0=T_current,
                model=model,
                duty_cycle_func=lambda t: duty,
                max_step=10.0

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

def simulate_continuous_mission():
    """
    Simulates a full mission profile where the final temperature of one phase
    becomes the initial temperature of the next.
    """
    print("Simulating Continuous Mission Profile...")
    
    # 1. Define the sequence of the mission
    # Format: (Condition, Duration in seconds, Duty Cycle)
    mission_profile = [
        (TAXI,   600,  0.30),  # Taxi for 10 mins (Low power)
        (CLIMB,  900,  1.00),  # Climb for 15 mins (Max power needed due to cooling)
        (CRUISE, 1800, 0.60)   # Cruise for 30 mins (Moderate power)
    ]
    
    # Initialize storage for plotting
    all_time = np.array([])
    all_temp = np.array([])
    
    # Initial State (Cold soak on runway)
    current_temp_k = TAXI.T_ambient
    cumulative_time = 0.0
    
    # Create a params instance
    params = WindshieldParams()

    plt.figure(figsize=(14, 6))

    # 2. Iterate through phases
    for condition, duration, duty in mission_profile:
        print(f"Running {condition.name} phase...")
        print(f"  - Start Temp: {current_temp_k - 273.15:.2f}°C")
        
        # Initialize model for THIS specific flight condition
        model = WindshieldThermalModel(params, condition)
        
        # Solve for this segment
        # Note: We reset simulator time to 0 for stability, but offset it for plotting later
        t_segment, T_segment = solve_windshield_temperature(
            t_span=(0, duration),
            T0=current_temp_k,  # <--- CRITICAL: Pass the temp from previous loop
            model=model,
            duty_cycle_func=lambda t: duty,
            max_step=1.0
        )
        
        # 3. Store Data (Adjust time to be cumulative)
        all_time = np.concatenate((all_time, t_segment + cumulative_time))
        all_temp = np.concatenate((all_temp, T_segment))
        
        # 4. Update state for the next loop
        current_temp_k = T_segment[-1] # Take the very last temperature point
        cumulative_time += duration
        
        # Add a vertical line to show phase change on plot
        plt.axvline(x=cumulative_time, color='black', linestyle='--', alpha=0.3)
        plt.text(cumulative_time - (duration/2), np.min(T_segment - 273.15), 
                 f"{condition.name}\n(Duty: {duty})", 
                 ha='center', fontsize=10, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 5. Plotting
    T_celsius = all_temp - 273.15
    plt.plot(all_time, T_celsius, linewidth=2, color='navy')
    
    plt.xlabel('Mission Time (seconds)')
    plt.ylabel('Windshield Temperature (°C)')
    plt.title('Continuous Mission Profile: Taxi $\u2192$ Climb $\u2192$ Cruise')
    plt.grid(True, alpha=0.3)
    
    # Optional: Draw Green Zone (Ideal Operating Range)
    plt.fill_between(all_time, 30, 50, color='green', alpha=0.1, label='Target Range')
    plt.legend()
    
    plt.tight_layout()
    plt.show()  

    # Add this to main.py

def simulate_auto_cutoff(condition: FlightCondition, 
                         duty_cycles: List[float], 
                         cutoff_temp_c: float = 50.0):
    """
    Simulates heating until a specific temperature is reached, 
    then immediately cuts power to observe cooling.
    """
    params = WindshieldParams()
    model = WindshieldThermalModel(params, condition)
    
    target_temp_k = 273.15 + cutoff_temp_c
    T0 = condition.T_ambient
    
    # Total max time allowed (prevent infinite loop if target never reached)
    t_max = 2000 
    
    plt.figure(figsize=(12, 8))
    
    # Define the event for the solver
    # logic: return 0 when target is reached
    def target_reached_event(t, y):
        return y[0] - target_temp_k
    
    target_reached_event.terminal = True  # Stop solver when this happens
    target_reached_event.direction = 1    # Only trigger when temp is RISING
    
    for duty in duty_cycles:
        print(f"Simulating Duty Cycle: {duty}...")
        
        # --- PHASE 1: HEATING ---
        t_heat, T_heat = solve_windshield_temperature(
            t_span=(0, t_max),
            T0=T0,
            model=model,
            duty_cycle_func=lambda t: duty,
            max_step=0.5,
            events=target_reached_event # Pass the event here
        )
        
        # Check if we actually reached the target
        if T_heat[-1] < target_temp_k - 1.0:
            print(f"  Warning: Duty {duty} never reached {cutoff_temp_c}°C!")
            # Plot what we have
            plt.plot(t_heat, T_heat - 273.15, label=f'Duty {duty} (Failed to reach target)')
            continue

        # --- PHASE 2: COOLING ---
        switch_time = t_heat[-1]
        remaining_time = t_max - switch_time
        
        # Start where Phase 1 ended
        t_cool, T_cool = solve_windshield_temperature(
            t_span=(switch_time, t_max),
            T0=T_heat[-1], # Start at hot temp
            model=model,
            duty_cycle_func=lambda t: 0.0, # Heater OFF
            max_step=0.5
        )
        
        # Stitch arrays together
        t_full = np.concatenate((t_heat, t_cool))
        T_full = np.concatenate((T_heat, T_cool))
        
        # Plot
        p = plt.plot(t_full, T_full - 273.15, linewidth=2, label=f'Duty {duty}')
        
        # Mark the cutoff point
        plt.scatter([switch_time], [cutoff_temp_c], color=p[0].get_color(), marker='o')
        print(f"  Reached {cutoff_temp_c}°C in {switch_time:.1f} seconds")

    plt.axhline(y=cutoff_temp_c, color='red', linestyle='--', alpha=0.5, label='Cutoff Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Heating Rise Time & Passive Cooling\nTarget: {cutoff_temp_c}°C - Condition: {condition.name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()  

def analyze_heat_loss_breakdown(condition: FlightCondition, duty_cycle: float = 0.6):
    """
    Simulates to steady state and creates a Pie Chart of heat losses.
    Demonstrates US Patent 5,496,989 internal vs external load logic.
    """
    print(f"\nAnalyzing Energy Budget for {condition.name}...")
    
    params = WindshieldParams()
    model = WindshieldThermalModel(params, condition)
    
    # 1. Run short simulation to reach steady state
    t_vals, T_vals = solve_windshield_temperature(
        t_span=(0, 1500), # 25 mins is enough for steady state
        T0=condition.T_ambient,
        model=model,
        duty_cycle_func=lambda t: duty_cycle,
        max_step=10.0
    )
    
    # 2. Get the final temperature
    T_final = T_vals[-1]
    
    # 3. Calculate all individual heat fluxes at this temperature
    components = model.get_heat_components(T_final, duty_cycle)
    
    # 4. Print Data
    print(f"  Final Temperature: {T_final - 273.15:.2f}°C")
    print(f"  Electrical Power In: {components['Power_Input']:.1f} W")
    print("-" * 30)
    print(f"  External Convection: {components['Q_Convection_Ext']:.1f} W")
    print(f"  Evaporation:         {components['Q_Evaporation']:.1f} W")
    print(f"  Sensible (Ice/Rain): {components['Q_Sensible_Ice']:.1f} W")
    print(f"  Radiation:           {components['Q_Radiation']:.1f} W")
    print(f"  Internal Conduction: {components['Q_Conduction_Int']:.1f} W (To Cockpit)")
    
    # 5. Plot Pie Chart
    labels = ['Ext. Convection', 'Evaporation', 'Sensible (Ice)', 'Radiation', 'Int. Conduction']
    sizes = [
        components['Q_Convection_Ext'],
        components['Q_Evaporation'],
        components['Q_Sensible_Ice'],
        components['Q_Radiation'],
        components['Q_Conduction_Int']
    ]
    
    # Filter out near-zero values for cleaner chart
    plot_labels = []
    plot_sizes = []
    for l, s in zip(labels, sizes):
        if s > 1.0: # Only plot if > 1 Watt
            plot_labels.append(l)
            plot_sizes.append(s)
            
    plt.figure(figsize=(10, 6))
    
    # Pie Chart
    plt.subplot(1, 2, 1)
    plt.pie(plot_sizes, labels=plot_labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Heat Loss Distribution\n{condition.name} Phase')
    
    # Bar Chart (Input vs Loss)
    plt.subplot(1, 2, 2)
    bars = ['Power Input', 'Total Loss']
    vals = [components['Power_Input'], components['Total_Loss']]
    plt.bar(bars, vals, color=['green', 'red'])
    plt.title(f'Power Balance (Steady State: {T_final-273.15:.1f}°C)')
    plt.ylabel('Watts')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
  

if __name__ == "__main__":
    print("Running Simulations...")
    #simulate_continuous_mission()
    #simulate_duty_cycle_comparison(TAXI, [0.25, 0.5, 0.75])
    #simulate_duty_cycle_comparison(CLIMB, [0.25, 0.5, 0.75])
    #simulate_duty_cycle_comparison(CRUISE, [0.25, 0.5, 0.75])
    #simulate_flight_profile()

   
    
    print("\n--- SCENARIO 4: Auto-Cutoff Experiment ---")
    
    
    
    #simulate_auto_cutoff(TAXI, duty_cycles=[0.4, 0.6, 0.8, 1.0])
    #simulate_auto_cutoff(CLIMB, duty_cycles=[0.4, 0.6, 0.8, 1.0])
    #simulate_auto_cutoff(CRUISE, duty_cycles=[0.4, 0.6, 0.8, 1.0])

    #from src.config import CRUISE
    analyze_heat_loss_breakdown(TAXI, duty_cycle=1.0)

    
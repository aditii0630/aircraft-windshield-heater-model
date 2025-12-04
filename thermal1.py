"""
Aircraft Windshield Heater Controller - Thermal Model

Control-oriented thermal simulation of electrically heated aircraft windshield
using Runge-Kutta 4th order method for unsteady state heat balance.

Based on:
- AFWAL-TR-80-3003 (Guidelines for Aircraft Windshield Systems)
- NACA Technical Note 1434 (Ice Formation on Aircraft)
- US Patent 5,496,989 (Windshield Heating Control)

Author: Aditi Shankar
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

# ========================================
# PHYSICAL CONSTANTS
# ========================================
GAMMA = 1.4                          # Ratio of specific heats for air
RECOVERY_FACTOR = 0.9                # Recovery factor near stagnation
L_V = 2.26e6                         # Latent heat of vaporization (J/kg)
SIGMA = 5.670374419e-8               # Stefan-Boltzmann constant (W/m²K⁴)
C_W = 4180                           # Specific heat of water (J/kg·K)
R_V = 461.5                          # Specific gas constant for water vapor (J/kg·K)

# ========================================
# FLIGHT CONDITION DATACLASS
# ========================================
@dataclass
class FlightCondition:
    """Defines environmental and operational parameters for each flight phase"""
    name: str
    velocity: float              # Aircraft velocity (m/s)
    T_ambient: float            # Ambient air temperature (K)
    T_cockpit: float            # Cockpit temperature (K)
    altitude: float             # Altitude (m)
    pressure: float             # Static pressure (Pa)
    LWC: float                  # Liquid Water Content (kg/m³)
    droplet_temp: float         # Droplet temperature (K)
    collection_efficiency: float # Collection efficiency η (0-1)
        

# Define three flight phases from paper (Table 1)
TAXI = FlightCondition(
    name="Taxi",
    velocity=7.5,               # Average of 5-10 m/s
    T_ambient=288.15,           # 15°C
    T_cockpit=293.15,           # 20°C
    altitude=0,
    pressure=101325,
    LWC=0.0005,                 # Light icing condition
    droplet_temp=273.15,        # 0°C
    collection_efficiency=0.8
)

CLIMB = FlightCondition(
    name="Climb",
    velocity=125.0,             # Average of 100-150 m/s
    T_ambient=263.15,           # -10°C
    T_cockpit=293.15,           # 20°C
    altitude=5000,
    pressure=54000,
    LWC=0.001,                  # Moderate icing
    droplet_temp=263.15,        # -10°C
    collection_efficiency=0.85
)

CRUISE = FlightCondition(
    name="Cruise",
    velocity=250.0,             # Average of 225-275 m/s
    T_ambient=218.15,           # -55°C
    T_cockpit=293.15,           # 20°C
    altitude=11000,
    pressure=22700,
    LWC=0.0002,                 # Light icing at altitude
    droplet_temp=218.15,        # -55°C
    collection_efficiency=0.9
)


# WINDSHIELD PARAMETERS

class WindshieldParams:
    """Physical parameters of the windshield system"""
    def __init__(self):
        # Geometry
        self.area = 0.8                    # Windshield area (m²)
        self.length = 1.2                  # Characteristic length (m)
        self.A_proj = 0.6                  # Forward-projected area (m²)
        
        # Thermal properties
        self.mass = 25.0                   # Total mass (kg)
        self.c_p = 840.0                   # Specific heat capacity (J/kg·K)
        self.emissivity = 0.9              # Surface emissivity
        
        # Outer ply (for conduction calculation)
        self.t_ply = 0.003                 # Outer ply thickness (m)
        self.k_ply = 1.0                   # Thermal conductivity (W/m·K)
        
        # Electrical system
        self.voltage = 28.0                # DC bus voltage (V)
        self.resistance = 0.2              # Heater resistance (Ω)


def air_properties(T: float, P: float = 101325) -> Tuple[float, float, float, float]:
    """
    Calculate air properties at given temperature and pressure
    
    Args:
        T: Temperature (K)
        P: Pressure (Pa)
    
    Returns:
        rho: Density (kg/m³)
        mu: Dynamic viscosity (kg/m·s)
        k: Thermal conductivity (W/m·K)
        Pr: Prandtl number
    """
    # Sutherland's formula for viscosity
    T_ref = 273.15
    mu_ref = 1.716e-5
    S = 110.4
    mu = mu_ref * (T/T_ref)**1.5 * (T_ref + S)/(T + S)
    
    # Ideal gas law for density
    R = 287.05  # Specific gas constant for air (J/kg·K)
    rho = P / (R * T)
    
    # Thermal conductivity (approximation)
    k = 0.024 + 7.5e-5 * (T - 273.15)
    
    # Prandtl number (approximately constant for air)
    Pr = 0.71
    
    return rho, mu, k, Pr


# THERMAL MODEL EQUATIONS

class WindshieldThermalModel:
    
    
    def __init__(self, params: WindshieldParams, condition: FlightCondition):
        self.params = params
        self.condition = condition
        
    def recovery_temperature(self) -> float:
        """
        Calculate recovery temperature (Equation 3)
        T_r = T_∞ * (1 + r*(γ-1)/2 * M²)
        """
        # Calculate Mach number
        a = np.sqrt(GAMMA * 287.05 * self.condition.T_ambient)  # Speed of sound
        M = self.condition.velocity / a
        
        T_r = self.condition.T_ambient * (
            1 + RECOVERY_FACTOR * (GAMMA - 1) / 2 * M**2
        )
        return T_r
    
    def convection_coefficient(self, T_r: float) -> float:
        """
        Calculate convective heat transfer coefficient (Equation 5)
        h = 1.15 * T_r^0.3 * V_a * ρ^0.8 * 0.51/s^0.2
        
        Note: s = characteristic length = 1.2m
        """
        rho, _, _, _ = air_properties(T_r, self.condition.pressure)
        V_a = self.condition.velocity
        s = self.params.length
        
        h = 1.15 * (T_r**0.3) * V_a * (rho**0.8) * 0.51 / (s**0.2)
        return h
    
    def Q_convection(self, T_s: float, h: float, T_r: float) -> float:
        """
        Convective heat loss (Equation 6)
        Q_conv = h * A * (T_s - T_r)
        """
        return h * self.params.area * (T_s - T_r)
    
    def Q_evaporation(self, T_s: float, h: float) -> float:
        """
        Evaporative heat loss (Equation 9 - simplified)
        Q_evap = 2.91 * L_x * h * (e_sw - e_w) / P_static
        
        Simplified using Magnus formula for saturation vapor pressure
        """
        # Saturation vapor pressure at surface (Magnus formula)
        T_s_C = T_s - 273.15
        e_sw = 611.2 * np.exp(17.67 * T_s_C / (T_s_C + 243.5))
        
        # Vapor pressure in ambient (assume 80% relative humidity)
        T_amb_C = self.condition.T_ambient - 273.15
        e_w = 0.8 * 611.2 * np.exp(17.67 * T_amb_C / (T_amb_C + 243.5))
        
        # Characteristic length for mass transfer
        L_x = self.params.length
        
        Q_evap = 2.91 * L_x * h * (e_sw - e_w) / self.condition.pressure
        
        # Ensure non-negative
        return max(0, Q_evap)
    
    def Q_sensible(self, T_s: float) -> float:
        """
        Sensible heat loss from droplet impingement (Equation 10, 11)
        Q_sensible = m_imp * c_w * (T_s - T_w)
        m_imp = A_proj * LWC * V * η
        """
        m_imp = (self.params.A_proj * 
                 self.condition.LWC * 
                 self.condition.velocity * 
                 self.condition.collection_efficiency)
        
        Q_sens = m_imp * C_W * (T_s - self.condition.droplet_temp)
        return max(0, Q_sens)  # Only loss if surface is warmer
    
    def Q_radiation(self, T_s: float) -> float:
        """
        Radiative heat loss (Equation 12)
        Q_rad = ε * σ * A * (T_s⁴ - T_∞⁴)
        """
        Q_rad = (self.params.emissivity * SIGMA * self.params.area * 
                 (T_s**4 - self.condition.T_ambient**4))
        return Q_rad
    
    def total_heat_loss(self, T_s: float, h: float, T_r: float) -> float:
        """
        Total heat loss (Equation 2)
        Q_loss = Q_conv + Q_evap + Q_sensible + Q_rad
        """
        Q_conv = self.Q_convection(T_s, h, T_r)
        Q_evap = self.Q_evaporation(T_s, h)
        Q_sens = self.Q_sensible(T_s)
        Q_rad = self.Q_radiation(T_s)
        
        return Q_conv + Q_evap + Q_sens + Q_rad
    
    def electrical_power(self, duty_cycle: float) -> float:
        """
        Electrical heating power (Equation 15)
        P_elec = d * V² / R
        """
        return duty_cycle * (self.params.voltage**2) / self.params.resistance


# ODE FUNCTION FOR RK4 SOLVER

def windshield_ode(t: float, T_s: float, model: WindshieldThermalModel, 
                   duty_cycle: float) -> float:
    """
    Unsteady state heat balance (Equation 1)
    
    dT_s/dt = (Q_in - Q_loss) / (m * c)
    
    Args:
        t: Time (s)
        T_s: Surface temperature (K)
        model: Thermal model instance
        duty_cycle: PWM duty cycle (0-1)
    
    Returns:
        dT_s/dt: Temperature rate of change (K/s)
    """
    # Calculate recovery temperature
    T_r = model.recovery_temperature()
    
    # Calculate convection coefficient
    h = model.convection_coefficient(T_r)
    
    # Heat input from electrical heater
    Q_in = model.electrical_power(duty_cycle)
    
    # Total heat loss
    Q_loss = model.total_heat_loss(T_s, h, T_r)
    
    # Temperature rate of change
    dT_dt = (Q_in - Q_loss) / (model.params.mass * model.params.c_p)
    
    return dT_dt


# RUNGE-KUTTA 4TH ORDER SOLVER

def runge_kutta_4(t0: float, T0: float, dt: float, t_final: float,
                  model: WindshieldThermalModel, 
                  duty_cycle_func) -> Tuple[np.ndarray, np.ndarray]:
    """
    4th order Runge-Kutta solver for windshield temperature
    
    Args:
        t0: Initial time (s)
        T0: Initial temperature (K)
        dt: Time step (s)
        t_final: Final simulation time (s)
        model: Thermal model instance
        duty_cycle_func: Function that returns duty cycle at time t
    
    Returns:
        t_vals: Array of time values
        T_vals: Array of temperature values
    """
    t_vals = [t0]
    T_vals = [T0]
    
    t = t0
    T = T0
    
    while t < t_final:
        # Get current duty cycle
        d = duty_cycle_func(t)
        
        # RK4 coefficients
        k1 = dt * windshield_ode(t, T, model, d)
        k2 = dt * windshield_ode(t + dt/2, T + k1/2, model, d)
        k3 = dt * windshield_ode(t + dt/2, T + k2/2, model, d)
        k4 = dt * windshield_ode(t + dt, T + k3, model, d)
        
        # Update temperature
        T = T + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t += dt
        
        t_vals.append(t)
        T_vals.append(T)
    
    return np.array(t_vals), np.array(T_vals)


# SIMULATION SCENARIOS

def simulate_duty_cycle_comparison(condition: FlightCondition, 
                                   duty_cycles: List[float] = [0.25, 0.5, 0.75]):
    """
    Replicate Figure 2 from paper: Compare different PWM duty cycles
    with heating phase followed by cooling phase
    """
    params = WindshieldParams()
    model = WindshieldThermalModel(params, condition)
    
    # Simulation parameters
    t_heating = 1000    # Heating duration (s)
    t_cooling = 1000    # Cooling duration (s)
    dt = 1.0            # Time step (s)
    T0 = condition.T_ambient  # Start at ambient temperature
    
    plt.figure(figsize=(12, 8))
    
    for duty in duty_cycles:
        # Define duty cycle function (heating then cooling)
        def duty_cycle_func(t):
            return duty if t < t_heating else 0.0
        
        # Run simulation
        t_vals, T_vals = runge_kutta_4(
            t0=0, T0=T0, dt=dt, t_final=t_heating + t_cooling,
            model=model, duty_cycle_func=duty_cycle_func
        )
        
        # Convert to Celsius for plotting
        T_celsius = T_vals - 273.15
        
        # Plot results
        plt.plot(t_vals, T_celsius, linewidth=2, label=f'Duty Cycle = {duty}')
        
        # Print steady-state info
        idx_ss = int(t_heating / dt)
        T_ss = T_vals[idx_ss]
        print(f"\nDuty Cycle {duty}:")
        print(f"  Steady-state temperature: {T_ss-273.15:.1f}°C ({T_ss:.1f}K)")
        print(f"  Temperature rise: {T_ss - T0:.1f}K")
        
        # Calculate steady-state power balance
        T_r = model.recovery_temperature()
        h = model.convection_coefficient(T_r)
        Q_loss = model.total_heat_loss(T_ss, h, T_r)
        Q_in = model.electrical_power(duty)
        print(f"  Power input: {Q_in:.1f}W")
        print(f"  Power loss: {Q_loss:.1f}W")
    
    plt.axvline(x=t_heating, color='red', linestyle='--', alpha=0.5, 
                label='Heater Off')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Windshield Surface Temperature (°C)', fontsize=12)
    plt.title(f'Windshield Thermal Response - {condition.name} Phase\n'
              f'V = {condition.velocity:.1f} m/s, T_amb = {condition.T_ambient-273.15:.1f}°C',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def simulate_flight_profile():
    """
    Simulate complete flight profile: Taxi → Climb → Cruise
    """
    params = WindshieldParams()
    
    # Flight phases with durations
    phases = [
        (TAXI, 300, 0.5),      # Taxi: 5 min at 50% duty
        (CLIMB, 600, 0.75),    # Climb: 10 min at 75% duty
        (CRUISE, 1800, 0.6)    # Cruise: 30 min at 60% duty
    ]
    
    plt.figure(figsize=(14, 10))
    
    # Temperature subplot
    plt.subplot(3, 1, 1)
    
    t_cumulative = 0
    T_current = TAXI.T_ambient
    all_times = []
    all_temps = []
    
    for i, (condition, duration, duty) in enumerate(phases):
        model = WindshieldThermalModel(params, condition)
        
        def duty_cycle_func(t):
            return duty
        
        t_vals, T_vals = runge_kutta_4(
            t0=0, T0=T_current, dt=1.0, t_final=duration,
            model=model, duty_cycle_func=duty_cycle_func
        )
        
        # Adjust time to be cumulative
        t_vals_adj = t_vals + t_cumulative
        all_times.extend(t_vals_adj)
        all_temps.extend(T_vals - 273.15)
        
        # Update for next phase
        t_cumulative = t_vals_adj[-1]
        T_current = T_vals[-1]
    
    plt.plot(all_times, all_temps, linewidth=2, color='darkblue')
    plt.axvline(x=300, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=900, color='gray', linestyle='--', alpha=0.5)
    plt.text(150, max(all_temps)*0.9, 'TAXI', ha='center', fontsize=10, fontweight='bold')
    plt.text(600, max(all_temps)*0.9, 'CLIMB', ha='center', fontsize=10, fontweight='bold')
    plt.text(1650, max(all_temps)*0.9, 'CRUISE', ha='center', fontsize=10, fontweight='bold')
    plt.ylabel('Temperature (°C)', fontsize=11)
    plt.title('Complete Flight Profile Simulation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Duty cycle subplot
    plt.subplot(3, 1, 2)
    duty_times = [0, 300, 300, 900, 900, 2700]
    duty_vals = [0.5, 0.5, 0.75, 0.75, 0.6, 0.6]
    plt.step(duty_times, duty_vals, where='post', linewidth=2, color='orange')
    plt.ylabel('PWM Duty Cycle', fontsize=11)
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    # Velocity subplot
    plt.subplot(3, 1, 3)
    vel_times = [0, 300, 900, 2700]
    vel_vals = [TAXI.velocity, CLIMB.velocity, CRUISE.velocity, CRUISE.velocity]
    plt.plot(vel_times, vel_vals, linewidth=2, color='green', marker='o')
    plt.xlabel('Time (s)', fontsize=11)
    plt.ylabel('Velocity (m/s)', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# MAIN EXECUTION

if __name__ == "__main__":
    print("=" * 70)
    print("AIRCRAFT WINDSHIELD HEATER CONTROLLER - THERMAL SIMULATION")
    print("=" * 70)
    print("\nBased on control-oriented lumped-parameter thermal model")
    print("Solving unsteady heat balance using Runge-Kutta 4th order method\n")
    
    # Scenario 1: Replicate Figure 2 from paper (Ground conditions)
    print("\n--- SCENARIO 1: Ground Operations (Replicating Paper Figure 2) ---")
    simulate_duty_cycle_comparison(TAXI, duty_cycles=[0.25, 0.5, 0.75])
    
    # Scenario 2: Climb phase comparison
    print("\n--- SCENARIO 2: Climb Phase Comparison ---")
    simulate_duty_cycle_comparison(CLIMB, duty_cycles=[0.5, 0.75, 1.0])
    
    # Scenario 3: Complete flight profile
    print("\n--- SCENARIO 3: Complete Flight Profile ---")
    simulate_flight_profile()
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
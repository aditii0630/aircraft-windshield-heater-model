import numpy as np
from typing import Tuple
from src.config import (
    FlightCondition, WindshieldParams, GAMMA, RECOVERY_FACTOR, 
    C_W, SIGMA
)

def air_properties(T: float, P: float = 101325) -> Tuple[float, float, float, float]:
    """Calculate air properties: rho, mu, k, Pr"""
    T_ref = 273.15
    mu_ref = 1.716e-5
    S = 110.4
    mu = mu_ref * (T/T_ref)**1.5 * (T_ref + S)/(T + S)
    
    R = 287.05
    rho = P / (R * T)
    k = 0.024 + 7.5e-5 * (T - 273.15)
    Pr = 0.71
    return rho, mu, k, Pr

class WindshieldThermalModel:
    def __init__(self, params: WindshieldParams, condition: FlightCondition):
        self.params = params
        self.condition = condition
        
    def recovery_temperature(self) -> float:
        a = np.sqrt(GAMMA * 287.05 * self.condition.T_ambient)
        M = self.condition.velocity / a
        return self.condition.T_ambient * (1 + RECOVERY_FACTOR * (GAMMA - 1) / 2 * M**2)
    
    def convection_coefficient(self, T_r: float) -> float:
        rho, _, _, _ = air_properties(T_r, self.condition.pressure)
        h = 1.15 * (T_r**0.3) * ((self.condition.velocity * rho)**0.8) * 0.51 / (self.params.length**0.2)
        return h
    
    def Q_convection(self, T_s: float, h: float, T_r: float) -> float:
        return h * self.params.area * (T_s - T_r)
    
    def Q_evaporation(self, T_s: float, h: float) -> float:
        T_s_C = T_s - 273.15
        e_sw = 611.2 * np.exp(17.67 * T_s_C / (T_s_C + 243.5))
        
        T_amb_C = self.condition.T_ambient - 273.15
        e_w = 0.8 * 611.2 * np.exp(17.67 * T_amb_C / (T_amb_C + 243.5))
        
        Q_evap = 2.91 * self.params.length * h * (e_sw - e_w) / self.condition.pressure
        return max(0, Q_evap)
    
    def Q_sensible(self, T_s: float) -> float:
        m_imp = (self.params.A_proj * self.condition.LWC * 
                 self.condition.velocity * self.condition.collection_efficiency)
        return max(0, m_imp * C_W * (T_s - self.condition.droplet_temp))
    
    def Q_radiation(self, T_s: float) -> float:
        return (self.params.emissivity * SIGMA * self.params.area * 
                (T_s**4 - self.condition.T_ambient**4))
    
    def Q_inner_loss(self, T_s: float) -> float:
        """
        Calculate Internal Heat Loss (qi) to the cockpit.
        Based on US Patent 5,496,989 (Eq 9 and Eq 11).
        
        q_i = h_i * A * (T_s - T_cockpit)
        """
        # Ensure we don't calculate negative loss (heat gain) if cockpit is hotter 
        # than windshield (rare in de-icing, but possible on ground).
        # We allow negative return here to represent heat GAIN from cockpit.
        return self.params.h_internal * self.params.area * (T_s - self.condition.T_cockpit)

    
    def total_heat_loss(self, T_s: float, h: float, T_r: float) -> float:
        """
        Total heat loss = External Losses + Internal Loss
        """
        Q_ext_conv = self.Q_convection(T_s, h, T_r)
        Q_ext_evap = self.Q_evaporation(T_s, h)
        Q_ext_sens = self.Q_sensible(T_s)
        Q_ext_rad  = self.Q_radiation(T_s)
        
        # Add the new internal loss
        Q_int = self.Q_inner_loss(T_s)
        return Q_ext_conv + Q_ext_evap + Q_ext_sens + Q_ext_rad + Q_int
    
    def get_heat_components(self, T_s: float, duty_cycle: float) -> dict:
        """
        Diagnostic function to return all heat components separately.
        Useful for main.py analysis graphs.
        """
        T_r = self.recovery_temperature()
        h = self.convection_coefficient(T_r)
        
        return {
            "Power_Input": self.electrical_power(duty_cycle),
            "Q_Convection_Ext": self.Q_convection(T_s, h, T_r),
            "Q_Evaporation": self.Q_evaporation(T_s, h),
            "Q_Sensible_Ice": self.Q_sensible(T_s),
            "Q_Radiation": self.Q_radiation(T_s),
            "Q_Conduction_Int": self.Q_inner_loss(T_s), # The new internal loss
            "Total_Loss": self.total_heat_loss(T_s, h, T_r),
            "Net_Flux": self.electrical_power(duty_cycle) - self.total_heat_loss(T_s, h, T_r)
        }
    
    def electrical_power(self, duty_cycle: float) -> float:
        return duty_cycle * (self.params.voltage**2) / self.params.resistance

def windshield_ode(t: float, T_s: float, model: WindshieldThermalModel, 
                   duty_cycle: float) -> float:
    """dT_s/dt = (Q_in - Q_loss) / (m * c)"""
    T_r = model.recovery_temperature()
    h = model.convection_coefficient(T_r)
    Q_in = model.electrical_power(duty_cycle)
    Q_loss = model.total_heat_loss(T_s, h, T_r)
    return (Q_in - Q_loss) / (model.params.mass * model.params.c_p)
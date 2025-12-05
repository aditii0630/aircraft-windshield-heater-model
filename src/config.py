from dataclasses import dataclass

# ========================================
# PHYSICAL CONSTANTS
# ========================================
GAMMA = 1.4
RECOVERY_FACTOR = 0.9
L_V = 2.26e6
SIGMA = 5.670374419e-8
C_W = 4180
R_V = 461.5

# ========================================
# DATA STRUCTURES
# ========================================
@dataclass
class FlightCondition:
    """Defines environmental and operational parameters for each flight phase"""
    name: str
    velocity: float              # m/s
    T_ambient: float             # K
    T_cockpit: float             # K
    altitude: float              # m
    pressure: float              # Pa
    LWC: float                   # kg/m³
    droplet_temp: float          # K
    collection_efficiency: float # 0-1

class WindshieldParams:
    """Physical parameters of the windshield system"""
    def __init__(self):
        # Geometry
        self.area = 0.8       # m²
        self.length = 1.2     # m
        self.A_proj = 0.6     # m²
        
        # Thermal properties
        self.mass = 6.5
        self.c_p = 840.0
        self.emissivity = 0.9
        
        # Electrical system
        self.voltage = 28.0
        self.resistance = 0.4

        self.h_internal= 12.0  # W/m²K 

# ========================================
# PRESETS
# ========================================
TAXI = FlightCondition(
    name="Taxi", velocity=7.5, T_ambient=288.15, T_cockpit=293.15,
    altitude=0, pressure=101325, LWC=0.0005, droplet_temp=273.15,
    collection_efficiency=0.8
)

CLIMB = FlightCondition(
    name="Climb", velocity=60.0, T_ambient=263.15, T_cockpit=293.15,
    altitude=3000, pressure=90000, LWC=0.001, droplet_temp=263.15,
    collection_efficiency=0.85
)

CRUISE = FlightCondition(
    name="Cruise", velocity=150.0, T_ambient=218.15, T_cockpit=293.15,
    altitude=8839, pressure=30100, LWC=0.0002, droplet_temp=218.15,
    collection_efficiency=0.9
)
# Aircraft Windshield Heater Thermal Model

Real-time thermal simulation of an electrically heated aircraft windshield anti-icing system using numerical methods and control theory.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Overview

This project implements a control-oriented thermal model for aircraft windshield heating systems, simulating heat transfer through multiple mechanisms (convection, radiation, evaporation, and droplet impingement) across different flight phases.

### Key Features
- ✅ Multi-physics heat transfer modeling
- ✅ PWM duty cycle control simulation
- ✅ Runge-Kutta 4th order numerical integration
- ✅ Complete flight profile simulation (Taxi/Climb/Cruise)
- ✅ Parametric studies for different operating conditions

## 🔬 Technical Foundation

Based on aerospace industry standards and research:
- **MIL-E-38453A** environmental requirements
- **AFWAL-TR-80-3003** windshield design guidelines  
- **NACA TN-1434** ice formation correlations
- **US Patent 5,496,989** intelligent windshield heating

## 📊 Results

### Duty Cycle Comparison (Ground Operations)
![Duty Cycle Comparison](results/duty_cycle_comparison.png)

*Temperature response for 0.25, 0.50, and 0.75 duty cycles during 1000s heating + 1000s cooling*

### Complete Flight Profile
![Flight Profile](results/flight_profile.png)

*Windshield temperature evolution through Taxi → Climb → Cruise phases*

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/windshield-heater-thermal-model.git
cd windshield-heater-thermal-model

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from windshield_thermal_model import WindshieldThermalModel, CRUISE, simulate_duty_cycle_comparison

# Run cruise phase simulation with different duty cycles
simulate_duty_cycle_comparison(CRUISE, duty_cycles=[0.5, 0.75, 1.0])
```

### Run All Simulations
```bash
python windshield_thermal_model.py
```

## 📐 Mathematical Model

### Governing Equation (Unsteady Heat Balance)
```
Q_in - Q_loss = m·c·(dT_s/dt)
```

Where:
- `Q_in = d·V²/R` (PWM-controlled electrical heating)
- `Q_loss = Q_conv + Q_evap + Q_sensible + Q_rad`

### Heat Transfer Mechanisms

1. **Convection** (high-speed airflow)
```
   Q_conv = h·A·(T_s - T_r)
   h = 1.15·T_r^0.3·V_a·ρ^0.8·0.51/s^0.2
```

2. **Evaporative Cooling** (water on surface)
```
   Q_evap = 2.91·L_x·h·(e_sw - e_w)/P_static
```

3. **Sensible Heat** (droplet impingement)
```
   Q_sensible = ṁ_imp·c_w·(T_s - T_w)
```

4. **Radiation** (thermal emission)
```
   Q_rad = ε·σ·A·(T_s⁴ - T_∞⁴)
```

## 🎓 Project Structure
```
├── windshield_thermal_model.py    # Main simulation code
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── docs/
│   └── wshcpaper_final.pdf        # Technical paper
├── examples/
│   └── simulation_demo.ipynb      # Interactive Jupyter notebook
└── results/
    ├── duty_cycle_comparison.png
    └── flight_profile.png
```

## 📈 Use Cases

- **Control System Design**: Test PID/adaptive controllers before hardware implementation
- **Power Optimization**: Minimize energy consumption while maintaining de-icing
- **Safety Analysis**: Verify temperature limits under extreme conditions
- **Embedded Systems**: Computationally efficient for STM32/embedded targets

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and plotting
- **RK4 Method**: 4th-order Runge-Kutta for ODE integration

## 🔮 Future Enhancements

- [ ] PID controller implementation
- [ ] Model predictive control (MPC)
- [ ] CAN protocol integration (STM32)
- [ ] Real-time hardware-in-the-loop testing
- [ ] Multi-zone heating optimization
- [ ] Validation against flight test data

## 📝 Research Paper

Full technical details available in [`docs/wshcpaper_final.pdf`](docs/wshcpaper_final.pdf)

## 👨‍💻 Author

**[Your Name]**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🌐 Portfolio: [yourwebsite.com](https://yourwebsite.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Douglas Aircraft Company for foundational windshield thermal modeling research
- AFWAL Flight Dynamics Laboratory for design guidelines
- NACA for ice formation heat transfer correlations

---


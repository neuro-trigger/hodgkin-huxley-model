# hodgkin-huxley-model

## 📌 Overview

This project presents a computational implementation, analysis, and qualitative evaluation of the **Hodgkin-Huxley model** for neuronal action potentials. Developed as part of a Bioinformatics coursework at the *National University of Colombia*.

The simulation solves the system of non-linear differential equations that describe the ionic mechanisms underlying the initiation and propagation of action potentials in the squid giant axon, adapted for pyramidal cortical neurons.

### Key Objectives
- **Formulate** the biophysical mathematical model.
- **Implement** the model in Python using the **Runge-Kutta 4 (RK4)** numerical integration method.
- **Analyze** the neuronal response to varying input currents.
- **Compare** simulation results with biological data from mammalian and *Aplysia* neurons.

## ⚙️ Mathematical Model

The model treats the cell membrane as an electrical circuit consisting of a capacitor and three parallel branches representing ionic currents: Sodium ($Na^+$), Potassium ($K^+$), and a Leak current ($L$).

The total current $I(t)$ is described by:

$$C_m \frac{dV}{dt} = I_{inj}(t) - \bar{g}_{Na} m^3 h (V - E_{Na}) - \bar{g}_K n^4 (V - E_K) - g_L (V - E_L)$$

Where $m$, $n$, and $h$ are the gating variables evolving according to:

$$\frac{dx}{dt} = \alpha_x(V)(1-x) - \beta_x(V)x, \quad x \in \{m, n, h\}$$

## 🚀 Features

* **Numerical Solver:** Custom implementation of the 4th-order Runge-Kutta method (RK4) for high-precision integration.
* **Dynamic Stimulation:** Supports constant and time-varying input currents.
* **Visualization:** Generates plots for Membrane Potential ($V_m$), Gating Variables ($m, n, h$), and Ionic Currents.
* **Modularity:** Code structured to allow easy parameter modification for different neuron types.

## 📊 Results

### 1. Action Potential Generation
The simulation successfully reproduces the characteristic spike shape, including the depolarization phase, repolarization, and hyperpolarization overshoot.

### 2. Frequency vs. Current
Increasing the injected current results in a higher firing frequency, consistent with the linear frequency-current relationship observed in Type I neurons, until a depolarization block occurs at high currents.

### 3. Limitations
The study concluded that while the model is excellent for fast-spiking neurons, it requires modifications (additional channels like CaL, CaR) to simulate complex behaviors such as the **Plateau Potentials** observed in *Aplysia californica*.

## 💻 Installation & Usage

### Prerequisites
* Python 3.x
* NumPy
* Matplotlib

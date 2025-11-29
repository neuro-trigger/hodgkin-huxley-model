"""
Hodgkin-Huxley neuronal model.
Implements membrane equations and gating variables using RK4 integration.
Based on Hodgkin & Huxley (1952).

Dependencies:
- numpy
- matplotlib

Author: Carlos Andres Rios Rojas
"""

import numpy as np
import matplotlib.pyplot as plot

"""
Parameter dictionary example. This dictionary contains the original
Hodgkin-Huxley parameters.
"""
HH_PARAMETERS = {
    'E_Na' : 55.0,  # Nernst potential for Na channel (mV)
    'g_Na' : 40,    # Maximum conductance for Na channel (mS/cm^2)
    'E_K' : -77.0,  # Nernst potential for K channel (mV)
    'g_K' : 35,     # Maximum conductance for K channel (mS/cm^2)
    'E_L' : -65.0,  # Nernst potential for leak channel (mV)
    'g_L' : 0.3,    # Maximum conductance for leak channel (mS/cm^2)
    'C' : 1         # Membrane capacitancy (pico F/cm^2)
}

"""
Differential functions.
"""
def membrane_potential_differential(u, m, n, h, p, i):
   na_current = p['g_Na']*(m**3)*h*(u - p['E_Na'])
   k_current = p['g_K']*(n**4)*(u - p['E_K'])
   l_current = p['g_L']*(u - p['E_L'])

   return ( i - (na_current + k_current + l_current) ) / p['C']

def gate_m_differential(u, m):
    alpha = ( 0.182 * (u + 35) ) / ( 1 - np.exp( -(u + 35)/9 ) )
    beta = ( -0.124 * (u + 35) ) / ( 1 - np.exp( (u + 35)/9 ) )

    return alpha*(1 - m) - beta*m

def gate_n_differential(u, n):
    alpha = ( 0.02 * (u - 25) ) / ( 1 - np.exp( -(u - 25)/9 ) )
    beta = ( -0.002 * (u - 25) ) / ( 1 - np.exp( (u - 25)/9 ) )

    return alpha*(1 - n) - beta*n

def gate_h_differential(u, h):
    alpha = 0.25 * np.exp( -(u + 90)/12 )
    beta = 0.25 * np.exp( (u + 62)/6 ) / np.exp( (u + 90)/12 )

    return alpha*(1 - h) - beta*h

"""
RK4 function.
"""
def rk4_step_(u, m, n, h, p, input_current, t, dt):
    i = input_current(t)
    uk1 = membrane_potential_differential(u, m, n, h, p, i)
    mk1 = gate_m_differential(u, m)
    nk1 = gate_n_differential(u, n)
    hk1 = gate_h_differential(u, h)

    i = input_current(t + dt/2)
    uk2 = membrane_potential_differential(u + uk1*dt/2,
                                          m + mk1*dt/2,
                                          n + nk1*dt/2,
                                          h + hk1*dt/2,
                                          p,
                                          i)
    mk2 = gate_m_differential(u + uk1*dt/2, m + mk1*dt/2)
    nk2 = gate_n_differential(u + uk1*dt/2, n + nk1*dt/2)
    hk2 = gate_h_differential(u + uk1*dt/2, h + hk1*dt/2)

    uk3 = membrane_potential_differential(u + uk2*dt/2,
                                          m + mk2*dt/2,
                                          n + nk2*dt/2,
                                          h + hk2*dt/2,
                                          p,
                                          i)
    mk3 = gate_m_differential(u + uk2*dt/2, m + mk2*dt/2)
    nk3 = gate_n_differential(u + uk2*dt/2, n + nk2*dt/2)
    hk3 = gate_h_differential(u + uk2*dt/2, h + hk2*dt/2)

    i = input_current(t + dt)
    uk4 = membrane_potential_differential(u + uk3*dt,
                                          m + mk3*dt,
                                          n + nk3*dt,
                                          h + hk3*dt,
                                          p,
                                          i)
    mk4 = gate_m_differential(u + uk3*dt, m + mk3*dt)
    nk4 = gate_n_differential(u + uk3*dt, n + nk3*dt)
    hk4 = gate_h_differential(u + uk3*dt, h + hk3*dt)

    f_u = u + dt/6 * (uk1 + 2*uk2 + 2*uk3 + uk4)
    f_m = m + dt/6 * (mk1 + 2*mk2 + 2*mk3 + mk4)
    f_n = n + dt/6 * (nk1 + 2*nk2 + 2*nk3 + nk4)
    f_h = h + dt/6 * (hk1 + 2*hk2 + 2*hk3 + hk4)

    return f_u, f_m, f_n, f_h

"""
Plot functions.
"""
def plot_gate_variables(m, n, h, t):
    plot.figure(figsize=(8, 5))

    plot.plot(t, n, label='n(t)')
    plot.plot(t, m, label='m(t)')
    plot.plot(t, h, label='h(t)')

    plot.xlabel('Time (ms)')
    plot.ylabel('Gate Value')
    plot.title('Gate Variable Evolution')
    plot.legend()
    plot.grid(True)

    plot.show()

"""
Simulation function.
"""
def simulate_model(p, input_current, final_t, dt):



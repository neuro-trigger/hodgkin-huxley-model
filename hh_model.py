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
import matplotlib as plot

"""
Parameter dictionary example. This dictionary contains the original
Hodgkin-Huxley parameters.
"""
HH_PARAMETERS = {
    "E_Na" : 55.0,  # Nernst potential for Na channel (mV)
    "g_Na" : 40,    # Maximum conductance for Na channel (mS/cm^2)
    "E_K" : -77.0,  # Nernst potential for K channel (mV)
    "g_K" : 35,     # Maximum conductance for K channel (mS/cm^2)
    "E_L : -65.0,   # Nernst potential for leak channel (mV)
    "g_L" : 0.3,    # Maximum conductance for leak channel (mS/cm^2)
    "C" : 1         # Membrane capacitancy (pico F/cm^2)
}

def membrane_potential_differential(s, p, i):
   na_current = p["g_Na"]*(s[1]**3)*s[3]*(s[0] - p["E_Na"])
   k_current = p["g_K"]*(s[2]**4)*(s[0] - p["E_K"])
   l_current = p["g_L"]*(s[0] - p["E_L"])

   return ( i - (na_current + k_current + l_current) ) / p[C]

def gate_m_differential(s):
    alpha = ( 0.182 * (s[0] + 35) ) / ( 1 - np.exp( -(s[0] + 35)/9 ) )
    beta = ( -0.124 * (s[0] + 35) ) / ( 1 - np.exp( (s[0] + 35)/9 ) )

    return alpha*(1 - s[1]) - beta*s[1]

def gate_h_differential(s):
    alpha = 0.25 * np.exp( -(s[0] + 90)/12 )
    beta = 0.25 * np.exp( (s[0] + 62)/6 ) / np.exp( (s[0] + 90)/12 )

    return alpha*(1 - s[3]) - beta*s[3]

def gate_n_differential(s):
    alpha = ( 0.02 * (s[0] - 25) ) / ( 1 - np.exp( -(s[0] - 25)/9 ) )
    beta = ( -0.002 * (s[0] - 25) ) / ( 1 - np.exp( (s[0] - 25)/9 ) )

    return alpha*(1 - s[2]) - beta*s[2]





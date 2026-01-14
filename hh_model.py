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

def vtrap(x, y):
    if abs(x/y) < 1e-6:
        return y * (1 + x/y/2) 
    else:
        return x / (1 - np.exp(-x/y))

"""
Differential functions.
"""
def membrane_potential_differential(u, m, n, h, p, i):
   na_current = p['g_Na']*(m**3)*h*(u - p['E_Na'])
   k_current = p['g_K']*(n**4)*(u - p['E_K'])
   l_current = p['g_L']*(u - p['E_L'])

   return ( i - (na_current + k_current + l_current) ) / p['C']

def gate_m_differential(u, m):
    alpha = 0.182 * vtrap(u + 35, 9)
    beta = 0.124 * vtrap(-(u + 35), 9)

    return alpha*(1 - m) - beta*m

def gate_n_differential(u, n):
    alpha = 0.02 * vtrap(u - 25, 9)
    beta = 0.002 * vtrap(-(u - 25), 9)

    return alpha*(1 - n) - beta*n

def gate_h_differential(u, h):
    alpha = 0.25 * np.exp( -(u + 90)/12 )
    beta = 0.25 * np.exp( (u + 62)/6 ) / np.exp( (u + 90)/12 )

    return alpha*(1 - h) - beta*h

"""
RK4 function.
"""
def rk4_step(u, m, n, h, p, input_current, t, dt):
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
def plot_results(t, u, m, n, h, i):
    fig, (ax1, ax2, ax3) = plot.subplots(3, 1, sharex=True, figsize=(8, 8))
    
    # Membrane potential plot (u)
    ax1.plot(t, u, 'b', linewidth=2)
    ax1.set_ylabel('V [mV]')
    ax1.set_title('HH Neuron, step current')
    ax1.grid(True)

    ax1.set_ylim(-75, 50) 

    # 2. Gate variables plot (m, n, h)
    ax2.plot(t, m, 'k', label='m', linewidth=1.5)
    ax2.plot(t, n, 'b', label='n', linewidth=1.5)
    ax2.plot(t, h, 'r', label='h', linewidth=1.5)
    ax2.set_ylabel('act./inact.')
    ax2.legend(loc='right')
    ax2.grid(True)
    ax2.set_ylim(0, 1.1)

    # 3. Input current plot (i)
    ax3.plot(t, i, 'b', linewidth=2)
    ax3.set_ylabel('I [micro A]')
    ax3.set_xlabel('t [ms]')
    ax3.grid(True)

    ax3.set_ylim(-0.5, max(i)*1.1 if max(i) > 0 else 1)

    plot.tight_layout()
    plot.show()

"""
Simulation function.
"""
def simulate_model(u0, m0, n0, h0, p, input_current, final_t, dt):
    t = [0]
    u = [u0]
    m = [m0]
    n = [n0]
    h = [h0]
    i = [input_current(0)]

    while t[-1] < final_t:
        nu, nm, nn, nh = rk4_step(u[-1], m[-1], n[-1], h[-1], p, input_current,
                                  t[-1], dt)
        t.append(t[-1] + dt)
        u.append(nu)
        m.append(nm)
        n.append(nn)
        h.append(nh)
        i.append(input_current(t[-1]))

    return (
            np.array(t),
            np.array(u),
            np.array(m),
            np.array(n),
            np.array(h),
            np.array(i)
            )

if __name__ == "__main__":

    def step_current(t):
        if 1.0 <= t <= 120.0:
            return 1.4
        else:
            return 0.0

    u0 = -65  
    m0 = 0.05
    n0 = 0.0
    h0 = 0.65

    final_t = 140
    dt = 0.001

    print("Simulando...")
    t, u, m, n, h, i = simulate_model(u0, m0, n0, h0, HH_PARAMETERS, step_current, final_t, dt)
    print("Graficando resultados...")
    plot_results(t, u, m, n, h, i)


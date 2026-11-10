
import numpy as np
from scipy.optimize import root_scalar
from scipy.special import eval_legendre
from scipy.interpolate import interp1d

#reconvert r_star to r
def get_r_from_r_star(r_star_val, M):

  res = root_scalar(
      lambda r: r_star_val - (r + 2 * M * np.log(r / (2.0 * M) - 1.0)),
      bracket=[2.0 * M + 1e-4, r_star_val + 10.0 * M],
  )
  return res.root

#cosmic string motion
def string_pos(t_current, t_array, r_array):
    idx = np.abs(t_current - t_array).argmin()
    return r_array[idx]


def del_r(r, r_s, sigma):      #Assuming a narrow gaussian
    z = (r-r_s) / sigma
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * sigma)

def del_p_r(r, r_s, sigma):
    delta = del_r(r, r_s, sigma)
    return -(r - r_s) / sigma**2 * delta


def zerilli_solver(
    M_values,
    mu_values,
    E_values,
    l,
    theta,
    Nt,
    Nr,
    target_rs_values,
    r_obs
):

    
    M_values, mu_values, E_values = np.broadcast_arrays(
        np.asarray(M_values, dtype=float),
        np.asarray(mu_values, dtype=float),
        np.asarray(E_values, dtype=float),
    )

    l = np.asarray(l)
    nummodes = len(l)

    results = []


    for M, mu, E in zip(
        M_values.flat, mu_values.flat, E_values.flat
    ):

        tau_min = 0
        tau_max = np.arccos(M/np.sqrt(M**2 + E**2))
        tau = np.linspace(tau_min, tau_max * (1.0 - 1e-9), 1000)


        d = (np.sqrt(M**2 + E**2) - M) / E

        t = E * tau + 2 * M * np.log(
            np.abs(
                (np.tan(tau / 2) + d)
                / (np.tan(tau / 2) - d)
            )
        )

        r = M + np.sqrt(M**2 + E**2) * np.cos(tau)

        sort_indices = np.argsort(t)
        t = t[sort_indices]
        r = r[sort_indices]

        d = (np.sqrt(M**2 + E**2) - M)/E
        t = np.array(E*tau + 2*M*np.log(np.abs((np.tan(tau/2) + d)/(np.tan(tau/2) - d))))       #physical time

        r = np.array(M + np.sqrt(M**2 + E**2)*np.cos(tau))      #physical radius

        sort_indices = np.argsort(t)  # finds the order to put time from smallest to largest
        t = t[sort_indices]           # reorganizes the time values
        r = r[sort_indices]  


        epsilon = 0.02 #just above horizon
        r_min = 2.0 * M + epsilon    #horizon

        #tortoise coordinate
        r_star_min = r_min + 2 * M * np.log(r_min / (2.0 * M) - 1.0)  
        r_star_max = 100 * M + 2 * M * np.log(100 * M / (2.0 * M) - 1.0)

        r_star = np.linspace(r_star_min, r_star_max, Nr)
        dr_star = r_star[1] - r_star[0]  
        dt = 0.5 * dr_star

        sigma = 6*dr_star   #gaussian parameter

        r_i = np.array([get_r_from_r_star(rs, M) for rs in r_star])
        r_dot_arg = np.maximum(0.0, E**2 - r_i**2 * (1.0 - 2.0 * M / r_i))
        r_dot = np.sqrt(r_dot_arg)

        f = 1-2*M/r_i   #schwarzschild term
        f_p = 2*M/r_i**2



        obs_idx = [np.argmin(np.abs(r_i - val)) for val in r_obs]

 
        #Angular parameters


        gamma = ((l - 1) * (l + 2))[:, None]
        lam = gamma + 6 * M / r_i

        #Zerilli Potential
        V_even = 1/lam**2 * (gamma**2 * ((gamma+2)/r_i**2 + M/r_i**3) + 36*M**2/r_i**4*(gamma + 2*M/r_i))
        V_z = f*V_even

   
        #Source term

        P_l = eval_legendre(l, np.cos(theta))
        Y_l_1d = np.sqrt((2*l+1)/(4*np.pi)) * P_l
        Y_l= Y_l_1d[:,np.newaxis]
        K_1 = (32*np.pi**2 * mu * Y_l) / (E * lam) 
        K_2 = gamma*(gamma-2) + 12*(gamma-3)*M/r_i + 84*M**2/r_i**2

        A_l = K_1 * (lam*f*r_i/gamma + 1/(gamma+2) * ((24*f*M*r_dot**2)/(lam*r_i**2) - 6*f**2*r_i - r_i*f*K_2/lam))

        B_l = K_1 * (2 * r_i**2 * f**2)/(gamma + 2)

        # Initial conditions

        r_s0 = string_pos(0, t, r)
        S_even = A_l*del_r(r_i, r_s0, sigma) - B_l*del_p_r(r_i, r_s0,sigma)
        S = -f*S_even
        
        psi_prev = np.zeros((nummodes, Nr))
        psi_curr = np.zeros((nummodes, Nr))
        psi_next = 0.5 * dt**2 * S

        psi_prev = psi_curr.copy()
        psi_curr = psi_next.copy()

        psi_ext = np.zeros((nummodes, Nt, len(obs_idx)))
        
        wave_snapshots = {}

     
        #Time evolution of Zerilli equation
        # d^2 psi/dt^2 = d^2psi/dr_star^2 - fV_even psi - fS_even

        for n in range(Nt):

            t_current = n*dt
            r_s = string_pos(t_current, t, r)
            
            
            S_even = A_l*del_r(r_i, r_s, sigma) - B_l*del_p_r(r_i, r_s,sigma)
            S = -f*S_even
            
            d2psi_drstar2 = (
            psi_curr[:, 2:] - 2 * psi_curr[:, 1:-1] + psi_curr[:, :-2]) / dr_star**2
                
            V_z_term = V_z[:, 1:-1] * psi_curr[:, 1:-1]
                
            psi_next[:, 1:-1] = (
            2.0 * psi_curr[:, 1:-1]
            - psi_prev[:, 1:-1]
            + dt**2 * (d2psi_drstar2 - V_z_term + S[:, 1:-1]))
            
            psi_next[:, 0] = psi_curr[:, 0] + (dt / dr_star) * (psi_curr[:, 1] - psi_curr[:, 0])
            psi_next[:, -1] = psi_curr[:, -1] - (dt / dr_star) * (psi_curr[:, -1] - psi_curr[:, -2])
            psi_ext[:, n, :] = psi_curr[:, obs_idx]
            psi_prev = psi_curr.copy()
            psi_curr = psi_next.copy()
            for target in target_rs_values:
                if target not in wave_snapshots and r_s <= target:
                    wave_snapshots[target] = (r_s, psi_curr.copy())

        results.append({
            "M": M,
            "mu": mu,
            "E": E,
            "l": l.copy(),
            "r_i": r_i,
            "r_star": r_star,
            "t": t,
            "r": r,
            "dt": dt,
            "dr_star": dr_star,
            "f": f,
            "f_p": f_p,
            "r_dot": r_dot,
            "V_z": V_z,
            "r_obs": r_obs,
            "obs_idx": obs_idx,
            "psi_ext": psi_ext,
            "wave_snapshots": wave_snapshots,
        })

    return results


def psi_at_infinity(M, t_sim, r_obs, psi_ext, l_modes):
    #using 1/r polynomial fitting
    num_l = len(l_modes)
    Nt = len(t_sim)


    r_obs_max = r_obs[-1]
    r_star_max = r_obs_max + 2.0 * M * np.log(r_obs_max / (2.0 * M) - 1.0)
    u_global = t_sim - r_star_max

    psi_even_inf = np.zeros((num_l, Nt))
    psi_u_grid = np.zeros((len(r_obs), num_l, Nt))

    for idx, r_val in enumerate(r_obs):
        r_star = r_val + 2.0 * M * np.log(r_val / (2.0 * M) - 1.0)
        u_local = t_sim - r_star

        for i in range(num_l):
            f_interp = interp1d(
                u_local,
                psi_ext[i, :, idx],
                kind='cubic',
                bounds_error=False,
                fill_value=0.0,
            )
            psi_u_grid[idx, i, :] = f_interp(u_global)

    inv_r = 1.0 / r_obs

    for i in range(num_l):
        for t_idx in range(Nt):
            psi_values = psi_u_grid[:, i, t_idx]
            poly_coeffs = np.polyfit(inv_r, psi_values, deg=1)
            psi_even_inf[i, t_idx] = poly_coeffs[1]  

    return u_global, psi_even_inf

def h_plus_infinity(u_time, psi_even_inf, l_modes, theta):
  x = np.cos(theta)
  h_plus = np.zeros_like(u_time)

  for i, l_val in enumerate(l_modes):
    P_l = eval_legendre(l_val, x)
    P_l_minus_1 = eval_legendre(l_val - 1, x)
    P_l_p = (l_val / np.sin(theta) ** 2) * (P_l_minus_1 - x * P_l)

    N_l = np.sqrt((2.0 * l_val + 1.0) / (4.0 * np.pi))
    W_l0 = N_l * (x * P_l_p - l_val * (l_val + 1.0) * P_l)

    h_plus += psi_even_inf[i, :] * W_l0

  return h_plus
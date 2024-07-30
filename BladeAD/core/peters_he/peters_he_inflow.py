import numpy as np 
import csdl_alpha as csdl
import matplotlib.pyplot as plt
from BladeAD.utils.integration_schemes import integrate_quantity
from BladeAD.utils.var_groups import RotorAnalysisOutputs
from BladeAD.utils.smooth_quantities_spanwise import smooth_quantities_spanwise
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel
from BladeAD.utils.switching_fun import sigmoid
import pandas as pd
from BladeAD import PETERS_HE_NUM_SHAPE_FUN_PER_HARM_CSV

df = pd.read_csv(PETERS_HE_NUM_SHAPE_FUN_PER_HARM_CSV)

def double_factorial(n):
    """Implement the double factorial operation."""
    if n < 0:
        raise ValueError("Double factorial is not defined for negative integers.")
    elif n == 0 or n == 1:
        return 1
    else:
        return n * double_factorial(n - 2)

# def compute_gamma(r, m, j, n):
#     if (r + m) % 2 == 0:
#         num_1 = (-1)**((n + j -2 * r) / 2)
#         den_1 = (compute_h(n, m) * compute_h(j, r))**0.5
#         term_1 = num_1 / den_1

#         num_2 = 2 * ((2 * n + 1) * (2 * j + 1))**0.5
#         den_2 = (j + n) * (j + n + 2) * ((j - n)**2 - 1)
#         term_2 = num_2 / den_2

#         return term_1 * term_2
    
#     elif ((r + m) % 2 != 0) and (abs(j - 1) == 1):
#         num_1 = np.pi
#         den_1 = (compute_h(n, m) * compute_h(j, r))**0.5
#         term_1 = num_1 / den_1

#         num_2 = np.sign(r - m)
#         den_2 = (2 * n + 1) * (2 * j + 1)
#         term_2 = num_2 / den_2

#         return term_1 * term_2
    
#     elif  ((r + m) % 2 != 0) and (abs(j - 1) != 1):
#         return 0
    
#     else:
#         raise NotImplementedError

# def compute_L(Q, r_harmonic, m_harmonic, x=2, type='cos'):
#     if Q % 2 == 0:
#         sub_mat_size = int((Q + 2) / 2)
#     else:
#         sub_mat_size = int((Q + 1) / 2)

#     if type == 'sin':
#         if isinstance(x, csdl.Variable):
#             shape = (r_harmonic * sub_mat_size, m_harmonic * sub_mat_size)
#             L_mat = csdl.Variable(shape=shape, value=np.zeros(shape))
#         else:
#             shape = (r_harmonic * sub_mat_size, m_harmonic * sub_mat_size)
#             L_mat = np.zeros(shape)
#     else:
#         if isinstance(x, csdl.Variable):
#             shape = ((r_harmonic + 1) * sub_mat_size, (m_harmonic +1) * sub_mat_size)
#             L_mat = csdl.Variable(shape=shape, value=np.zeros(shape))
#         else:
#             shape = ((r_harmonic + 1) * sub_mat_size, (m_harmonic +1) * sub_mat_size)
#             L_mat = np.zeros(shape)

#     vec_ind_list = []
#     for r in range(r_harmonic):
#         print("r", r)
#         if type == 'sin':
#             r += 1
#         else:
#             r = r
#         # if r > r_harmonic:
#         #     break
#         for m in range(m_harmonic):
#             print("m", m)
#             if type == 'sin':
#                 m += 1
#             else:
#                 m = m
#             # if m > m_harmonic:
#             #     break
#             if r == 0:
#                     multiplier = x**m
#             else:
#                 if type == 'sin':
#                     multiplier = x**(((m - r)**2)**0.5) - (-1)**min(r, m) * x**(m + r)
#                 else:
#                     multiplier = x**(((m - r)**2)**0.5) + (-1)**min(r, m) * x**(m + r)
#             if isinstance(x, csdl.Variable):
#                 sub_mat = csdl.Variable(shape=(sub_mat_size, sub_mat_size), value=0)
#             else:
#                 sub_mat = np.zeros((sub_mat_size, sub_mat_size))
#             for j in range(sub_mat_size):
#                 for n  in range(sub_mat_size):
#                     if isinstance(x, csdl.Variable):
#                         sub_mat = sub_mat.set(
#                             slices=csdl.slice[j, n], 
#                             value=multiplier * compute_gamma(r, m, r + 2 * j + 1, m + 2 * n + 1),
#                         )
#                     else:
#                         sub_mat[j, n] = multiplier * compute_gamma(r, m, r + 2 * j + 1, m + 2 * n + 1)
#             #         print(r, m, r + 2 * j + 1, m + 2 * n + 1)
#             # print("submat", sub_mat)

#             if type == 'sin':
#                 row_ind_1 = sub_mat_size * (r-1)
#                 row_ind_2 = row_ind_1 + sub_mat_size
#                 col_ind_1 = sub_mat_size * (m-1)
#                 col_ind_2 = col_ind_1 + sub_mat_size
#                 if isinstance(x, csdl.Variable):
#                     L_mat = L_mat.set(
#                         slices=csdl.slice[row_ind_1:row_ind_2, col_ind_1:col_ind_2],
#                         value=sub_mat
#                     )
#                 else:   
#                     L_mat[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = sub_mat
    
#             else:
#                 row_ind_1 = sub_mat_size * r
#                 row_ind_2 = row_ind_1 + sub_mat_size
#                 col_ind_1 = sub_mat_size * m
#                 col_ind_2 = col_ind_1 + sub_mat_size
#                 if isinstance(x, csdl.Variable):
#                     L_mat = L_mat.set(
#                         slices=csdl.slice[row_ind_1:row_ind_2, col_ind_1:col_ind_2],
#                         value=sub_mat
#                     )
#                 else:   
#                     L_mat[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = sub_mat

#             if type == 'sin':
#                 if m == 1:
#                     for vec_ind in range(sub_mat_size):
#                         vec_ind_list.append((r, r + 2 * vec_ind + 1))
#             else:
#                 print("r, m", r, m)
#                 if m == 0:
#                     for vec_ind in range(sub_mat_size):
#                         vec_ind_list.append((r, r + 2 * vec_ind + 1))
#                         # vec_ind.append((r, r + 1))
#                         # vec_ind.append((r, r + 3))


#             # L_mat[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = sub_mat

#     return L_mat, vec_ind_list

# def compute_h(j, r):
#     num_1 = double_factorial(j + r - 1)
#     if j - r < 0:
#         num_2 = 1
#     else:
#         num_2 = double_factorial(j - r - 1)
#     den_1 = double_factorial(j + r)
#     if j - r < 0:
#         den_2 = 1
#     else:
#         den_2 = double_factorial(j - r)

#     return (num_1 * num_2) / (den_1 * den_2)

def compute_h(j, r):
    num_1 = double_factorial(j + r - 1)
    if j - r < 0:
        num_2 = 1
    else:
        num_2 = double_factorial(j - r - 1)
    den_1 = double_factorial(j + r)
    if j - r < 0:
        den_2 = 1
    else:
        den_2 = double_factorial(j - r)

    return (num_1 * num_2) / (den_1 * den_2)

def compute_gamma(r, m, j, n):
    if (r + m) % 2 == 0:
        num_1 = (-1)**((n + j -2 * r) / 2)
        den_1 = (compute_h(n, m) * compute_h(j, r))**0.5
        term_1 = num_1 / den_1

        num_2 = 2 * ((2 * n + 1) * (2 * j + 1))**0.5
        den_2 = (j + n) * (j + n + 2) * ((j - n)**2 - 1)
        term_2 = num_2 / den_2

        return term_1 * term_2
    
    elif ((r + m) % 2 != 0) and (abs(j - 1) == 1):
        num_1 = np.pi
        den_1 = (compute_h(n, m) * compute_h(j, r))**0.5
        term_1 = num_1 / den_1

        num_2 = np.sign(r - m)
        den_2 = (2 * n + 1) * (2 * j + 1)
        term_2 = num_2 / den_2

        return term_1 * term_2
    
    elif  ((r + m) % 2 != 0) and (abs(j - 1) != 1):
        return 0
    
    else:
        raise NotImplementedError

def compute_L(Q: int, M:int, x=2, type_:str="cos"):
    """Compute the L matrix

    Parameters
    ----------
    Q : int
        Highest power of non-dimensional radius
    M : int
        Highest harmonic number
    x : int, optional
        wake skew angle, by default 2 (dummy number)
    type_ : str, optional
        wether to make cosine or sine L matrix, by default "cos"

    Returns
    -------
    _type_
        _description_
    """

    if isinstance(x, csdl.Variable):
        do_csdl = True
    else:
        do_csdl = False

    if Q % 2 == 0: # Even
        sub_mat_size = int((Q + 2) / 2)
    else: # Odd
        sub_mat_size = int((Q + 1) / 2)

    if type_ == "cos": # Include 0th harmonic -> M+1 total harmonics
        num_harm = M + 1
        shape = (num_harm * sub_mat_size, num_harm * sub_mat_size)
        if do_csdl:
            L_mat = csdl.Variable(shape=shape, value=0)
        else:
            L_mat = np.zeros(shape)
    
    else: # Exclude 0th harmonic -> M total harmonics
        num_harm = M 
        shape = (num_harm * sub_mat_size, num_harm * sub_mat_size)
        if do_csdl:
            L_mat = csdl.Variable(shape=shape, value=0)
        else:
            L_mat = np.zeros(shape)

    vec_ind = []
    for a in range(num_harm):
        for b in range(num_harm):
            if type_ == "sin":
                r = a + 1
                m = b + 1
            else:
                r = a
                m = b

            if r == 0:
                    multiplier = x**m
            else:
                min_L = min(r, m)
                if type_ == 'sin':
                    multiplier = x**(abs(m - r)) - (-1)**min_L * x**(m + r)
                else:
                    multiplier = x**(abs(m - r)) + (-1)**min_L * x**(m + r)

            if do_csdl:
                sub_mat = csdl.Variable(shape=(sub_mat_size, sub_mat_size), value=0.)
            else:
                sub_mat = np.zeros((sub_mat_size, sub_mat_size))
            for c in range(sub_mat_size):
                j = r +  (2 *c + 1)
                for d in range(sub_mat_size):
                    n = m +  (2 *d + 1)

                    # Populate the sub-matrix
                    if do_csdl:
                        sub_mat = sub_mat.set(
                            slices=csdl.slice[c, d],
                            value=multiplier * compute_gamma(r, m, j, n)
                        )
                    else:
                        sub_mat[c, d] = multiplier * compute_gamma(r, m, j, n)
                    
                    # Append the harmonic (m) and shape function index (n)
                    # for the loading coefficients
                    if a == 0 and c == 0: # Only get the top row of the L matrix
                        vec_ind.append((m, n))
            
            # Get the number of shape functions per harmonic
            max_harmonic = max(r, m)
            num_fun_per_harm = int(df.iloc[int(Q), max_harmonic+1])
            
            # zero out sub-mat indices if necessary (refer to data frame)
            if num_fun_per_harm < sub_mat_size:
                for sub_mat_i in range(sub_mat_size):
                    for sub_mat_j in range(sub_mat_size):
                        if sub_mat_i >= num_fun_per_harm or sub_mat_j >=num_fun_per_harm:
                            if do_csdl:
                                sub_mat = sub_mat.set(
                                    slices=csdl.slice[sub_mat_i, sub_mat_j],
                                    value=0.
                                )
                            else:
                                sub_mat[sub_mat_i, sub_mat_j] = 0

            # print(sub_mat)

            if type_ == 'sin':
                row_ind_1 = sub_mat_size * (r-1)
                row_ind_2 = row_ind_1 + sub_mat_size
                col_ind_1 = sub_mat_size * (m-1)
                col_ind_2 = col_ind_1 + sub_mat_size

            else:
                row_ind_1 = sub_mat_size * r
                row_ind_2 = row_ind_1 + sub_mat_size
                col_ind_1 = sub_mat_size * m
                col_ind_2 = col_ind_1 + sub_mat_size

            if do_csdl:
                L_mat = L_mat.set(
                    slices=csdl.slice[row_ind_1:row_ind_2, col_ind_1:col_ind_2],
                    value=sub_mat,
                )
            else:
                L_mat[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = sub_mat
    
    # print(L_mat)

    return L_mat, vec_ind                    

def compute_phi(r, j, radius, Q):
    term_1 = ((2 * j + 1) * compute_h(j, r))**0.5
    summation = csdl.Variable(shape=radius.shape, value=0)
    for q in range(r, j, 2):
        # if (q > Q) or (q > (j - 1)):
        if (q > (j - 1)):
            break
        num_1 = (-1)**((q - r) / 2)
        num_2 = double_factorial(j + q)
        den_1 = double_factorial(q - r)
        den_2 = double_factorial(q + r)
        den_3 = double_factorial(j - q - 1)
        summation = summation + radius**q * num_1 * num_2 / (den_1 * den_2 * den_3)
    
    return term_1 * summation

def solve_for_steady_state_inflow(
    shape: tuple,
    psi: csdl.Variable,
    rpm: csdl.Variable,
    radius: csdl.Variable,
    mu_z: csdl.Variable,
    mu: csdl.Variable,
    tangential_velocity: csdl.Variable,
    frame_velocity: csdl.Variable,
    radius_vec: csdl.Variable,
    chord_profile: csdl.Variable,
    twist_profile: csdl.Variable,
    airfoil_model: csdl.Variable,
    # atmos_states: csdl.VariableGroup,
    rho_exp: csdl.Variable,
    mu_exp: csdl.Variable,
    a_exp: csdl.Variable,
    num_blades: int,
    dr: csdl.Variable,
    integration_scheme: str, 
    hub_radius,
    radius_vec_exp,
    norm_radius_vec_exp,
    disk_inclination_angle,
    tip_loss,
    Q : int = 4,
    M : int = 4,
    initial_value: csdl.Variable = None
) -> RotorAnalysisOutputs:
    """Solve the Peters--He [1] dynamic inflow model for steady state.
    The implementation procedure is based on a Master's thesis by Kevin Li [2].

    Parameters
    ----------
    shape : tuple
        _description_
    psi : csdl.Variable
        _description_
    rpm : csdl.Variable
        _description_
    radius : csdl.Variable
        _description_
    mu_z : csdl.Variable
        _description_
    mu : csdl.Variable
        _description_
    tangential_velocity : csdl.Variable
        _description_
    frame_velocity : csdl.Variable
        _description_
    radius_vec : csdl.Variable
        _description_
    chord_profile : csdl.Variable
        _description_
    twist_profile : csdl.Variable
        _description_
    airfoil_model : csdl.Variable
        _description_
    atmos_states : csdl.VariableGroup
        _description_
    num_blades : int
        _description_
    dr : csdl.Variable
        _description_
    integration_scheme : str
        _description_
    Q : int, optional
        _description_, by default 3
    M : int, optional
        _description_, by default 3

    Returns
    -------
    RotorAnalysisOutputs : csdl.VariableGroup
        _description_

    References
    ----------
    [1] Li, S. Development of Rotorcraft Forward Flight Analysis Code Using the Finite-State Dynamic Inflow Model. Diss. Masters thesis, University of California Davis, 2020.

    [2] Peters, David A., and Cheng Jian He. "Finite state induced flow models. II-Three-dimensional rotor disk." Journal of Aircraft 32.2 (1995): 323-333.
    """

    # Pre-processing
    num_nodes = shape[0]
    num_radial = shape[1]
    num_azimuthal = shape[2]

    B = num_blades
    omega = rpm * 2 * np.pi / 60
    Vt = tangential_velocity
    Vx = frame_velocity[:, :, :, 0]
    Vr = (Vx**2 + Vt**2) ** 0.5
    mu_atmos = mu_exp #atmos_states.dynamic_viscosity
    rho = rho_exp #atmos_states.density
    a = a_exp #atmos_states.speed_of_sound
    Re = rho * Vr * chord_profile / mu_atmos
    Ma = Vr / a
    
    # setting up the dimensions of the state vector
    _, vec_ind_sin = compute_L(Q, M, type_="sin")
    _, vec_ind_cos = compute_L(Q, M, type_="cos")

    ns_sin = len(vec_ind_sin)
    ns_cos = len(vec_ind_cos)

    # setting variables to store quantities of interest
    states_container = csdl.Variable(shape=(num_nodes, ns_cos + ns_sin), value=0)
    ux_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    alpha_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    dT_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    dL_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    dQ_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    thrust_container = csdl.Variable(shape=(num_nodes, ), value=0)
    power_container = csdl.Variable(shape=(num_nodes, ), value=0)
    torque_container = csdl.Variable(shape=(num_nodes, ), value=0)
    residual_container = csdl.Variable(shape=(num_nodes, ns_cos + ns_sin), value=0)
    C_T_containter = csdl.Variable(shape=(num_nodes, ), value=0)
    C_P_containter = csdl.Variable(shape=(num_nodes, ), value=0)
    C_Q_containter = csdl.Variable(shape=(num_nodes, ), value=0)
    eta_container = csdl.Variable(shape=(num_nodes, ), value=0)
    FoM_container = csdl.Variable(shape=(num_nodes, ), value=0)
    inflow_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)

    for i in range(num_nodes):
        # Initialize implicit variables (state vector)
        state_vec = csdl.ImplicitVariable(shape=((ns_cos + ns_sin, )), value=0)
        inflow = csdl.Variable(shape=(num_radial, num_azimuthal), value=0)

        for nsc in range(ns_cos):
            r = vec_ind_cos[nsc][0]
            j = vec_ind_cos[nsc][1]

            phi_vec = compute_phi(r, j, radius_vec[i, :, :] / radius, Q)
            cos_psi = csdl.cos(r * psi[i, :, :])

            inflow = inflow + phi_vec * state_vec[nsc] * cos_psi

        for nss in range(ns_sin):
            r = vec_ind_sin[nss][0]
            j = vec_ind_sin[nss][1]
            
            phi_vec = compute_phi(r, j, radius_vec[i, :, :] / radius, Q)
            sin_psi = csdl.sin(r * psi[i, :, :])
            inflow = inflow + phi_vec * state_vec[nss+ns_cos] * sin_psi

        ux = (inflow + mu_z[i]) * omega[i] * radius
        phi = csdl.arctan((ux) / Vt[i, :, :])
        
        # Compute sectional Cl, Cd
        alpha = twist_profile[i, :, :] - phi
        Cl, Cd = airfoil_model.evaluate(alpha, Re[i, :, :], Ma[i, :, :])

        # Smooth the transition if airfoil model consists of multiple airfoils
        if isinstance(airfoil_model, CompositeAirfoilModel):
            if airfoil_model.smoothing:
                window = airfoil_model.transition_window
                indices = airfoil_model.stop_indices
                Cl = smooth_quantities_spanwise(Cl, indices, window)
                Cd = smooth_quantities_spanwise(Cd, indices, window)

        # Prandtl tip losses 
        f_tip = num_blades / 2 * (radius - radius_vec_exp[i, :, :]) / radius / csdl.sin(phi)
        f_hub = num_blades / 2 * (radius_vec_exp[i, :, :] - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-(f_tip**2)**0.5))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-(f_hub**2)**0.5))

        F_initial = F_tip * F_hub 
        if tip_loss:
            F = (1 + (F_initial - 1) * sigmoid(disk_inclination_angle[i], np.deg2rad(12))) #* 0.6
        else:
            F = 1

        # Compute thrust (and torque)
        # sectional (dT)
        Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)
        dT = 0.5 * B * rho[i, :, :] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr * F

        # total thrust/torque and their coefficients
        thrust = integrate_quantity(dT, scheme=integration_scheme)
        C_T = thrust / (rho[i, 0, 0] * np.pi * radius**2 * (omega[i] * radius) ** 2)

        # Initialize loading coefficients
        tau_cos = csdl.Variable(shape=(ns_cos, ), value=0)
        tau_sin = csdl.Variable(shape=(ns_sin, ), value=0)
        
        # cosine terms 
        for nsc in range(ns_cos):
            m = vec_ind_cos[nsc][0] # harmonic number 
            n = vec_ind_cos[nsc][1] # shape function index

            # compute shape function 
            phi_cos = compute_phi(m, n, radius_vec[i, :, :]/radius, Q)
            L_cos = dT * phi_cos

            # compute cosine harmonic and apply to loading
            if m == 0:
                pass
                # L_cos = L_cos
            else:
                L_cos = L_cos * csdl.cos(m * psi[i, :, :]) 

            if m == 0:
                tau_cos = tau_cos.set(
                    slices=csdl.slice[nsc], value=integrate_quantity(L_cos, integration_scheme) / (2 * np.pi * rho[i, 0, 0] * omega[i]**2 * radius**4)
                )
            else:
                tau_cos = tau_cos.set(
                    slices=csdl.slice[nsc], value=integrate_quantity(L_cos, integration_scheme) / (np.pi * rho[i, 0, 0] * omega[i]**2 * radius**4)
                )
                

        # sine terms
        for nss in range(ns_sin):
            m = vec_ind_sin[nss][0] # harmonic number
            n = vec_ind_sin[nss][1] # shape function index

            # compute shape function 
            phi_sin = compute_phi(m, n, radius_vec[i, :, :]/radius, Q)
            L_sin = dT * phi_sin

            # compute sine harmonic and apply to loading
            L_sin = L_sin * csdl.sin(m * psi[i, :, :]) 

            tau_sin = tau_sin.set(
                slices=csdl.slice[nss], value=integrate_quantity(L_sin, integration_scheme)/ (np.pi * rho[i, 0, 0] * omega[i]**2 * radius**4)
            )

        tau_cos = tau_cos #/ (2 * np.pi * rho * omega[i]**2 * radius**4) #csdl.sum(tau_cos / (2 * np.pi * rho * omega**2 * radius**4)) / num_azimuthal
        tau_sin = tau_sin #/ (2 * np.pi * rho * omega[i]**2 * radius**4) #csdl.sum(tau_sin / (2 * np.pi * rho * omega**2 * radius**4)) / num_azimuthal

        # assemble loading vector
        tau = csdl.Variable(shape=(ns_cos + ns_sin, ), value=0)
        tau = tau.set(slices=csdl.slice[0:ns_cos], value=tau_cos)
        tau = tau.set(slices=csdl.slice[ns_cos:], value=tau_sin)

        # compute tangential inflow via Newton iteration
        lam_i = csdl.ImplicitVariable(shape=(1, ), value=0.1)
        lam_i_res = lam_i - mu[i] * csdl.tan(-disk_inclination_angle[i]) - C_T / (2 * (mu[i]**2 + lam_i**2)**0.5)
        d_lam0res_d_lam0 = 1 + C_T / 2 * (mu[i]**2 + lam_i**2)**(-3/2) * lam_i

        lam_i_solver = csdl.nonlinear_solvers.GaussSeidel(print_status=False)
        lam_i_solver.add_state(lam_i, lam_i_res, state_update= lam_i - lam_i_res / d_lam0res_d_lam0)
        lam_i_solver.run()
        lam_m = lam_i * 3**0.5
        lam = lam_m + mu_z[i]

        # lam_i = csdl.ImplicitVariable(shape=(1, ), value=0.1)
        # lam_i_res = lam_i - C_T / (2 * (mu[i]**2 + (mu_z[i] + lam_i)**2)**0.5)
        # lam_i_solver = csdl.nonlinear_solvers.Newton(print_status=False)
        # lam_i_solver.add_state(lam_i, lam_i_res)
        # lam_i_solver.run()

        # # lam = lam_i + mu_z[i]
        # lam = lam_i * 3**0.5 + mu_z[i]

        # assemble L matrices 
        chi = csdl.arctan((mu[i]  + 1e-7) / lam) # wake skew angle chi
        x = csdl.tan(((chi / 2)**2)**0.5)

        # Effective velocities
        V_eff = (mu[i]**2 + lam * (lam + lam_m)) / (mu[i] ** 2 + lam**2) ** 0.5
        V_eff_t = (mu[i] ** 2 + lam**2) ** 0.5

        L_cos, vec_ind_cos = compute_L(Q, M, x=x, type_='cos') 
        L_sin, vec_ind_sin = compute_L(Q, M, x=x, type_='sin') 
        # exit()

        # assemble diagonal V matrices
        V_cos = csdl.Variable(shape=(ns_cos, ns_cos), value=0)
        V_sin = csdl.Variable(shape=(ns_sin, ns_sin), value=0)
        for vc_i in range(ns_cos):
            for vc_j in range(ns_cos):
                if vc_i == vc_j:
                    if vec_ind_cos[vc_i][0] == 0:
                        V_cos = V_cos.set(slices=csdl.slice[vc_i, vc_j], value= 1 / V_eff_t) 
                    else:
                        V_cos = V_cos.set(slices=csdl.slice[vc_i, vc_j], value=1 / V_eff)
        for vs_i in range(ns_sin):
            for vs_j in range(ns_sin):
                if vec_ind_sin[vs_i][0] == 0:
                    V_sin = V_sin.set(slices=csdl.slice[vs_i, vs_j], value= 1 / V_eff_t) 
                else:
                    V_sin = V_sin.set(slices=csdl.slice[vs_i, vs_j], value=1 / V_eff)
        
        L_cos = csdl.matmat(L_cos, V_cos)  
        L_sin = csdl.matmat(L_sin, V_sin)

        # assemble cos/sin matrices into L-block matrix
        L_block = csdl.Variable(shape=((ns_cos + ns_sin, ns_cos + ns_sin)), value=0)
        L_block = L_block.set(slices=csdl.slice[0:ns_cos, 0:ns_cos], value=L_cos)
        L_block = L_block.set(slices=csdl.slice[ns_cos:, ns_cos:], value=L_sin)

        # define residual based on steady state assumption
        residual = csdl.matvec(L_block, tau) - state_vec

        if initial_value is None:
            # basic fixed point iteration for state update
            # state_update = 0.05 * (csdl.matvec(L_block, tau) - state_vec)
            solver = csdl.nonlinear_solvers.GaussSeidel(max_iter=100, elementwise_states=False)
            # solver.add_state(state_vec, residual, state_update=state_update)
            solver.add_state(state_vec, residual, state_update=state_vec + 0.05 * residual)
        else:
            solver = csdl.nonlinear_solvers.Newton(max_iter=40, elementwise_states=False)
            solver.add_state(state_vec, residual, initial_value=initial_value[i, :])
            # solver.add_state(state_vec, residual, state_update=state_vec + state_update)

        solver.run()
        
        dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr  * F
        dQ = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Ct * radius_vec[i, :, :] * dr * F
        dL = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cl * dr
        thrust = integrate_quantity(dT, scheme=integration_scheme)
        torque = integrate_quantity(dQ, scheme=integration_scheme)
        
        # compute thrust/torque/power coefficient
        CT = thrust / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius)**4
        area = np.pi * radius**2
        V_tip = rpm[i] / 60 * 2 * np.pi * radius
        C_T_new = thrust / rho[i, 0, 0]/ area / V_tip**2

        CQ = torque / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius)**5
        CP = 2 * np.pi * CQ

        power = CP * rho[i, 0, 0] * (rpm[i] / 60)**3 * (2 * radius)**5


        # Compute advance ratio and efficiency and FOM
        J = Vx[i, 0, 0] / (rpm[i] / 60) / (2 * radius)
        eta = CT * J / CP
        FoM = CT * (CT/2)**0.5 / CP

        # Storing data
        inflow_container = inflow_container.set(csdl.slice[i, :, :], inflow)# + mu_z[i])
        ux_container = ux_container.set(csdl.slice[i, :, :], ux)
        alpha_container = alpha_container.set(csdl.slice[i, :, :], alpha * 180 /np.pi)
        dT_container = dT_container.set(csdl.slice[i, :, :], dT)
        dL_container = dT_container.set(csdl.slice[i, :, :], dL)
        dQ_container = dQ_container.set(csdl.slice[i, :, :], dQ)
        thrust_container = thrust_container.set(csdl.slice[i], thrust)
        torque_container = torque_container.set(csdl.slice[i], torque)
        power_container = power_container.set(csdl.slice[i], power)
        C_T_containter = C_T_containter.set(csdl.slice[i], C_T_new)
        C_P_containter = C_P_containter.set(csdl.slice[i], CP)
        C_Q_containter = C_Q_containter.set(csdl.slice[i], CQ)
        eta_container = eta_container.set(csdl.slice[i], eta)
        FoM_container = FoM_container.set(csdl.slice[i], FoM)
        

        states_container = states_container.set(csdl.slice[i, :], state_vec)
        residual_container = residual_container.set(csdl.slice[i, :], residual)

    outputs = RotorAnalysisOutputs(
        axial_induced_velocity=ux_container,
        sectional_drag=None,
        tangential_induced_velocity=None,
        sectional_thrust=dT_container,
        sectional_torque=dQ_container,
        total_thrust=thrust_container,
        total_torque=torque_container,
        total_power=power_container,
        efficiency=eta_container,
        figure_of_merit=FoM_container,
        thrust_coefficient=C_T_containter,
        torque_coefficient=C_Q_containter,
        power_coefficient=C_P_containter,
        residual=residual_container,
    )

    outputs.Cl = Cl
    outputs.Cd = Cd
    outputs.ux = ux
    outputs.state_vec = state_vec
    outputs.inflow = inflow_container
    outputs.sectional_lift = dL_container
    outputs.states = states_container
    outputs.alpha = alpha_container


    return outputs







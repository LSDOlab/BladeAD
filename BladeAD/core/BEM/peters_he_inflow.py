import numpy as np 
import csdl_alpha as csdl
import matplotlib.pyplot as plt
from BladeAD.utils.integration_schemes import integrate_quantity


def double_factorial(n):
    if n < 0:
        raise ValueError("Double factorial is not defined for negative integers.")
    elif n == 0 or n == 1:
        return 1
    else:
        return n * double_factorial(n - 2)

def compute_X(chi):
    return csdl.tan(((chi / 2)**2)**0.5)

def compute_chi(mu, lam):
    return np.arctan(((mu / lam)**2)**0.5)

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


# def compute_L(mat, r, m, x=1, type='cos'):
#     list = []
#     vec_ind = []
#     for r_ind in range(r):
#         for m_ind in range(m):
#             if type == 'cos':
#                 sub_mat = np.array([
#                     [np.str_(f"{r_ind}_{m_ind}_{2 * r_ind + 1 - r_ind}_{2 * m_ind + 1 - m_ind}"), np.str_(f"{r_ind}_{m_ind}_{2 * r_ind + 1 - r_ind}_{2 * (m_ind+1) + 1 - m_ind}")],
#                     [np.str_(f"{r_ind}_{m_ind}_{2 * r_ind + 1 - r_ind + 2}_{2 * m_ind + 1 - m_ind}"), np.str_(f"{r_ind}_{m_ind}_{2 * r_ind + 1 - r_ind + 2}_{2 * (m_ind+1) + 1 - m_ind}")],
#                 ])
#                 if r_ind == 0:
#                     sub_mat_2 = x**m_ind * csdl.Variable(shape=(2, 2), value=np.array([
#                         [compute_gamma(r_ind, m_ind, 2 * r_ind + 1 - r_ind, 2 * m_ind + 1 - m_ind), compute_gamma(r_ind, m_ind, 2 * r_ind + 1 - r_ind, 2 * (m_ind+1) + 1 - m_ind)],
#                         [compute_gamma(r_ind, m_ind, 2 * r_ind + 1 - r_ind + 2, 2 * m_ind + 1 - m_ind), compute_gamma(r_ind, m_ind, 2 * r_ind + 1 - r_ind + 2, 2 * (m_ind+1) + 1 - m_ind)]
#                     ]))
#                 else:
#                     sub_mat_2 =  (x**(((m_ind - r_ind)**2)**0.5) + (-1)**min(r_ind, m_ind) * x**(m_ind +  r_ind)) * csdl.Variable(shape=(2, 2), value=np.array([
#                         [compute_gamma(r_ind, m_ind, 2 * r_ind + 1 - r_ind, 2 * m_ind + 1 - m_ind), compute_gamma(r_ind, m_ind, 2 * r_ind + 1 - r_ind, 2 * (m_ind+1) + 1 - m_ind)],
#                         [compute_gamma(r_ind, m_ind, 2 * r_ind + 1 - r_ind + 2, 2 * m_ind + 1 - m_ind), compute_gamma(r_ind, m_ind, 2 * r_ind + 1 - r_ind + 2, 2 * (m_ind+1) + 1 - m_ind)]
#                     ]))

#                 list.append(sub_mat)

#                 row_ind_1 = 2 * r_ind
#                 row_ind_2 = row_ind_1 + 2
#                 col_ind_1 = 2 * m_ind
#                 col_ind_2 = col_ind_1 + 2
#                 if mat is None:
#                     pass
#                 else:
#                     mat = mat.set(csdl.slice[row_ind_1:row_ind_2, col_ind_1: col_ind_2], value=sub_mat_2)

#                 if m_ind == 0:
#                     vec_ind.append((r_ind, 2 * r_ind + 1 - r_ind))
#                     vec_ind.append((r_ind, 2 * r_ind + 1 - r_ind + 2))
#             else:
#                 sub_mat = np.array([
#                     [np.str_(f"{r_ind+1}_{m_ind+1}_{2 * r_ind + 1 - r_ind+1}_{2 * m_ind + 1 - m_ind+1}"), np.str_(f"{r_ind+1}_{m_ind+1}_{2 * r_ind + 1 - r_ind+ 1}_{2 * (m_ind+1) + 1 - m_ind+1}")],
#                     [np.str_(f"{r_ind+1}_{m_ind+1}_{2 * r_ind + 1 - r_ind + 2+1}_{2 * m_ind + 1 - m_ind+1}"), np.str_(f"{r_ind+1}_{m_ind+1}_{2 * r_ind + 1 - r_ind + 2+1}_{2 * (m_ind+1) + 1 - m_ind+1}")],
#                 ])
#                 # if r_ind == 0:
#                 #     sub_mat_2 = x**m_ind * csdl.Variable(shape=(2, 2), value=np.array([
#                 #         [compute_gamma(r_ind+1, m_ind+1, 2 * r_ind + 1 - r_ind+1, 2 * m_ind + 1 - m_ind+1), compute_gamma(r_ind+1, m_ind+1, 2 * r_ind + 1 - r_ind+1, 2 * (m_ind+1) + 1 - m_ind+1)],
#                 #         [compute_gamma(r_ind+1, m_ind+1, 2 * r_ind + 1 - r_ind + 2+1, 2 * m_ind + 1 - m_ind+1), compute_gamma(r_ind+1, m_ind+1, 2 * r_ind + 1 - r_ind + 2+1, 2 * (m_ind+1) + 1 - m_ind+1)]
#                 #     ]))
#                 # else:
#                 sub_mat_2 =  (x**(((m_ind - r_ind)**2)**0.5) - (-1)**min(r_ind, m_ind) * x**(m_ind +  r_ind)) * csdl.Variable(shape=(2, 2), value=np.array([
#                     [compute_gamma(r_ind+1, m_ind+1, 2 * r_ind + 1 - r_ind+1, 2 * m_ind + 1 - m_ind+1), compute_gamma(r_ind+1, m_ind+1, 2 * r_ind + 1 - r_ind+1, 2 * (m_ind+1) + 1 - m_ind+1)],
#                     [compute_gamma(r_ind+1, m_ind+1, 2 * r_ind + 1 - r_ind + 2+1, 2 * m_ind + 1 - m_ind+1), compute_gamma(r_ind+1, m_ind+1, 2 * r_ind + 1 - r_ind + 2+1, 2 * (m_ind+1) + 1 - m_ind+1)]
#                 ]))

#                 list.append(sub_mat)

#                 row_ind_1 = 2 * r_ind
#                 row_ind_2 = row_ind_1 + 2
#                 col_ind_1 = 2 * m_ind
#                 col_ind_2 = col_ind_1 + 2
#                 if mat is None:
#                     pass
#                 else:
#                     mat = mat.set(csdl.slice[row_ind_1:row_ind_2, col_ind_1: col_ind_2], value=sub_mat_2)

#                 if m_ind == 0:
#                     vec_ind.append((r_ind+1, 2 * r_ind + 1 - r_ind+1))
#                     vec_ind.append((r_ind+1, 2 * r_ind + 1 - r_ind + 2+1))

#     return mat, list, vec_ind

def compute_L(Q, r_harmonic, m_harmonic, x=2, type='cos'):
    if Q % 2 == 0:
        sub_mat_size = int((Q + 2) / 2)
    else:
        sub_mat_size = int((Q + 1) / 2)

    if type == 'sin':
        if isinstance(x, csdl.Variable):
            shape = ((r_harmonic - 1)* sub_mat_size, (m_harmonic-1) * sub_mat_size)
            L_mat = csdl.Variable(shape=shape, value=np.zeros(shape))
        else:
            L_mat = np.zeros(((r_harmonic - 1)* sub_mat_size, (m_harmonic-1) * sub_mat_size))
    else:
        if isinstance(x, csdl.Variable):
            shape = (r_harmonic * sub_mat_size, m_harmonic * sub_mat_size)
            L_mat = csdl.Variable(shape=shape, value=np.zeros(shape))
        else:
            L_mat = np.zeros((r_harmonic * sub_mat_size, m_harmonic * sub_mat_size))

    vec_ind_list = []
    for r in range(r_harmonic):
        if type == 'sin':
            r += 1
        if r > (r_harmonic-1):
            break
        for m in range(m_harmonic):
            if type == 'sin':
                m += 1
            if m > (m_harmonic-1):
                break
            if r == 0:
                multiplier = x**m
                print("mult", multiplier)
            else:
                if type == 'sin':
                    multiplier = x**(((m - r)**2)**0.5) - (-1)**min(r, m) * x**(m + r)
                    print("mult", multiplier)
                else:
                    multiplier = x**(((m - r)**2)**0.5) + (-1)**min(r, m) * x**(m + r)
                    print("mult", multiplier)
            if isinstance(x, csdl.Variable):
                sub_mat = csdl.Variable(shape=(sub_mat_size, sub_mat_size), value=0)
            else:
                sub_mat = np.zeros((sub_mat_size, sub_mat_size))
            for j in range(sub_mat_size):
                for n  in range(sub_mat_size):
                    if isinstance(x, csdl.Variable):
                        sub_mat = sub_mat.set(
                            slices=csdl.slice[j, n], 
                            value=multiplier * compute_gamma(r, m, r + 2 * j + 1, m + 2 * n + 1),
                        )
                    else:
                        sub_mat[j, n] = multiplier * compute_gamma(r, m, r + 2 * j + 1, m + 2 * n + 1)
            #         print(r, m, r + 2 * j + 1, m + 2 * n + 1)
            # print("submat", sub_mat)

            if type == 'sin':
                row_ind_1 = sub_mat_size * (r-1)
                row_ind_2 = row_ind_1 + sub_mat_size
                col_ind_1 = sub_mat_size * (m-1)
                col_ind_2 = col_ind_1 + sub_mat_size
                if isinstance(x, csdl.Variable):
                    L_mat = L_mat.set(
                        slices=csdl.slice[row_ind_1:row_ind_2, col_ind_1:col_ind_2],
                        value=sub_mat
                    )
                else:   
                    L_mat[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = sub_mat
    
            else:
                row_ind_1 = sub_mat_size * r
                row_ind_2 = row_ind_1 + sub_mat_size
                col_ind_1 = sub_mat_size * m
                col_ind_2 = col_ind_1 + sub_mat_size
                if isinstance(x, csdl.Variable):
                    L_mat = L_mat.set(
                        slices=csdl.slice[row_ind_1:row_ind_2, col_ind_1:col_ind_2],
                        value=sub_mat
                    )
                else:   
                    L_mat[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = sub_mat



            if type == 'sin':
                if m == 1:
                    for vec_ind in range(sub_mat_size):
                        vec_ind_list.append((r, r + 2 * vec_ind + 1))
            else:
                if m == 0:
                    for vec_ind in range(sub_mat_size):
                        vec_ind_list.append((r, r + 2 * vec_ind + 1))
                        # vec_ind.append((r, r + 1))
                        # vec_ind.append((r, r + 3))

            # print("\n")

            # L_mat[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = sub_mat

    return L_mat, vec_ind_list


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

def compute_phi(r, j, radius, Q, print_var=False):
    term_1 = ((2 * j + 1) * compute_h(j, r))**0.5
    if print_var:
        pass
        # print("term_1", term_1, "r", r, "j", j)
    summation = csdl.Variable(shape=radius.shape, value=0)
    for q in range(r, j, 2):
        if (q > Q) or (q > (j - 1)):
            break
        print("q", q, j)
        num_1 = (-1)**((q - r) / 2)
        num_2 = double_factorial(j + q)
        den_1 = double_factorial(q - r)
        den_2 = double_factorial(q + r)
        den_3 = double_factorial(j - q - 1)
        if print_var:
            pass
            # print((radius**q * num_1 * num_2 / (den_1 *  den_2 * den_3)).value)
        summation = summation + radius**q * num_1 * num_2 / (den_1 *  den_2 * den_3)
        # print("summation", num_1 * num_2 / (den_1 *  den_2 * den_3))
        # print("summation",  (term_1 * radius**q * num_1 * num_2 / (den_1 *  den_2 * den_3)).value)
        # print("hello", (radius**q * num_1 * num_2 / (den_1 *  den_2 * den_3)).value)
    if print_var: 
        pass
        # print(summation.value)
    
    return term_1 * summation


def solve_for_steady_state_inflow(
    shape,
    r_norm,
    psi,
    rpm,
    radius,
    mu_z,
    mu,
    tangential_velocity,
    frame_velocity,
    radius_vec,
    chord_profile,
    twist_profile,
    airfoil_model,
    atmos_states,
    num_blades,
    dr,
    integration_scheme, 
):
    # Pre-processing
    num_nodes = shape[0]
    num_radial = shape[1]
    num_azimuthal = shape[2]

    B = num_blades
    omega = rpm * 2 * np.pi / 60
    Vt = tangential_velocity
    Vx = frame_velocity[:, :, :, 0]
    Vr = (Vx**2 + Vt**2) ** 0.5
    mu_atmos = atmos_states.dynamic_viscosity
    rho = atmos_states.density
    a = atmos_states.speed_of_sound
    Re = rho * Vr * chord_profile / mu_atmos
    Ma = Vr / a

    
    Q = 3
    harmonics_r = 3
    harmonics_m = 3

    _, vec_ind_sin = compute_L(Q, harmonics_r, harmonics_m, type="sin")
    _, vec_ind_cos = compute_L(Q, harmonics_m, harmonics_m, type="cos")

    ns_sin = len(vec_ind_sin)
    ns_cos = len(vec_ind_cos)

    state_vec = csdl.ImplicitVariable(shape=((ns_cos + ns_sin, )), value=-0)
    inflow = csdl.Variable(shape=(num_radial, num_azimuthal), value=0)

    for nsc in range(ns_cos):
        r = vec_ind_cos[nsc][0]
        j = vec_ind_cos[nsc][1]

        phi_vec = csdl.reshape(compute_phi(r, j, radius_vec / radius, Q), shape=(num_radial, num_azimuthal))
        cos_psi = csdl.reshape(csdl.cos(r * psi), shape=(num_radial, num_azimuthal))

        inflow = inflow + phi_vec * state_vec[nsc] * cos_psi

    for nss in range(ns_sin):
        r = vec_ind_sin[nss][0]
        j = vec_ind_sin[nss][1]
        
        phi_vec = csdl.reshape(compute_phi(r, j, radius_vec / radius, Q), shape=(num_radial, num_azimuthal))
        sin_psi = csdl.reshape(csdl.sin(r * psi), shape=(num_radial, num_azimuthal))
        inflow = inflow + phi_vec * state_vec[nss+ns_cos] * sin_psi

    ux = (inflow + mu_z) * omega * csdl.reshape(radius_vec, (num_radial, num_azimuthal))
    
    phi = csdl.arctan(ux / csdl.reshape(Vt, (num_radial, num_azimuthal)))
    
    # Compute sectional Cl, Cd
    alpha = csdl.reshape(twist_profile, (num_radial, num_azimuthal)) - phi
    Cl, Cd = airfoil_model.evaluate(alpha, Re, Ma)

    tau_cos = csdl.Variable(shape=(ns_cos, ),value=0)
    tau_sin = csdl.Variable(shape=(ns_sin, ),value=0)

    dT = 0.5 * B * rho * (ux[:, :]**2 + csdl.reshape(Vt[:, :, :], (num_radial, num_azimuthal))**2) * csdl.reshape(chord_profile[:, :, :], (num_radial, num_azimuthal))  \
                    * (csdl.reshape(Cl[:, :], (num_radial, num_azimuthal )) * csdl.cos(phi[:, :]) - csdl.reshape(Cd[:, :], (num_radial, num_azimuthal )) * csdl.sin(phi[:, :])) * dr


    thrust = csdl.sum(dT) / num_azimuthal

    C_T = thrust / (rho * np.pi * radius**2 * (omega * radius) ** 2)

    
    for nsc in range(ns_cos):
        m = vec_ind_cos[nsc][0]
        n = vec_ind_cos[nsc][1]

        phi_cos = compute_phi(m, n, radius_vec[0, :, :]/radius, Q, print_var=False)
        L_cos = dT * phi_cos

        if m == 0:
            pass
        else:
            L_cos = L_cos * csdl.cos(m * psi.reshape((num_radial, num_azimuthal))) 

        tau_cos = tau_cos.set(
            slices=csdl.slice[nsc], value=csdl.sum(L_cos) / num_azimuthal
        )

    for nss in range(ns_sin):
        m = vec_ind_sin[nss][0]
        n = vec_ind_sin[nss][1]  

        phi_sin = compute_phi(m, n, radius_vec[0, :, :]/radius, Q, print_var=False)
        L_sin = dT * phi_sin

        L_sin = L_sin * csdl.sin(m * psi.reshape((num_radial, num_azimuthal))) 

        tau_sin = tau_sin.set(
            slices=csdl.slice[nss], value=csdl.sum(L_sin) / num_azimuthal 
        )

    tau_cos = tau_cos / (2 * np.pi * rho * omega**2 * radius**4) #csdl.sum(tau_cos / (2 * np.pi * rho * omega**2 * radius**4)) / num_azimuthal
    tau_sin = tau_sin / (2 * np.pi * rho * omega**2 * radius**4) #csdl.sum(tau_sin / (2 * np.pi * rho * omega**2 * radius**4)) / num_azimuthal


    tau = csdl.Variable(shape=(ns_cos + ns_sin, ), value=0)
    tau = tau.set(slices=csdl.slice[0:ns_cos], value=tau_cos)
    tau = tau.set(slices=csdl.slice[ns_cos:], value=tau_sin)

    lam_0 = csdl.ImplicitVariable(shape=(1, ), value=0.1)
    lam_0_res = lam_0 - C_T / (2 * (mu**2 + lam_0**2)**0.5)
    d_lam0res_d_lam0 = 1 + C_T / 2 * (mu**2 + lam_0**2)**(-3/2) * lam_0

    lam_0_solver = csdl.nonlinear_solvers.GaussSeidel()
    lam_0_solver.add_state(lam_0, lam_0_res, state_update= lam_0 - lam_0_res / d_lam0res_d_lam0)
    lam_0_solver.run()
    lam_m = lam_0 * 3**0.5
    lam = lam_m

    chi = csdl.arctan(mu / lam)
    x = compute_X(chi)
    V_eff = (mu**2 + lam * (lam + lam_m)) / (mu ** 2 + lam**2) ** 0.5
    V_eff_t = (mu ** 2 + lam**2) ** 0.5

    L_cos, vec_ind_cos = compute_L(Q, harmonics_r, harmonics_m, x, type='cos') 
    L_sin, vec_ind_sin = compute_L(Q, harmonics_r, harmonics_m, x, type='sin') 

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
    
    L_cos = csdl.matmat(L_cos, V_cos)  # L_cos / V_eff
    L_sin = csdl.matmat(L_sin, V_sin) # L_sin / V_eff
    L_block = csdl.Variable(shape=((ns_cos + ns_sin, ns_cos + ns_sin)), value=0)
    L_block = L_block.set(slices=csdl.slice[0:ns_cos, 0:ns_cos], value=L_cos)
    L_block = L_block.set(slices=csdl.slice[ns_cos:, ns_cos:], value=L_sin)


    residual = csdl.matvec(L_block, tau) - state_vec

    state_update = 0.1 * (csdl.matvec(L_block, tau) - state_vec)

    solver = csdl.nonlinear_solvers.GaussSeidel(max_iter=500)
    # solver.add_state(state_vec, residual, state_update=state_vec + 0.01 * residual)
    solver.add_state(state_vec, residual, state_update=state_vec + state_update)
    solver.run()

    # print(state_vec.value)
    # print(tau_cos.value)
    print("\n")
    print(residual.value)

    r = radius_vec[0, :, 0].value
    psi_ = psi[0, 0, :].value

    Psi, R = np.meshgrid(psi_, r)
    # dT = (0.5 * rho * B* (csdl.reshape(Vt, (num_radial, num_azimuthal))**2 + ux**2) * csdl.reshape(chord_profile, (num_radial, num_azimuthal)) * Cl * dr) # ux #lam_exp_i
    # dT =  0.5 * B * rho * (csdl.reshape(Vt, (num_radial, num_azimuthal))**2 + ux**2) * Cl * dr # ux #
    dT = 0.5 * B * rho * (ux**2 + csdl.reshape(Vt, (num_radial, num_azimuthal))**2) * csdl.reshape(chord_profile, (num_radial, num_azimuthal))  \
                    * (Cl * csdl.cos(phi) - Cd * csdl.sin(phi)) * dr
    # dT = inflow #= csdl.sum(phi_vec_cos * state_vec_exp[0:6, :, :] * cos_psi_exp + phi_vec_sin * state_vec_exp[6:, :, :] * sin_psi_exp, axes=(0, ))
    
    Psi = Psi - np.pi / 2
    
    print("inflow", inflow.value[20, 20])
    print((csdl.sum(dT)/num_azimuthal).value)
    max_idx = np.unravel_index(np.argmax(dT.value.reshape(num_radial, num_azimuthal)), dT.value.reshape(num_radial, num_azimuthal).shape)
    min_idx = np.unravel_index(np.argmin(dT.value.reshape(num_radial, num_azimuthal)), dT.value.reshape(num_radial, num_azimuthal).shape)


    plt.figure(figsize=(8, 6))
    plt.polar(Psi, R, alpha=0.)  # Plot the polar grid
    plt.pcolormesh(Psi, R,  dT.value[:, :])  # Plot the color map representing thrust
    plt.colorbar(label='Rotor Thrust')
    plt.title('Azimuthal and Radial Variation of Rotor Thrust')
    plt.plot(Psi[max_idx], R[max_idx], 'ro', label='Max Thrust')
    plt.plot(Psi[min_idx], R[min_idx], 'bo', label='Min Thrust')
    # plt.set_theta_zero_location("S")

    # Add contour lines
    contour_levels = np.linspace(np.min(dT.value), np.max(dT.value), 15)  # Adjust number of contour levels as needed
    plt.contour(Psi, R, dT.value[:, :], levels=contour_levels, colors='black', linestyles='dashed')

    plt.figure()
    plt.plot((r / radius.value).flatten(), dT[:, 30].value)

    # print(np.rad2deg(Psi[max_idx]))
    # print(np.rad2deg(Psi[min_idx]))
    # print(mu_z.value)
    print(state_vec.value)

    plt.show()


    return state_vec







import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass

@dataclass
class PittPetersImplicitOutputs(csdl.VariableGroup):
    phi: csdl.Variable
    Cl: csdl.Variable
    Cd: csdl.Variable
    ux: csdl.Variable


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

    # Initialize implicit variables
    state_vec = csdl.ImplicitVariable(shape=(3, ), value=0.01*np.ones((3, )))

    if num_nodes == 1:
        lam_0_exp = csdl.expand(state_vec[0], shape)
        lam_c_exp = csdl.expand(state_vec[1], shape)
        lam_s_exp = csdl.expand(state_vec[2], shape)
    else:
        lam_0_exp = csdl.expand(state_vec[:, 0], shape, action="i->ijk")
        lam_c_exp = csdl.expand(state_vec[:, 1], shape, action="i->ijk")
        lam_s_exp = csdl.expand(state_vec[:, 2], shape, action="i->ijk")

    # initial guess of normalized inflow
    lam_exp_i = (
        lam_0_exp
        + lam_c_exp * radius_vec / radius * csdl.cos(psi)
        + lam_s_exp * radius_vec / radius * csdl.sin(psi)
    )

    # compute axial-induced velocity
    ux = (lam_exp_i + mu_z) * omega * radius_vec

    # compute inflow angle (ignoring tangential induction)
    phi = csdl.arctan(ux / Vt)

    # Compute sectional Cl, Cd
    alpha = twist_profile - phi
    Cl, Cd = airfoil_model.evaluate(alpha, Re, Ma)

    # Compute sectional thrust, torque, moments and coefficients
    dT = 0.5 * B * rho * (ux**2 + Vt**2) * chord_profile * (Cl * csdl.cos(phi) - Cd * csdl.sin(phi)) * dr
    dQ = 0.5 * B * rho * (ux**2 + Vt**2) * chord_profile * (Cl * csdl.sin(phi) + Cd * csdl.cos(phi)) * radius_vec * dr
    dMx = radius_vec * csdl.sin(psi) * dT
    dMy = radius_vec * csdl.cos(psi) * dT

    thrust = csdl.Variable(shape=(num_nodes, ), value=0.)
    torque = csdl.Variable(shape=(num_nodes, ), value=0.)
    Mx = csdl.Variable(shape=(num_nodes, ), value=0.)
    My = csdl.Variable(shape=(num_nodes, ), value=0.)

    if integration_scheme == 'trapezoidal':
        for i in csdl.frange(num_nodes):
            # for j in csdl.frange(num_radial-1):
            thrust = thrust.set(
                csdl.slice[i], csdl.sum((dT[i, 0, :] + dT[i, -1, :]) / 2)  + csdl.sum(dT[i, 1:-1, :]) 
            )
            torque = torque.set(
                csdl.slice[i], csdl.sum((dQ[i, 0, :] + dQ[i, -1, :]) / 2)  + csdl.sum(dQ[i, 1:-1, :]) 
            )
            Mx = Mx.set(
                csdl.slice[i], csdl.sum((dMx[i, 0, :] + dMx[i, -1, :]) / 2)  + csdl.sum(dMx[i, 1:-1, :]) 
            )
            My = My.set(
                csdl.slice[i], csdl.sum((dMy[i, 0, :] + dMy[i, -1, :]) / 2)  + csdl.sum(dMy[i, 1:-1, :])
            )
        thrust =  thrust / num_azimuthal
        torque = torque / num_azimuthal
        Mx = Mx / num_azimuthal
        My = My / num_azimuthal

    elif integration_scheme == 'Simpson':
        for i in csdl.frange(num_nodes):
            value_thrust = csdl.sum(dT[i, 0, :]) + csdl.sum(dT[i, -1, :]) + 4 * csdl.sum(dT[i, 1:-1:2, :]) + 2 * csdl.sum(dT[i, 2:-1:2, :])
            thrust = thrust.set(
                slices=csdl.slice[i],
                value=value_thrust
            )

            value_torque = csdl.sum(dQ[i, 0, :] + dQ[i, -1, :]) + 4 * csdl.sum(dQ[i, 1:-1:2, :]) + 2 * csdl.sum(dQ[i, 2:-1:2, :])
            torque = torque.set(
                slices=csdl.slice[i],
                value=value_torque
            )

            value_Mx = csdl.sum(dMx[i, 0, :] + dMx[i, -1, :]) + 4 * csdl.sum(dMx[i, 1:-1:2, :]) + 2 * csdl.sum(dMx[i, 2:-1:2, :])
            Mx = Mx.set(
                slices=csdl.slice[i],
                value=value_Mx
            )

            value_My = csdl.sum(dMy[i, 0, :] + dMy[i, -1, :]) + 4 * csdl.sum(dMy[i, 1:-1:2, :]) + 2 * csdl.sum(dMy[i, 2:-1:2, :])
            My = My.set(
                slices=csdl.slice[i],
                value=value_My
            )

        thrust =  thrust / 3 / num_azimuthal
        torque = torque / 3 / num_azimuthal
        Mx = Mx / 3/ num_azimuthal
        My = My / 3/ num_azimuthal

    elif integration_scheme == 'Riemann':
        thrust = csdl.sum(dT, axes=(1, 2)) / num_azimuthal
        torque = csdl.sum(dQ, axes=(1, 2)) / num_azimuthal
        Mx = csdl.sum(dMx, axes=(1, 2)) / num_azimuthal
        My = csdl.sum(dMy, axes=(1, 2)) / num_azimuthal
    
    else:
        raise NotImplementedError("Unknown integration scheme.")
    

    for i in csdl.frange(num_nodes):
        C_T = thrust[i] / (rho * np.pi * radius**2 * (omega[i] * radius) ** 2)
        C_Mx = Mx[i] / (rho * np.pi * radius**2 * (omega[i] * radius) ** 2 * radius)
        C_My = My[i] / (rho * np.pi * radius**2 * (omega[i] * radius) ** 2 * radius)

        lam_0 = csdl.ImplicitVariable(shape=(1, ), value=0.1)
        lam_0_res = lam_0 - C_T / (2 * (mu**2 + lam_0**2)**0.5)
        d_lam0res_d_lam0 = 1 + C_T / 2 * (mu**2 + lam_0**2)**(-3/2) * lam_0

        lam_0_solver = csdl.nonlinear_solvers.GaussSeidel()
        lam_0_solver.add_state(lam_0, lam_0_res, state_update= lam_0 - lam_0_res / d_lam0res_d_lam0)
        lam_0_solver.run()

        lam_m = lam_0 * 3**0.5
        lam = lam_m

        # lam_i = csdl.average(lam_exp_i[i, :, :])
        # lam = lam_i + mu_z[i]

        chi = csdl.arctan(mu[i] / lam)
        V_eff = (mu[i] ** 2 + lam * (lam + lam_m)) / (mu[i] ** 2 + lam**2) ** 0.5
        # V_eff = (mu[i] ** 2 + lam**2) ** 0.5

        L_mat = csdl.Variable(shape=(3, 3), value=np.zeros((3, 3)))
        L_mat = L_mat.set(csdl.slice[0, 0], 0.5)
        L_mat = L_mat.set(csdl.slice[0, 1], -15 * np.pi / 64 * ((1 - csdl.cos(chi)) / (1 + csdl.cos(chi))) ** 0.5)
        L_mat = L_mat.set(csdl.slice[1, 1], 4 * csdl.cos(chi) / (1 + csdl.cos(chi)))
        L_mat = L_mat.set(csdl.slice[1, 2], -1 * (
            -15 * np.pi / 64 * ((1 - csdl.cos(chi)) / (1 + csdl.cos(chi))) ** 0.5
        ))
        L_mat = L_mat.set(csdl.slice[2, 2], 4 / (1 + csdl.cos(chi)))

        L_mat_new = L_mat / V_eff

        rhs = csdl.Variable(shape=(3, ), value=np.zeros((3, )))
        rhs = rhs.set(csdl.slice[0], C_T)
        rhs = rhs.set(csdl.slice[1], C_My)
        rhs = rhs.set(csdl.slice[2], C_Mx)

        residual = state_vec - csdl.matvec(L_mat_new, rhs)

    # print(state_vec.value)
    # print(residual.value)
    print(thrust.value)
    print(torque.value)

    solver = csdl.nonlinear_solvers.GaussSeidel(max_iter=100)
    
    state_update = 0.5 * (csdl.matvec(L_mat_new, rhs) - state_vec)

    solver.add_state(state_vec, residual=residual, state_update=state_vec + state_update)
    solver.run()
    # print(state_vec.value)
    # print(ux.value)
    # print(lam_i.value)
    # print(lam_i_2.value)
    print(thrust.value)
    print(torque.value)
    n = rpm / 60
    C_T_2 = thrust / rho / n**2 / (2 * radius)**4
    C_Q = torque / rho / n**2 / (2 * radius)**5
    C_P = 2 * np.pi * C_Q
    J = Vx[0, 0, 0] / n /  (2 * radius)
    eta = C_T_2 * J / C_P

    print(eta.value)
    print(state_vec.value)
    print(thrust.value)


    import matplotlib.pyplot as plt
    r = radius_vec[0, :, 0].value
    psi_ = psi[0, 0, :].value

    Psi, R = np.meshgrid(psi_, r)

    print(chi.value * 180 / np.pi)
    # dT = lam_exp_i # ux #lam_exp_i
    # dT = 0.5 * B * rho * (Vt**2 + ux**2) * Cl * dr
    max_idx = np.unravel_index(np.argmax(dT.value.reshape(num_radial, num_azimuthal)), dT.value.reshape(num_radial, num_azimuthal).shape)
    min_idx = np.unravel_index(np.argmin(dT.value.reshape(num_radial, num_azimuthal)), dT.value.reshape(num_radial, num_azimuthal).shape)


    plt.figure(figsize=(8, 6))
    plt.polar(Psi, R, alpha=0.)  # Plot the polar grid
    plt.pcolormesh(Psi, R,  dT.value[0, :, :])  # Plot the color map representing thrust
    plt.colorbar(label='Rotor Thrust')
    plt.title('Azimuthal and Radial Variation of Rotor Thrust')
    plt.plot(Psi[max_idx], R[max_idx], 'ro', label='Max Thrust')
    plt.plot(Psi[min_idx], R[min_idx], 'bo', label='Min Thrust')
    # plt.xlabel('Azimuthal Angle (radians)')
    # plt.ylabel('Radial Distance')
    # plt.grid(True)
    print(np.rad2deg(Psi[max_idx]))
    print(np.rad2deg(Psi[min_idx]))
    print(mu_z.value)
    print(thrust.value)

    contour_levels = np.linspace(np.min(dT.value), np.max(dT.value), 10)  # Adjust number of contour levels as needed
    plt.contour(Psi, R, dT.value[0, :, :], levels=contour_levels, colors='black', linestyles='dashed')

    plt.show()

    return state_vec

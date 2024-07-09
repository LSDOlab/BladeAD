import csdl_alpha as csdl
import numpy as np
from BladeAD.utils.var_groups import RotorAnalysisOutputs
from BladeAD.utils.integration_schemes import integrate_quantity
from BladeAD.utils.smooth_quantities_spanwise import smooth_quantities_spanwise
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel
from BladeAD.utils.switching_fun import sigmoid


def solve_for_steady_state_inflow(
    shape,
    psi,
    rpm,
    radius,
    mu_z,
    mu,
    tangential_velocity,
    frame_velocity,
    norm_radius_vec,
    radius_vec,
    chord_profile,
    twist_profile,
    airfoil_model,
    disk_inclination_angle,
    # atmos_states,
    mu_atmos, 
    rho,
    a,
    num_blades,
    dr,
    hub_radius, 
    radius_vec_exp,
    integration_scheme, 
    tip_loss,
) -> RotorAnalysisOutputs:
    # Pre-processing
    num_nodes = shape[0]
    num_radial = shape[1]
    num_azimuthal = shape[2]

    B = num_blades
    omega = rpm * 2 * np.pi / 60
    Vt = tangential_velocity
    Vx = frame_velocity[:, :, :, 0]
    Vr = (Vx**2 + Vt**2) ** 0.5
    # mu_atmos = atmos_states.dynamic_viscosity
    # rho = atmos_states.density
    # a = atmos_states.speed_of_sound
    Re = rho * Vr * chord_profile / mu_atmos
    Ma = Vr / a

    # setting variables to store quantities of interest
    states_container = csdl.Variable(shape=(num_nodes, 3), value=0)
    ux_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    dT_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    dQ_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    dD_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    thrust_container = csdl.Variable(shape=(num_nodes, ), value=0)
    torque_container = csdl.Variable(shape=(num_nodes, ), value=0)
    residual_container = csdl.Variable(shape=(num_nodes, 3), value=0)
    C_T_containter = csdl.Variable(shape=(num_nodes, ), value=0)
    C_P_containter = csdl.Variable(shape=(num_nodes, ), value=0)
    C_Q_containter = csdl.Variable(shape=(num_nodes, ), value=0)
    eta_container = csdl.Variable(shape=(num_nodes, ), value=0)
    FoM_container = csdl.Variable(shape=(num_nodes, ), value=0)


    for i in range(num_nodes):
        # Initialize implicit variables (state vector)
        state_vec = csdl.ImplicitVariable(shape=(3, ), value=0.01*np.ones((3, )))

        lam_0_exp = csdl.expand(state_vec[0], (num_radial, num_azimuthal))
        lam_c_exp = csdl.expand(state_vec[1], (num_radial, num_azimuthal))
        lam_s_exp = csdl.expand(state_vec[2], (num_radial, num_azimuthal))

        # initial guess of normalized inflow
        lam_exp_i = (
            lam_0_exp
            + lam_c_exp * radius_vec[i, :, :] / radius * csdl.cos(psi[i, :, :])
            + lam_s_exp * radius_vec[i, :, :] / radius * csdl.sin(psi[i, :, :])
            # + lam_c_exp * norm_radius_vec[i, :, :] * csdl.cos(psi[i, :, :])
            # + lam_s_exp * norm_radius_vec[i, :, :] * csdl.sin(psi[i, :, :])
        )


        # compute axial-induced velocity
        ux = (lam_exp_i + mu_z[i]) * omega[i] * radius #radius_vec[i, :, :]
        # ux = (lam_exp_i) * omega[i] * radius # radius_vec[i, :, :] #

        # compute inflow angle (ignoring tangential induction)
        phi = csdl.arctan(ux / Vt[i, :, :])

        # Compute sectional Cl, Cd
        alpha = twist_profile[i, :, :] - phi
        Cl, Cd = airfoil_model.evaluate(alpha, Re[i, :, :], Ma[i, :, :])

        # Prandtl tip losses 
        f_tip = num_blades / 2 * (radius - radius_vec_exp[i, :, :]) / radius / csdl.sin(phi)
        f_hub = num_blades / 2 * (radius_vec_exp[i, :, :] - hub_radius) / hub_radius / csdl.sin(phi)

        F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-(f_tip**2)**0.5))
        F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-(f_hub**2)**0.5))

        F_initial = F_tip * F_hub
        if tip_loss:
            F = 1 + (F_initial - 1) * sigmoid(disk_inclination_angle[i], np.deg2rad(12))
        else:
            F = 1

        # Smooth the transition if airfoil model consists of multiple airfoils
        if isinstance(airfoil_model, CompositeAirfoilModel):
            if airfoil_model.smoothing:
                window = airfoil_model.transition_window
                indices = airfoil_model.stop_indices
                Cl = smooth_quantities_spanwise(Cl, indices, window)
                Cd = smooth_quantities_spanwise(Cd, indices, window)

        # Compute sectional thrust, torque, moments and coefficients
        Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
        Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)
        dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr * F
        dMx = radius_vec[i, :, :] * csdl.sin(psi[i, :, :]) * dT
        dMy = radius_vec[i, :, :] * csdl.cos(psi[i, :, :]) * dT

        thrust = integrate_quantity(dT, integration_scheme)
        Mx = integrate_quantity(dMx, integration_scheme)
        My = integrate_quantity(dMy, integration_scheme)

        C_T = thrust / (rho[i, 0, 0] * np.pi * radius**2 * (omega[i] * radius) ** 2)
        C_Mx = Mx / (rho[i, 0, 0] * np.pi * radius**2 * (omega[i] * radius) ** 2 * radius)
        C_My = My / (rho[i, 0, 0] * np.pi * radius**2 * (omega[i] * radius) ** 2 * radius)

        # Solver for lam_i (different from lam_exp_i)
        lam_i = csdl.ImplicitVariable(shape=(1, ), value=0.1)
        lam_i_res = lam_i - C_T / (2 * (mu[i]**2 + (mu_z[i] + lam_i)**2)**0.5)
        lam_i_solver = csdl.nonlinear_solvers.Newton(print_status=False)
        lam_i_solver.add_state(lam_i, lam_i_res)
        lam_i_solver.run()

        lam = mu_z[i] + lam_i

        # Compute wake skew and effective velocities
        chi = csdl.arctan((mu[i] +1e-5) / lam)
        V_eff = (mu[i] ** 2 + lam * (lam + lam_i)) / (mu[i] ** 2 + lam**2) ** 0.5
        V_eff_2 = (mu[i]**2 + lam**2)**0.5

        L_mat = csdl.Variable(shape=(3, 3), value=np.zeros((3, 3)))
        L_mat = L_mat.set(csdl.slice[0, 0], 0.5)
        L_mat = L_mat.set(csdl.slice[0, 1], -15 * np.pi / 64 * ((1 - csdl.cos(chi)) / (1 + csdl.cos(chi))) ** 0.5)
        L_mat = L_mat.set(csdl.slice[1, 1], 4 * csdl.cos(chi) / (1 + csdl.cos(chi)))
        L_mat = L_mat.set(csdl.slice[1, 2], -1 * (
            -15 * np.pi / 64 * ((1 - csdl.cos(chi)) / (1 + csdl.cos(chi))) ** 0.5
        ))
        L_mat = L_mat.set(csdl.slice[2, 2], 4 / (1 + csdl.cos(chi)))

        L_mat_new = L_mat 
        L_mat_new = L_mat_new.set(csdl.slice[:, 0], L_mat_new[:, 0]/ V_eff)
        L_mat_new = L_mat_new.set(csdl.slice[:, 1], L_mat_new[:, 1]/ V_eff_2)
        L_mat_new = L_mat_new.set(csdl.slice[:, 2], L_mat_new[:, 2]/ V_eff_2)

        rhs = csdl.Variable(shape=(3, ), value=np.zeros((3, )))
        rhs = rhs.set(csdl.slice[0], C_T)
        rhs = rhs.set(csdl.slice[1], -C_My)
        rhs = rhs.set(csdl.slice[2], C_Mx)

        # define residual based on steady state assumption
        residual = state_vec - csdl.matvec(L_mat_new, rhs)
    
        # basic fixed point iteration for state update
        state_update = 0.5 * (csdl.matvec(L_mat_new, rhs) - state_vec)

        solver = csdl.nonlinear_solvers.GaussSeidel(
            max_iter=100, 
            residual_jac_kwargs={'elementwise' : False, 'loop' : True}
        )
        solver.add_state(state_vec, residual=residual, state_update=state_vec + state_update)
        solver.run()

        dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr * F
        dQ = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Ct * radius_vec[i, :, :] * dr * F
        thrust = integrate_quantity(dT, integration_scheme)
        torque = integrate_quantity(dQ, integration_scheme)

        # compute thrust/torque/power coefficient
        CT = thrust / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius)**4
        CQ = torque / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius)**5
        CP = 2 * np.pi * CQ

        power = CP * rho[i, 0, 0] * (rpm[i] / 60)**3 * (2 * radius)**5


        # Compute advance ratio and efficiency and FOM
        J = Vx[i, 0, 0] / (rpm[i] / 60) / (2 * radius)
        eta = CT * J / CP
        FoM = CT * (CT/2)**0.5 / CP

        # Storing data
        ux_container = ux_container.set(csdl.slice[i, :, :], ux)
        dT_container = dT_container.set(csdl.slice[i, :, :], dT)
        dQ_container = dQ_container.set(csdl.slice[i, :, :], dQ)
        dD_container = dD_container.set(csdl.slice[i, :, :], dQ / radius_vec[i, :, :])
        thrust_container = thrust_container.set(csdl.slice[i], thrust)
        torque_container = torque_container.set(csdl.slice[i], torque)
        C_T_containter = C_T_containter.set(csdl.slice[i], CT)
        C_P_containter = C_P_containter.set(csdl.slice[i], CP)
        C_Q_containter = C_Q_containter.set(csdl.slice[i], CQ)
        eta_container = eta_container.set(csdl.slice[i], eta)
        FoM_container = FoM_container.set(csdl.slice[i], FoM)

        states_container = states_container.set(csdl.slice[i, :], state_vec)
        residual_container = residual_container.set(csdl.slice[i, :], residual)

    outputs = RotorAnalysisOutputs(
        axial_induced_velocity=ux_container,
        tangential_induced_velocity=None,
        sectional_thrust=dT_container,
        sectional_torque=dQ_container,
        sectional_drag=dD_container,
        total_thrust=thrust_container,
        total_torque=torque_container,
        total_power=power,
        efficiency=eta_container,
        figure_of_merit=FoM_container,
        thrust_coefficient=C_T_containter,
        torque_coefficient=C_Q_containter,
        power_coefficient=C_P_containter,
        residual=residual_container,
        )

    # outputs.Cl = Cl
    # outputs.Cd = Cd


    # # print(state_vec.value)
    # # print(residual.value)
    # print(thrust.value)
    # print(torque.value)

    
    # # print(state_vec.value)
    # # print(ux.value)
    # # print(lam_i.value)
    # # print(lam_i_2.value)
    # print(thrust.value)
    # print(torque.value)
    # n = rpm / 60
    # C_T_2 = thrust / rho / n**2 / (2 * radius)**4
    # C_Q = torque / rho / n**2 / (2 * radius)**5
    # C_P = 2 * np.pi * C_Q
    # J = Vx[0, 0, 0] / n /  (2 * radius)
    # eta = C_T_2 * J / C_P

    # print(eta.value)
    # print(state_vec.value)
    # print(thrust.value)


    # import matplotlib.pyplot as plt
    # r = radius_vec[0, :, 0].value
    # psi_ = psi[0, 0, :].value

    # Psi, R = np.meshgrid(psi_, r)

    # Psi = Psi - np.pi / 2

    # print(chi.value * 180 / np.pi)
    # # dT = lam_exp_i # ux #lam_exp_i
    # # dT = 0.5 * B * rho * (Vt**2 + ux**2) * Cl * dr
    # max_idx = np.unravel_index(np.argmax(dT.value.reshape(num_radial, num_azimuthal)), dT.value.reshape(num_radial, num_azimuthal).shape)
    # min_idx = np.unravel_index(np.argmin(dT.value.reshape(num_radial, num_azimuthal)), dT.value.reshape(num_radial, num_azimuthal).shape)


    # plt.figure(figsize=(8, 6))
    # plt.polar(Psi, R, alpha=0.)  # Plot the polar grid
    # plt.pcolormesh(Psi, R,  dT.value)  # Plot the color map representing thrust
    # plt.colorbar(label='Rotor Thrust')
    # plt.title('Azimuthal and Radial Variation of Rotor Thrust')
    # plt.plot(Psi[max_idx], R[max_idx], 'ro', label='Max Thrust')
    # plt.plot(Psi[min_idx], R[min_idx], 'bo', label='Min Thrust')
    # # plt.xlabel('Azimuthal Angle (radians)')
    # # plt.ylabel('Radial Distance')
    # # plt.grid(True)
    # print(np.rad2deg(Psi[max_idx]))
    # print(np.rad2deg(Psi[min_idx]))
    # print(mu_z.value)
    # print(thrust.value)

    # contour_levels = np.linspace(np.min(dT.value), np.max(dT.value), 10)  # Adjust number of contour levels as needed
    # plt.contour(Psi, R, dT.value, levels=contour_levels, colors='black', linestyles='dashed')

    # plt.show()

    return outputs

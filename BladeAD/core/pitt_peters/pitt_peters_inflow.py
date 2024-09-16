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
    tip_loss=1,
    mode="standard",
    use_frange_in_nl_solver=False,
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
    lambda_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
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
    inflow_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)
    phi_container = csdl.Variable(shape=(num_nodes, num_radial, num_azimuthal), value=0)


    if mode == "standard":
        if use_frange_in_nl_solver:
            for i in csdl.frange(num_nodes):
            # for i in range(num_nodes):
                # Initialize implicit variables (state vector)
                state_vec = csdl.ImplicitVariable(shape=(3, ), value=0.01*np.ones((3, )))

                lam_0_exp = csdl.expand(state_vec[0], (num_radial, num_azimuthal))
                lam_c_exp = csdl.expand(state_vec[1], (num_radial, num_azimuthal))
                lam_s_exp = csdl.expand(state_vec[2], (num_radial, num_azimuthal))

                # initial guess of normalized inflow
                lam_exp_i = (
                    lam_0_exp
                    + lam_c_exp * radius_vec[i, :, :] / radius[i, :, :] * csdl.cos(psi[i, :, :])
                    + lam_s_exp * radius_vec[i, :, :] / radius[i, :, :] * csdl.sin(psi[i, :, :])
                    # + lam_c_exp * norm_radius_vec[i, :, :] * csdl.cos(psi[i, :, :])
                    # + lam_s_exp * norm_radius_vec[i, :, :] * csdl.sin(psi[i, :, :])
                )


                # compute axial-induced velocity
                ux = (lam_exp_i + mu_z[i]) * omega[i] * radius[i, :, :] #radius_vec[i, :, :]
                # ux = (lam_exp_i) * omega[i] * radius # radius_vec[i, :, :] #

                # compute inflow angle (ignoring tangential induction)
                phi = csdl.arctan(ux / Vt[i, :, :])

                # Compute sectional Cl, Cd
                alpha = twist_profile[i, :, :] - phi
                Cl, Cd = airfoil_model.evaluate(alpha, Re[i, :, :], Ma[i, :, :])

                # Prandtl tip losses 
                f_tip = num_blades / 2 * (radius[i, :, :] - radius_vec_exp[i, :, :]) / radius[i, :, :] / csdl.sin(phi)
                f_hub = num_blades / 2 * (radius_vec_exp[i, :, :] - hub_radius[i, :, :]) / hub_radius[i, :, :] / csdl.sin(phi)

                F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-(f_tip**2)**0.5))
                F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-(f_hub**2)**0.5))

                F_initial = F_tip * F_hub
                F = tip_loss * F_initial + (1 - tip_loss) * 1

                # Smooth the transition if airfoil model consists of multiple airfoils
                # if isinstance(airfoil_model, CompositeAirfoilModel):
                #     if airfoil_model.smoothing:
                #         window = airfoil_model.transition_window
                #         indices = airfoil_model.stop_indices
                #         Cl = smooth_quantities_spanwise(Cl, indices, window)
                #         Cd = smooth_quantities_spanwise(Cd, indices, window)

                # Compute sectional thrust, torque, moments and coefficients
                Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
                Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)
                dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr[i, :, :] * F
                dMx = radius_vec[i, :, :] * csdl.sin(psi[i, :, :]) * dT
                dMy = -radius_vec[i, :, :] * csdl.cos(psi[i, :, :]) * dT

                thrust = integrate_quantity(dT, integration_scheme)
                Mx = integrate_quantity(dMx, integration_scheme)
                My = integrate_quantity(dMy, integration_scheme)

                C_T = thrust / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2)
                C_Mx = Mx / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2 * radius[i, 0, 0])
                C_My = My / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2 * radius[i, 0, 0])

                # Solver for lam_i (different from lam_exp_i)
                lam_i = csdl.ImplicitVariable(shape=(1, ), value=0.1)
                lam_i_res = lam_i - C_T / (2 * (mu[i]**2 + (mu_z[i] + lam_i)**2)**0.5)
                lam_i_solver = csdl.nonlinear_solvers.Newton(print_status=False)
                lam_i_solver.add_state(lam_i, lam_i_res)
                lam_i_solver.run()

                lam = mu_z[i] + csdl.average(lam_i)

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
                L_mat_new = L_mat_new.set(csdl.slice[:, 0], L_mat_new[:, 0]/ V_eff*2) 
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

                dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr[i, :, :] * F
                dQ = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Ct * radius_vec[i, :, :] * dr[i, :, :] * F
                thrust = integrate_quantity(dT, integration_scheme)
                torque = integrate_quantity(dQ, integration_scheme)

                # compute thrust/torque/power coefficient
                CT = thrust / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius[i, 0, 0])**4
                area = np.pi * radius[i, 0, 0]**2
                V_tip = rpm[i] / 60 * 2 * np.pi * radius[i, 0, 0]
                C_T_new = thrust / rho[i, 0, 0]/ area / V_tip**2
                CQ = torque / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius[i, 0, 0])**5
                CP = 2 * np.pi * CQ

                CT_H = CT * 4 / np.pi**3
                CQ_H = CQ * 8 / np.pi**3

                power = CP * rho[i, 0, 0] * (rpm[i] / 60)**3 * (2 * radius[i, 0, 0])**5


                # Compute advance ratio and efficiency and FOM
                J = Vx[i, 0, 0] / (rpm[i] / 60) / (2 * radius[i, 0, 0])
                eta = CT * J / CP
                # FoM = CT * (CT/2)**0.5 / CP
                FoM = CT_H * (CT_H/2)**0.5 / CQ_H

                # Storing data
                inflow_container = inflow_container.set(csdl.slice[i, :, :], lam_exp_i)
                ux_container = ux_container.set(csdl.slice[i, :, :], ux)
                dT_container = dT_container.set(csdl.slice[i, :, :], dT)
                dQ_container = dQ_container.set(csdl.slice[i, :, :], dQ)
                dD_container = dD_container.set(csdl.slice[i, :, :], dQ / radius_vec[i, :, :])
                phi_container = phi_container.set(csdl.slice[i, :, :], phi)
                thrust_container = thrust_container.set(csdl.slice[i], thrust)
                torque_container = torque_container.set(csdl.slice[i], torque)
                C_T_containter = C_T_containter.set(csdl.slice[i], C_T_new)
                C_P_containter = C_P_containter.set(csdl.slice[i], CP)
                C_Q_containter = C_Q_containter.set(csdl.slice[i], CQ)
                eta_container = eta_container.set(csdl.slice[i], eta)
                FoM_container = FoM_container.set(csdl.slice[i], FoM)

                states_container = states_container.set(csdl.slice[i, :], state_vec)
                residual_container = residual_container.set(csdl.slice[i, :], residual)

        else:
            for i in range(num_nodes):
                # Initialize implicit variables (state vector)
                state_vec = csdl.ImplicitVariable(shape=(3, ), value=0.01*np.ones((3, )))

                lam_0_exp = csdl.expand(state_vec[0], (num_radial, num_azimuthal))
                lam_c_exp = csdl.expand(state_vec[1], (num_radial, num_azimuthal))
                lam_s_exp = csdl.expand(state_vec[2], (num_radial, num_azimuthal))

                # initial guess of normalized inflow
                lam_exp_i = (
                    lam_0_exp
                    + lam_c_exp * radius_vec[i, :, :] / radius[i, :, :] * csdl.cos(psi[i, :, :])
                    + lam_s_exp * radius_vec[i, :, :] / radius[i, :, :] * csdl.sin(psi[i, :, :])
                    # + lam_c_exp * norm_radius_vec[i, :, :] * csdl.cos(psi[i, :, :])
                    # + lam_s_exp * norm_radius_vec[i, :, :] * csdl.sin(psi[i, :, :])
                )


                # compute axial-induced velocity
                ux = (lam_exp_i + mu_z[i]) * omega[i] * radius[i, :, :] #radius_vec[i, :, :]
                # ux = (lam_exp_i) * omega[i] * radius # radius_vec[i, :, :] #

                # compute inflow angle (ignoring tangential induction)
                phi = csdl.arctan(ux / Vt[i, :, :])

                # Compute sectional Cl, Cd
                alpha = twist_profile[i, :, :] - phi
                Cl, Cd = airfoil_model.evaluate(alpha, Re[i, :, :], Ma[i, :, :])

                # Prandtl tip losses 
                f_tip = num_blades / 2 * (radius[i, :, :] - radius_vec_exp[i, :, :]) / radius[i, :, :] / csdl.sin(phi)
                f_hub = num_blades / 2 * (radius_vec_exp[i, :, :] - hub_radius[i, :, :]) / hub_radius[i, :, :] / csdl.sin(phi)

                F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-(f_tip**2)**0.5))
                F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-(f_hub**2)**0.5))

                F_initial = F_tip * F_hub
                F = tip_loss * F_initial + (1 - tip_loss) * 1

                # Smooth the transition if airfoil model consists of multiple airfoils
                # if isinstance(airfoil_model, CompositeAirfoilModel):
                #     if airfoil_model.smoothing:
                #         window = airfoil_model.transition_window
                #         indices = airfoil_model.stop_indices
                #         Cl = smooth_quantities_spanwise(Cl, indices, window)
                #         Cd = smooth_quantities_spanwise(Cd, indices, window)

                # Compute sectional thrust, torque, moments and coefficients
                Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
                Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)
                dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr[i, :, :] * F
                dMx = radius_vec[i, :, :] * csdl.sin(psi[i, :, :]) * dT
                dMy = -radius_vec[i, :, :] * csdl.cos(psi[i, :, :]) * dT

                thrust = integrate_quantity(dT, integration_scheme)
                Mx = integrate_quantity(dMx, integration_scheme)
                My = integrate_quantity(dMy, integration_scheme)

                C_T = thrust / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2)
                C_Mx = Mx / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2 * radius[i, 0, 0])
                C_My = My / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2 * radius[i, 0, 0])

                # Solver for lam_i (different from lam_exp_i)
                lam_i = csdl.ImplicitVariable(shape=(1, ), value=0.1)
                lam_i_res = lam_i - C_T / (2 * (mu[i]**2 + (mu_z[i] + lam_i)**2)**0.5)
                lam_i_solver = csdl.nonlinear_solvers.Newton(print_status=False)
                lam_i_solver.add_state(lam_i, lam_i_res)
                lam_i_solver.run()

                lam = mu_z[i] + csdl.average(lam_i)

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
                L_mat_new = L_mat_new.set(csdl.slice[:, 0], L_mat_new[:, 0]/ V_eff*2) 
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

                dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr[i, :, :] * F
                dQ = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Ct * radius_vec[i, :, :] * dr[i, :, :] * F
                thrust = integrate_quantity(dT, integration_scheme)
                torque = integrate_quantity(dQ, integration_scheme)

                # compute thrust/torque/power coefficient
                CT = thrust / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius[i, 0, 0])**4
                area = np.pi * radius[i, 0, 0]**2
                V_tip = rpm[i] / 60 * 2 * np.pi * radius[i, 0, 0]
                C_T_new = thrust / rho[i, 0, 0]/ area / V_tip**2
                CQ = torque / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius[i, 0, 0])**5
                CP = 2 * np.pi * CQ

                CT_H = CT * 4 / np.pi**3
                CQ_H = CQ * 8 / np.pi**3

                power = CP * rho[i, 0, 0] * (rpm[i] / 60)**3 * (2 * radius[i, 0, 0])**5


                # Compute advance ratio and efficiency and FOM
                J = Vx[i, 0, 0] / (rpm[i] / 60) / (2 * radius[i, 0, 0])
                eta = CT * J / CP
                # FoM = CT * (CT/2)**0.5 / CP
                FoM = CT_H * (CT_H/2)**0.5 / CQ_H

                # Storing data
                inflow_container = inflow_container.set(csdl.slice[i, :, :], lam_exp_i)
                ux_container = ux_container.set(csdl.slice[i, :, :], ux)
                dT_container = dT_container.set(csdl.slice[i, :, :], dT)
                dQ_container = dQ_container.set(csdl.slice[i, :, :], dQ)
                dD_container = dD_container.set(csdl.slice[i, :, :], dQ / radius_vec[i, :, :])
                phi_container = phi_container.set(csdl.slice[i, :, :], phi)
                thrust_container = thrust_container.set(csdl.slice[i], thrust)
                torque_container = torque_container.set(csdl.slice[i], torque)
                C_T_containter = C_T_containter.set(csdl.slice[i], C_T_new)
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
        outputs.normalized_axial_induced_flow = lambda_container
        outputs.inflow = inflow_container
        outputs.sectional_inflow_angle = phi_container

        return outputs

    else:
        state_vec = csdl.ImplicitVariable(shape=(num_nodes * 3, ), value=0.01)
        rhs_block = csdl.Variable(shape=(num_nodes * 3, ), value=0.)
        L_block_mat = csdl.Variable(shape=(num_nodes * 3, num_nodes *3), value=0.)

        for i in csdl.frange(num_nodes):
        # for i in range(num_nodes):
            # print(i.value)
            # Initialize implicit variables (state vector)
            num_node_index = i * 3 

            lam_0_exp = csdl.expand(state_vec[num_node_index], (num_radial, num_azimuthal))
            lam_c_exp = csdl.expand(state_vec[num_node_index+1], (num_radial, num_azimuthal))
            lam_s_exp = csdl.expand(state_vec[num_node_index+2], (num_radial, num_azimuthal))

            # initial guess of normalized inflow
            lam_exp_i = (
                lam_0_exp
                + lam_c_exp * radius_vec[i, :, :] / radius[i, :, :] * csdl.cos(psi[i, :, :])
                + lam_s_exp * radius_vec[i, :, :] / radius[i, :, :] * csdl.sin(psi[i, :, :])
                # + lam_c_exp * norm_radius_vec[i, :, :] * csdl.cos(psi[i, :, :])
                # + lam_s_exp * norm_radius_vec[i, :, :] * csdl.sin(psi[i, :, :])
            )


            # compute axial-induced velocity
            ux = (lam_exp_i + mu_z[i]) * omega[i] * radius[i, :, :] #radius_vec[i, :, :]
            # ux = (lam_exp_i) * omega[i] * radius # radius_vec[i, :, :] #

            # compute inflow angle (ignoring tangential induction)
            phi = csdl.arctan(ux / Vt[i, :, :])

            # Compute sectional Cl, Cd
            alpha = twist_profile[i, :, :] - phi
            Cl, Cd = airfoil_model.evaluate(alpha, Re[i, :, :], Ma[i, :, :])

            # Prandtl tip losses 
            f_tip = num_blades / 2 * (radius[i, :, :] - radius_vec_exp[i, :, :]) / radius[i, :, :] / csdl.sin(phi)
            f_hub = num_blades / 2 * (radius_vec_exp[i, :, :] - hub_radius[i, :, :]) / hub_radius[i, :, :] / csdl.sin(phi)

            F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-(f_tip**2)**0.5))
            F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-(f_hub**2)**0.5))

            F_initial = F_tip * F_hub
            F = tip_loss * F_initial + (1 - tip_loss) * 1

            # Smooth the transition if airfoil model consists of multiple airfoils
            # if isinstance(airfoil_model, CompositeAirfoilModel):
            #     if airfoil_model.smoothing:
            #         window = airfoil_model.transition_window
            #         indices = airfoil_model.stop_indices
            #         Cl = smooth_quantities_spanwise(Cl, indices, window)
            #         Cd = smooth_quantities_spanwise(Cd, indices, window)

            # Compute sectional thrust, torque, moments and coefficients
            Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
            Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)
            dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr[i, :, :] * F
            dQ = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Ct * radius_vec[i, :, :] * dr[i, :, :] * F
            dMx = radius_vec[i, :, :] * csdl.sin(psi[i, :, :]) * dT
            dMy = -radius_vec[i, :, :] * csdl.cos(psi[i, :, :]) * dT

            thrust = integrate_quantity(dT, integration_scheme)
            torque = integrate_quantity(dQ, integration_scheme)
            Mx = integrate_quantity(dMx, integration_scheme)
            My = integrate_quantity(dMy, integration_scheme)

            C_T = thrust / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2)
            C_Mx = Mx / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2 * radius[i, 0, 0])
            C_My = My / (rho[i, 0, 0] * np.pi * radius[i, 0, 0]**2 * (omega[i] * radius[i, 0, 0]) ** 2 * radius[i, 0, 0])

            # Solver for lam_i (different from lam_exp_i)
            lam_i = csdl.ImplicitVariable(shape=(1, ), value=0.1)
            lam_i_res = lam_i - C_T / (2 * (mu[i]**2 + (mu_z[i] + lam_i)**2)**0.5)
            lam_i_solver = csdl.nonlinear_solvers.Newton(print_status=False)
            lam_i_solver.add_state(lam_i, lam_i_res)
            lam_i_solver.run()

            lam = mu_z[i] + csdl.average(lam_i)

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
            L_mat_new = L_mat_new.set(csdl.slice[:, 0], L_mat_new[:, 0]/ V_eff*2) 
            L_mat_new = L_mat_new.set(csdl.slice[:, 1], L_mat_new[:, 1]/ V_eff_2)
            L_mat_new = L_mat_new.set(csdl.slice[:, 2], L_mat_new[:, 2]/ V_eff_2)

            L_block_mat = L_block_mat.set(
                slices=csdl.slice[num_node_index:num_node_index+3, num_node_index:num_node_index+3],
                value=L_mat_new
            )
            rhs_block = rhs_block.set(csdl.slice[num_node_index], C_T)
            rhs_block = rhs_block.set(csdl.slice[num_node_index+1], -C_My)
            rhs_block = rhs_block.set(csdl.slice[num_node_index+2], C_Mx)

            # compute thrust/torque/power coefficient
            CT = thrust / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius[i, 0, 0])**4
            area = np.pi * radius[i, 0, 0]**2
            V_tip = rpm[i] / 60 * 2 * np.pi * radius[i, 0, 0]
            C_T_new = thrust / rho[i, 0, 0]/ area / V_tip**2
            CQ = torque / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius[i, 0, 0])**5
            CP = 2 * np.pi * CQ

            CT_H = CT * 4 / np.pi**3
            CQ_H = CQ * 8 / np.pi**3

            power = CP * rho[i, 0, 0] * (rpm[i] / 60)**3 * (2 * radius[i, 0, 0])**5

            # Compute advance ratio and efficiency and FOM
            J = Vx[i, 0, 0] / (rpm[i] / 60) / (2 * radius[i, 0, 0])
            eta = CT * J / CP
            # FoM = CT * (CT/2)**0.5 / CP
            FoM = CT_H * (CT_H/2)**0.5 / CQ_H

            # Storing data
            inflow_container = inflow_container.set(csdl.slice[i, :, :], lam_exp_i)
            ux_container = ux_container.set(csdl.slice[i, :, :], ux)
            dT_container = dT_container.set(csdl.slice[i, :, :], dT)
            dQ_container = dQ_container.set(csdl.slice[i, :, :], dQ)
            dD_container = dD_container.set(csdl.slice[i, :, :], dQ / radius_vec[i, :, :])
            phi_container = phi_container.set(csdl.slice[i, :, :], phi)
            thrust_container = thrust_container.set(csdl.slice[i], thrust)
            torque_container = torque_container.set(csdl.slice[i], torque)
            C_T_containter = C_T_containter.set(csdl.slice[i], C_T_new)
            C_P_containter = C_P_containter.set(csdl.slice[i], CP)
            C_Q_containter = C_Q_containter.set(csdl.slice[i], CQ)
            eta_container = eta_container.set(csdl.slice[i], eta)
            FoM_container = FoM_container.set(csdl.slice[i], FoM)

            states_container = states_container.set(csdl.slice[i, :], state_vec[num_node_index:num_node_index+3])
            # residual_container = residual_container.set(csdl.slice[i, :], residual[num_node_index:num_node_index+3])

        # define residual based on steady state assumption
        residual = state_vec - csdl.matvec(L_block_mat, rhs_block)
        
    
        # basic fixed point iteration for state update
        state_update = 0.5 * (csdl.matvec(L_block_mat, rhs_block) - state_vec)

        solver = csdl.nonlinear_solvers.GaussSeidel(
            max_iter=100, 
            residual_jac_kwargs={'elementwise' : False, 'loop' : True}
        )
        solver.add_state(state_vec, residual=residual, state_update=state_vec + state_update)
        solver.run()

        # for i in csdl.frange(num_nodes):
        #     num_node_index = i * 3 

        #     dT = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Cx * dr[i, :, :] * F
        #     dQ = 0.5 * B * rho[i, 0, 0] * (ux**2 + Vt[i, :, :]**2) * chord_profile[i, :, :] * Ct * radius_vec[i, :, :] * dr[i, :, :] * F
        #     thrust = integrate_quantity(dT, integration_scheme)
        #     torque = integrate_quantity(dQ, integration_scheme)

        #     # compute thrust/torque/power coefficient
        #     CT = thrust / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius[i, 0, 0])**4
        #     area = np.pi * radius[i, 0, 0]**2
        #     V_tip = rpm[i] / 60 * 2 * np.pi * radius[i, 0, 0]
        #     C_T_new = thrust / rho[i, 0, 0]/ area / V_tip**2
        #     CQ = torque / rho[i, 0, 0] / (rpm[i] / 60)**2 / (2 * radius[i, 0, 0])**5
        #     CP = 2 * np.pi * CQ

        #     CT_H = CT * 4 / np.pi**3
        #     CQ_H = CQ * 8 / np.pi**3

        #     power = CP * rho[i, 0, 0] * (rpm[i] / 60)**3 * (2 * radius[i, 0, 0])**5

        #     # Compute advance ratio and efficiency and FOM
        #     J = Vx[i, 0, 0] / (rpm[i] / 60) / (2 * radius[i, 0, 0])
        #     eta = CT * J / CP
        #     # FoM = CT * (CT/2)**0.5 / CP
        #     FoM = CT_H * (CT_H/2)**0.5 / CQ_H

        #     # Storing data
        #     inflow_container = inflow_container.set(csdl.slice[i, :, :], lam_exp_i)
        #     ux_container = ux_container.set(csdl.slice[i, :, :], ux)
        #     dT_container = dT_container.set(csdl.slice[i, :, :], dT)
        #     dQ_container = dQ_container.set(csdl.slice[i, :, :], dQ)
        #     dD_container = dD_container.set(csdl.slice[i, :, :], dQ / radius_vec[i, :, :])
        #     phi_container = phi_container.set(csdl.slice[i, :, :], phi)
        #     thrust_container = thrust_container.set(csdl.slice[i], thrust)
        #     torque_container = torque_container.set(csdl.slice[i], torque)
        #     C_T_containter = C_T_containter.set(csdl.slice[i], C_T_new)
        #     C_P_containter = C_P_containter.set(csdl.slice[i], CP)
        #     C_Q_containter = C_Q_containter.set(csdl.slice[i], CQ)
        #     eta_container = eta_container.set(csdl.slice[i], eta)
        #     FoM_container = FoM_container.set(csdl.slice[i], FoM)

        #     states_container = states_container.set(csdl.slice[i, :], state_vec[num_node_index:num_node_index+3])
        #     residual_container = residual_container.set(csdl.slice[i, :], residual[num_node_index:num_node_index+3])

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
        outputs.normalized_axial_induced_flow = lambda_container
        outputs.inflow = inflow_container
        outputs.sectional_inflow_angle = phi_container

        return outputs


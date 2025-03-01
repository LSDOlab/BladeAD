import csdl_alpha as csdl
import numpy as np
from BladeAD.utils.var_groups import RotorAnalysisOutputs
from BladeAD.utils.integration_schemes import integrate_quantity
from BladeAD.utils.smooth_quantities_spanwise import smooth_quantities_spanwise
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel


def compute_quantities_of_interest(
    shape: tuple,
    phi: csdl.Variable,
    Vx: csdl.Variable,
    Vt: csdl.Variable,
    rpm: csdl.Variable,
    sigma: csdl.Variable,
    rho: csdl.Variable,
    F: csdl.Variable,
    chord_profile: csdl.Variable,
    Cl: csdl.Variable,
    Cd: csdl.Variable,
    dr: csdl.Variable,
    radius_vector: csdl.Variable,
    radius: csdl.Variable,
    num_blades: int,
    integration_scheme: str,
    airfoil_model=None,
) -> RotorAnalysisOutputs:
    
    num_nodes = shape[0]

    # convert rpm to rps
    n = rpm / 60

    Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
    Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

    # axial-induced velocity
    ux = (4 * F * Vt * csdl.sin(phi)**2) / (4 * F * csdl.sin(phi) * csdl.cos(phi) +  sigma * Ct)
    ux_2 = Vx + sigma * Cx * Vt / (4 * F * csdl.sin(phi) * csdl.cos(phi) + sigma * Ct)

    # tangential-induced velocity
    ut = 2 * Vt * sigma * Ct / (2 * F * csdl.sin(2 * phi) + sigma * Ct)

    # Do smoothing if specified by user (do by default)
    if isinstance(airfoil_model, CompositeAirfoilModel):
        if airfoil_model.smoothing:
            window = airfoil_model.transition_window
            indices = airfoil_model.stop_indices
            ux = smooth_quantities_spanwise(ux, indices, window)
            ut = smooth_quantities_spanwise(ut, indices, window)
            Cx = smooth_quantities_spanwise(Cx, indices, window)
            Ct = smooth_quantities_spanwise(Ct, indices, window)

    # compute sectional thrust and torque
    # via momentum theory
    dT = 4 * np.pi * radius_vector * rho * ux * (ux - Vx) * F * dr
    dQ = 2 * np.pi * radius_vector**2 * rho * ux * ut * F * dr

    # via blade element theory (set up to be equivalent to momentum theory)
    # technically not necessary to compute both here
    dT2 = num_blades * Cx * 0.5 * rho * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord_profile * dr
    dQ2 = num_blades * Ct * 0.5 * rho * (ux_2**2 + (Vt - 0.5 * ut)**2) * chord_profile * dr * radius_vector

    # compute total thrust and torque
    thrust = integrate_quantity(dT, integration_scheme)
    torque = integrate_quantity(dQ, integration_scheme)
    
    # compute thrust/torque coefficient
    C_T = thrust / rho[:, 0, 0] / n**2 / (2 * radius[:, 0, 0])**4
    C_Q = torque / rho[:, 0, 0] / n**2 / (2 * radius[:, 0, 0])**5
    C_P = 2 * np.pi * C_Q

    CT_H = C_T * 4 / np.pi**3
    CQ_H = C_Q * 8 / np.pi**3

    area = np.pi * radius[:, 0, 0]**2
    V_tip = rpm / 60 * 2 * np.pi * radius[:, 0, 0]
    C_T_new = thrust / rho[:, 0, 0]/ area / V_tip**2


    power = C_P * rho[:, 0, 0] * n**3 * (2 * radius[:, 0, 0])**5

    # Compute advance ratio and efficiency and FOM
    Vx_num_nodes = Vx[:, 0, 0]
    J = Vx_num_nodes / n / (2 * radius[:, 0, 0])
    eta = C_T * J / C_P
    # FoM = C_T * (C_T/2)**0.5 / C_P
    FoM = CT_H * (CT_H/2)**0.5 / CQ_H


    bem_outputs = RotorAnalysisOutputs(
        axial_induced_velocity=ux,
        tangential_induced_velocity=ut,
        sectional_thrust=dT2,
        sectional_torque=dQ2,
        sectional_drag=dQ2 / radius_vector,
        total_thrust=thrust,
        total_torque=torque,
        total_power=power,
        efficiency=eta,
        figure_of_merit=FoM,
        thrust_coefficient=C_T_new,
        torque_coefficient=C_Q,
        power_coefficient=C_P
    )

    bem_outputs.advance_ratio = J

    return bem_outputs






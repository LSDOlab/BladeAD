import csdl_alpha as csdl
import numpy as np
from BladeAD.utils.var_groups import RotorAnalysisOutputs
from BladeAD.utils.integration_schemes import integrate_quantity


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
    C_T = thrust / rho / n**2 / (2 * radius)**4
    C_Q = torque / rho / n**2 / (2 * radius)**5
    C_P = 2 * np.pi * C_Q

    # Compute advance ratio and efficiency and FOM
    Vx_num_nodes = Vx[num_nodes, 0, 0]
    J = Vx_num_nodes / n / (2 * radius)
    eta = C_T * J / C_P
    FoM = C_T * (C_T/2)**0.5 / C_P

    bem_outputs = RotorAnalysisOutputs(
        axial_induced_velocity=ux,
        tangential_induced_velocity=ut,
        sectional_thrust=dT,
        sectional_torque=dQ,
        total_thrust=thrust,
        total_torque=torque,
        efficiency=eta,
        figure_of_merit=FoM,
        thrust_coefficient=C_T,
        torque_coefficient=C_Q,
        power_coefficient=C_P
    )

    return bem_outputs






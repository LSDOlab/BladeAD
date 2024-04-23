import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass


@dataclass
class BEMoutputs(csdl.VariableGroup):
    axial_induced_velocity: csdl.Variable
    tangential_induced_velocity: csdl.Variable
    sectional_thrust: csdl.Variable
    sectional_torque: csdl.Variable
    total_thrust: csdl.Variable
    total_torque: csdl.Variable
    efficiency: csdl.Variable
    figure_of_merit: csdl.Variable
    thrust_coefficient: csdl.Variable
    torque_coefficient: csdl.Variable
    power_coefficient: csdl.Variable


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
):
    num_nodes = shape[0]
    num_radial = shape[1]
    num_azimuthal = shape[2]

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
    thrust = csdl.Variable(shape=(num_nodes, ), value=0.)
    torque = csdl.Variable(shape=(num_nodes, ), value=0.)
    if integration_scheme == 'trapezoidal':
        for i in csdl.frange(num_nodes):
            for j in csdl.frange(num_radial-1):
                thrust = thrust.set(
                    csdl.slice[i], thrust + csdl.sum((dT[i, j+1, :] +  dT[i, j, :]) / 2)
                )
                torque = torque.set(
                    csdl.slice[i], torque + csdl.sum((dQ[i, j+1, :] +  dQ[i, j, :]) / 2)
                )
        thrust =  thrust / num_azimuthal
        torque = torque / num_azimuthal

    elif integration_scheme == 'Simpson':
        for i in csdl.frange(num_nodes):
            for j in csdl.frange(int(num_radial / 2)):
                j1 = 2 * (j + 1) - 2 
                j2 = 2 * (j + 1) - 1
                j3 = 2 * (j + 1)
                thrust = thrust.set(
                    csdl.slice[i], thrust[i] + csdl.sum(dT[i, j1, :] + 4 * dT[i, j2, :] + dT[i, j3, :]) 
                )

                torque = torque.set(
                    csdl.slice[i], torque[i] + csdl.sum(dQ[i, j1, :] + 4 * dQ[i, j2, :] + dQ[i, j3, :]) 
                )
        thrust =  thrust / 3 / num_azimuthal
        torque = torque / 3 / num_azimuthal

    elif integration_scheme == 'Riemann':
        thrust = csdl.sum(dT, axes=(1, 2)) / num_azimuthal
        torque = csdl.sum(dQ, axes=(1, 2)) / num_azimuthal
    
    else:
        raise NotImplementedError("Unknown integration scheme.")
    
    # compute thrust/torque coefficient
    C_T = thrust / rho / n**2 / (2 * radius)**4
    C_Q = torque / rho / n**2 / (2 * radius)**5
    C_P = 2 * np.pi * C_Q

    # Compute advance ratio and efficiency and FOM
    Vx_num_nodes = csdl.sum(Vx, axes=(1, 2)) / shape[1] / shape[2]
    J = Vx_num_nodes / n / (2 * radius)
    eta = C_T * J / C_P
    FoM = C_T * (C_T/2)**0.5 / C_P

    bem_outputs = BEMoutputs(
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






import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass


@dataclass
class ImplicitModelOutputs:
    bem_residual: csdl.Variable
    inflow_angle: csdl.Variable
    tip_loss_factor: csdl.Variable
    Cl: csdl.Variable
    Cd: csdl.Variable


def compute_inflow_angle(
    shape,
    num_blades,
    airfoil_model,
    atmos_states,
    chord_profile,
    twist_profile, 
    frame_velocity,
    tangential_velocity,
    radius_vec_exp,
    radius,
    hub_radius,
    sigma,

):
    mu = atmos_states.dynamic_viscosity
    rho = atmos_states.density
    a = atmos_states.speed_of_sound
    Vx = frame_velocity[:, :, :, 0] 
    Vt =  tangential_velocity
    Vr = (Vx**2 + Vt**2)**0.5

    Re = rho * Vr * chord_profile / mu
    Ma = Vr / a

    phi = csdl.ImplicitVariable(shape=shape, value=np.deg2rad(np.ones(shape)))
    alpha = twist_profile - phi

    Cl, Cd = airfoil_model.evaluate(alpha, Re, Ma)

    # Prandtl tip losses 
    f_tip = num_blades / 2 * (radius - radius_vec_exp) / radius / csdl.sin(phi)
    f_hub = num_blades / 2 * (radius_vec_exp - hub_radius) / hub_radius / csdl.sin(phi)

    F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-f_tip))
    F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-f_hub))

    F = F_tip * F_hub

    # Setting up residual function
    Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
    Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)


    term1 = Vt * (sigma * Cx - 4 * F * csdl.sin(phi)**2)
    term2 = Vx * (2 * F * csdl.sin(2 * phi) + Ct * sigma)
    
    
    # print(frame_velocity.value)

    bem_residual = term1 + term2

    eps = 1e-7
    solver = csdl.nonlinear_solvers.BracketedSearch(tolerance=1e-12)
    solver.add_state(phi, bem_residual, bracket=(0., np.pi / 2 - eps))
    solver.run()

    implicit_bem_outputs = ImplicitModelOutputs(
        bem_residual=bem_residual,
        inflow_angle=phi,
        tip_loss_factor=F,
        Cl=Cl,
        Cd=Cd,
    )

    # print("term 1", term1.value)
    # print("term 2", term2.value)
    # print("vt", Vt.value)
    # print("vt", Vt.shape)
    # print("radius", radius.value)
    # print("radius", radius_vec_exp.value)

    return implicit_bem_outputs





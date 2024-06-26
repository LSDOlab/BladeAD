import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass
from BladeAD.utils.smooth_quantities_spanwise import smooth_quantities_spanwise
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel


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
    # atmos_states,
    mu, 
    rho,
    a,
    chord_profile,
    twist_profile, 
    frame_velocity,
    tangential_velocity,
    radius_vec_exp,
    radius,
    hub_radius,
    sigma,
    tip_loss,
    initial_value=None,
):
    # mu = atmos_states.dynamic_viscosity
    # rho = atmos_states.density
    # a = atmos_states.speed_of_sound
    
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

    F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-(f_tip**2)**0.5))
    F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-(f_hub**2)**0.5))

    if tip_loss:
        F = F_tip * F_hub
    else:
        F = 1

    # Setting up residual 
    Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
    Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

    term1 = Vt * (sigma * Cx - 4 * F * csdl.sin(phi)**2)
    term2 = Vx * (2 * F * csdl.sin(2 * phi) + Ct * sigma)
    
    bem_residual = term1 + term2

    # Setting up bracketed search
    if initial_value is None:
        eps = 1e-7
        solver = csdl.nonlinear_solvers.BracketedSearch(max_iter=50, residual_jac_kwargs={'elementwise':True})
        solver.add_state(phi, bem_residual, bracket=(0., np.pi / 2 - eps))
    else:
        raise NotImplementedError
    
    solver.run()

    # if isinstance(airfoil_model, CompositeAirfoilModel):
    #     if airfoil_model.smoothing:
    #         window = airfoil_model.transition_window
    #         indices = airfoil_model.stop_indices
    #         # phi = smooth_quantities_spanwise(phi, indices, window)
    #         # F = smooth_quantities_spanwise(F, indices, window)
    #         Cl = smooth_quantities_spanwise(Cl, indices, window)
    #         # Cd = smooth_quantities_spanwise(Cd, indices, window)

    # Storing outputs
    implicit_bem_outputs = ImplicitModelOutputs(
        bem_residual=bem_residual,
        inflow_angle=phi,
        tip_loss_factor=F,
        Cl=Cl,
        Cd=Cd,
    )

    return implicit_bem_outputs





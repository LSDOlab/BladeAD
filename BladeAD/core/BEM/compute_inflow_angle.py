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
    memory_efficiency=False,
):
    
    Vx = frame_velocity[:, :, :, 0] 
    Vt =  tangential_velocity
    Vr = (Vx**2 + Vt**2)**0.5

    Re = rho * Vr * chord_profile / mu
    Ma = Vr / a

    if memory_efficiency:
        num_nodes = shape[0]
        num_radial = shape[1]
        num_azimuthal = shape[2]

        phi_container = csdl.Variable(shape=shape, value=0.)
        Cl_container = csdl.Variable(shape=shape, value=0.)
        Cd_container = csdl.Variable(shape=shape, value=0.)
        F_container = csdl.Variable(shape=shape, value=0.)
        res_container = csdl.Variable(shape=shape, value=0.)
        for i in csdl.frange(num_nodes):
            for j in csdl.frange(num_radial):
                for k in csdl.frange(num_azimuthal):
                    phi = csdl.ImplicitVariable(shape=(1,), value=np.deg2rad(1))
                    alpha = twist_profile[i, j, k] - phi

                    Cl, Cd = airfoil_model.evaluate(alpha, Re[i, j, k], Ma[i, j, k])

                    # Prandtl tip losses 
                    f_tip = num_blades / 2 * (radius - radius_vec_exp[i, j, k]) / radius / csdl.sin(phi)
                    f_hub = num_blades / 2 * (radius_vec_exp[i, j, k] - hub_radius) / hub_radius / csdl.sin(phi)

                    F_tip = 2 / np.pi * csdl.arccos(csdl.exp(-(f_tip**2)**0.5))
                    F_hub = 2 / np.pi * csdl.arccos(csdl.exp(-(f_hub**2)**0.5))

                    F = F_tip * F_hub

                    # Setting up residual 
                    Cx = Cl * csdl.cos(phi) - Cd * csdl.sin(phi)
                    Ct = Cl * csdl.sin(phi) + Cd * csdl.cos(phi)

                    term1 = Vt[i, j, k] * (sigma[i, j, k] * Cx - 4 * F * csdl.sin(phi)**2)
                    term2 = Vx[i, j, k] * (2 * F * csdl.sin(2 * phi) + Ct * sigma[i, j, k])

                    bem_residual = term1 + term2

                    # Setting up bracketed search
                    eps = 1e-7
                    solver = csdl.nonlinear_solvers.BracketedSearch(max_iter=50, residual_jac_kwargs={'elementwise':True, 'loop': True})
                    solver.add_state(phi, bem_residual, bracket=(0., np.pi / 2 - eps))
                    
                    solver.run()

                    phi_container = phi_container.set(csdl.slice[i, j, k], phi)
                    Cl_container = Cl_container.set(csdl.slice[i, j, k], Cl)
                    Cd_container = Cd_container.set(csdl.slice[i, j, k], Cd)
                    F_container = F_container.set(csdl.slice[i, j, k], F)
                    res_container = res_container.set(csdl.slice[i, j, k], bem_residual)

        implicit_bem_outputs = ImplicitModelOutputs(
            bem_residual=res_container,
            inflow_angle=phi_container,
            tip_loss_factor=F_container,
            Cl=Cl_container,
            Cd=Cd_container,
        )



    else:
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
        eps = 1e-7
        solver = csdl.nonlinear_solvers.BracketedSearch(max_iter=50, residual_jac_kwargs={'elementwise':True, 'loop': True})
        solver.add_state(phi, bem_residual, bracket=(0., np.pi / 2 - eps))
        
        solver.run()

        # Storing outputs
        implicit_bem_outputs = ImplicitModelOutputs(
            bem_residual=bem_residual,
            inflow_angle=phi,
            tip_loss_factor=F,
            Cl=Cl,
            Cd=Cd,
        )

    return implicit_bem_outputs





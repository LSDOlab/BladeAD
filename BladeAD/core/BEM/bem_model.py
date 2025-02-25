from typing import Union
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorAnalysisOutputs
from BladeAD.core.preprocessing.compute_local_frame_velocity import compute_local_frame_velocities
from BladeAD.core.preprocessing.preprocess_variables import preprocess_input_variables
from BladeAD.core.BEM.compute_inflow_angle import compute_inflow_angle
from BladeAD.core.BEM.compute_quantities_of_interest import compute_quantities_of_interest
import csdl_alpha as csdl
import numpy as np


class BEMModel:
    """Blade Element Momentum (BEM) model.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    airfoil_model : AirfoilModel
        Airfoil model.
    integration_scheme : str
        Integration scheme for thrust and torque computation. 
        Options are 'Simpson', 'Riemann', 'trapezoidal'. Default is 'trapezoidal'.
    tip_loss : bool
        Include tip loss. Default is True.
    """
    def __init__(
        self,
        num_nodes: int,
        airfoil_model,
        integration_scheme: str = 'trapezoidal',
        tip_loss:bool = True,
    ) -> None:
        csdl.check_parameter(num_nodes, 'num_nodes', types=int)
        csdl.check_parameter(integration_scheme, 'integration_scheme', values=('Simpson', 'Riemann', 'trapezoidal'))
        csdl.check_parameter(tip_loss, "tip_loss", types=bool)
        
        self.num_nodes = num_nodes
        self.airfoil_model = airfoil_model
        self.integration_scheme = integration_scheme
        self.tip_loss = tip_loss

    def evaluate(self, inputs: RotorAnalysisInputs, ref_point: Union[csdl.Variable, np.ndarray]=np.array([0., 0., 0.])) -> RotorAnalysisOutputs:
        """Evaluate the BEM solver.

        Parameters
        ----------
        inputs : RotorAnalysisInputs
            Input variables. Includes quantities like rpm, mesh_parameters, mesh_velocity, etc.
        ref_point : Union[csdl.Variable, np.ndarray], optional
            Reference point for computing moments, by default np.array([0., 0., 0.])

        Returns
        -------
        RotorAnalysisOutputs (dataclass)
            Output variables. Includes quanties like total/sectional thrust, total/sectional torque, etc.
        """
        csdl.check_parameter(inputs, "inputs", types=RotorAnalysisInputs)
        csdl.check_parameter(ref_point, "ref_point", types=(csdl.Variable, np.ndarray))

        try:
            ref_point.reshape((3, ))
        except:
            raise Exception("reference point must have shape (3, )")
        
        num_nodes = self.num_nodes
        num_radial = inputs.mesh_parameters.num_radial
        if self.integration_scheme == 'Simpson' and (num_radial % 2) == 0:
            raise ValueError("'num_radial' must be odd if integration scheme is Simpson.")

        num_azimuthal = inputs.mesh_parameters.num_azimuthal
        num_blades = inputs.mesh_parameters.num_blades
        norm_hub_radius = inputs.mesh_parameters.norm_hub_radius

        if isinstance(num_azimuthal, list):
            num_azimuthal = num_azimuthal[0]
        
        if isinstance(num_radial, list):
            num_radial = num_radial[0]

        if isinstance(num_blades, list):
            num_blades = num_blades[0]

        shape = (num_nodes, num_radial, num_azimuthal)
        thrust_vector = inputs.mesh_parameters.thrust_vector
        thrust_origin = inputs.mesh_parameters.thrust_origin
        mesh_velocity = inputs.mesh_velocity
        radius = inputs.mesh_parameters.radius
        chord_profile = inputs.mesh_parameters.chord_profile
        twist_profile = inputs.mesh_parameters.twist_profile
        rpm = inputs.rpm

        if mesh_velocity.shape[0] != num_nodes:
            raise Exception(f"Inconsisten shapes. 'num_nodes' is {self.num_nodes} but 'mesh_velocity' has shape {mesh_velocity.shape}. First dimension (num_nodes) needs to match")

        if isinstance(chord_profile, list):
            chord_profile = chord_profile[0]

        if isinstance(twist_profile, list):
            twist_profile = twist_profile[0]

        if isinstance(radius, list):
            radius = radius[0]

        if isinstance(norm_hub_radius, list):
            norm_hub_radius = norm_hub_radius[0]

        if isinstance(thrust_vector, list):
            thrust_vector_mat = csdl.Variable(shape=mesh_velocity.shape, value=0)
            for i in range(self.num_nodes):
                thrust_vector_mat = thrust_vector_mat.set(
                    csdl.slice[i, :], value=thrust_vector[i]
                )
            thrust_vector = thrust_vector_mat

        if isinstance(thrust_origin, list):
            thrust_origin_mat = csdl.Variable(shape=mesh_velocity.shape, value=0)
            for i in range(self.num_nodes):
                thrust_origin_mat = thrust_origin_mat.set(
                    csdl.slice[i, :], value=thrust_origin[i]
                )
            thrust_origin = thrust_origin_mat

        # if radius.shape == (num_nodes, ):
        radius = csdl.expand(radius, shape, action='i->ijk')
        
        pre_process_outputs = preprocess_input_variables(
            shape=shape,
            radius=radius,
            chord_profile=chord_profile,
            twist_profile=twist_profile,
            norm_hub_radius=norm_hub_radius,
            thrust_vector=thrust_vector,
            thrust_origin=thrust_origin,
            origin_velocity=mesh_velocity,
            rpm=rpm,
            num_blades=num_blades,
            atmos_states=inputs.atmos_states,
        )

        # Compute the rotor frame velocities from "mesh" velocities
        # i.e., ac states at the rotor origin
        local_frame_velocities = compute_local_frame_velocities(
            shape=shape,
            thrust_vector=pre_process_outputs.thrust_vector_exp,
            origin_velocity=pre_process_outputs.thrust_origin_vel_exp,
            angular_speed=pre_process_outputs.angular_speed_exp,
            azimuth_angle=pre_process_outputs.azimuth_angle_exp,
            radius_vec=pre_process_outputs.radius_vector_exp,
            radius=radius,
        )

        # Solve for phi once with bracketed search
        bem_implicit_outputs = compute_inflow_angle(
            shape=shape,
            num_blades=num_blades,
            airfoil_model=self.airfoil_model,
            # atmos_states=inputs.atmos_states,
            mu=pre_process_outputs.mu_exp, # viscosity, not inflow ratio
            rho=pre_process_outputs.rho_exp,
            a=pre_process_outputs.a_exp,
            chord_profile=pre_process_outputs.chord_profile_exp,
            twist_profile=pre_process_outputs.twist_profile_exp,
            frame_velocity=local_frame_velocities.local_frame_velocity,
            tangential_velocity=local_frame_velocities.tangential_velocity,
            radius_vec_exp=pre_process_outputs.radius_vector_exp,
            radius=radius,
            hub_radius=pre_process_outputs.hub_radius,
            sigma=pre_process_outputs.sigma,
            tip_loss=self.tip_loss,
        )

        # Post-processing
        bem_outputs = compute_quantities_of_interest(
            shape=shape,
            phi=bem_implicit_outputs.inflow_angle,
            Vx=local_frame_velocities.local_frame_velocity[:, :, :, 0],
            Vt=local_frame_velocities.tangential_velocity,
            rpm=rpm,
            radius_vector=pre_process_outputs.radius_vector_exp,
            radius=radius,
            sigma=pre_process_outputs.sigma,
            rho=pre_process_outputs.rho_exp,
            F=bem_implicit_outputs.tip_loss_factor,
            chord_profile=pre_process_outputs.chord_profile_exp,
            Cl=bem_implicit_outputs.Cl,
            Cd=bem_implicit_outputs.Cd,
            dr=pre_process_outputs.element_width,
            num_blades=num_blades,
            integration_scheme=self.integration_scheme,
            airfoil_model=self.airfoil_model,
        )

        bem_outputs.residual = bem_implicit_outputs.bem_residual
        bem_outputs.sectional_inflow_angle = bem_implicit_outputs.inflow_angle

        thrust_vec_exp = pre_process_outputs.thrust_vector_exp
        thrust_origin_exp = pre_process_outputs.thrust_origin_exp

        thrust = bem_outputs.total_thrust
        torque = bem_outputs.total_torque

        forces = csdl.Variable(shape=(num_nodes, 3), value=0.)
        moments = csdl.Variable(shape=(num_nodes, 3), value=0.)

        # transform forces into body-fixed frame
        if inputs.ac_states is not None:
            phi = inputs.ac_states.phi        
            theta = inputs.ac_states.theta
            psi = inputs.ac_states.psi

            if phi.shape != (num_nodes, ):
                phi = csdl.expand(phi, (num_nodes, ))

            if theta.shape != (num_nodes, ):
                theta = csdl.expand(theta, (num_nodes, ))

            if psi.shape != (num_nodes, ):
                psi = csdl.expand(psi, (num_nodes, ))
            
            # Rotate forces into body-fixed reference frame
            for i in csdl.frange(num_nodes):
                L2B_mat = csdl.Variable(shape=(3, 3), value=0.)
                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[0, 0],
                    value=csdl.cos(theta[i]) * csdl.cos(psi[i]),
                )

                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[0, 1],
                    value=csdl.cos(theta[i]) * csdl.sin(psi[i]),
                )

                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[0, 2],
                    value= -csdl.sin(theta[i]),
                )

                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[1, 0],
                    value=csdl.sin(phi[i]) * csdl.sin(theta[i]) * csdl.cos(psi[i]) - csdl.cos(phi[i]) * csdl.sin(psi[i]),
                )

                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[1, 1],
                    value=csdl.sin(phi[i]) * csdl.sin(theta[i]) * csdl.sin(psi[i]) + csdl.cos(phi[i]) * csdl.cos(psi[i]),
                )

                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[1, 2],
                    value=csdl.sin(phi[i]) * csdl.cos(theta[i]),
                )

                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[2, 0],
                    value=csdl.cos(phi[i]) * csdl.sin(theta[i]) * csdl.cos(psi[i]) + csdl.sin(phi[i]) * csdl.sin(psi[i]),
                )

                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[2, 1],
                    value=csdl.cos(phi[i]) * csdl.sin(theta[i]) * csdl.sin(psi[i]) - csdl.sin(phi[i]) * csdl.cos(psi[i]),
                )

                L2B_mat = L2B_mat.set(
                    slices=csdl.slice[2, 2],
                    value=csdl.cos(phi[i]) * csdl.cos(theta[i]),
                )
                

                forces = forces.set(
                    csdl.slice[i, :],
                    csdl.matvec(
                        L2B_mat,
                        thrust[i] * thrust_vec_exp[i, :],
                    )
                )

                moments = moments.set(
                    csdl.slice[i, :],
                    csdl.cross(
                        csdl.matvec(
                            L2B_mat,
                            thrust_origin_exp[i, :] - ref_point,
                        ),
                        forces[i, :])
                )
        else:
            for i in csdl.frange(num_nodes):
                forces = forces.set(
                        csdl.slice[i, :], thrust[i] * thrust_vec_exp[i, :],
                        )

                moments = moments.set(
                    csdl.slice[i, :],
                    csdl.cross(
                        thrust_origin_exp[i, :] - ref_point,
                        forces[i, :])
                    )

        bem_outputs.forces = forces
        bem_outputs.moments = moments

        return bem_outputs
    
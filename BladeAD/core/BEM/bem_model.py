from typing import Union
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorAnalysisOutputs
from BladeAD.core.preprocessing.compute_local_frame_velocity import compute_local_frame_velocities
from BladeAD.core.preprocessing.preprocess_variables import preprocess_input_variables
from BladeAD.core.BEM.compute_inflow_angle import compute_inflow_angle
from BladeAD.core.BEM.compute_quantities_of_interest import compute_quantities_of_interest
import csdl_alpha as csdl


class BEMModel:
    def __init__(
        self,
        num_nodes: int,
        airfoil_model,
        integration_scheme: str = 'trapezoidal'
    ) -> None:
        csdl.check_parameter(num_nodes, 'num_nodes', types=int)
        csdl.check_parameter(integration_scheme, 'integration_scheme', values=('Simpson', 'Riemann', 'trapezoidal'))
        
        self.num_nodes = num_nodes
        self.airfoil_model = airfoil_model
        self.integration_scheme = integration_scheme

    def evaluate(self, inputs: RotorAnalysisInputs) -> RotorAnalysisOutputs:
        """Evaluate the BEM solver.
        """
        num_nodes = self.num_nodes
        num_radial = inputs.mesh_parameters.num_radial
        if self.integration_scheme == 'Simpson' and (num_radial % 2) == 0:
            raise ValueError("'num_radial' must be odd if integration scheme is Simpson.")

        num_azimuthal = inputs.mesh_parameters.num_azimuthal
        num_blades = inputs.mesh_parameters.num_blades
        norm_hub_radius = inputs.mesh_parameters.norm_hub_radius

        shape = (num_nodes, num_radial, num_azimuthal)
        thrust_vector = inputs.mesh_parameters.thrust_vector
        mesh_velocity = inputs.mesh_velocity
        radius = inputs.mesh_parameters.radius
        chord_profile = inputs.mesh_parameters.chord_profile
        twist_profile = inputs.mesh_parameters.twist_profile
        rpm = inputs.rpm

        pre_process_outputs = preprocess_input_variables(
            shape=shape,
            radius=radius,
            chord_profile=chord_profile,
            twist_profile=twist_profile,
            norm_hub_radius=norm_hub_radius,
            thrust_vector=thrust_vector,
            origin_velocity=mesh_velocity,
            rpm=rpm,
            num_blades=num_blades,
        )

        local_frame_velocities = compute_local_frame_velocities(
            shape=shape,
            thrust_vector=pre_process_outputs.thrust_vector_exp,
            origin_velocity=pre_process_outputs.thrust_origin_vel_exp,
            angular_speed=pre_process_outputs.angular_speed_exp,
            azimuth_angle=pre_process_outputs.azimuth_angle_exp,
            radius_vec=pre_process_outputs.radius_vector_exp,
            radius=radius,
        )

        bem_implicit_outputs = compute_inflow_angle(
            shape=shape,
            num_blades=num_blades,
            airfoil_model=self.airfoil_model,
            atmos_states=inputs.atmos_states,
            chord_profile=pre_process_outputs.chord_profile_exp,
            twist_profile=pre_process_outputs.twist_profile_exp,
            frame_velocity=local_frame_velocities.local_frame_velocity,
            tangential_velocity=local_frame_velocities.tangential_velocity,
            radius_vec_exp=pre_process_outputs.radius_vector_exp,
            radius=radius,
            hub_radius=pre_process_outputs.hub_radius,
            sigma=pre_process_outputs.sigma,
        )

        bem_outputs = compute_quantities_of_interest(
            shape=shape,
            phi=bem_implicit_outputs.inflow_angle,
            Vx=local_frame_velocities.local_frame_velocity[:, :, :, 0],
            Vt=local_frame_velocities.tangential_velocity,
            rpm=rpm,
            radius_vector=pre_process_outputs.radius_vector_exp,
            radius=radius,
            sigma=pre_process_outputs.sigma,
            rho=inputs.atmos_states.density,
            F=bem_implicit_outputs.tip_loss_factor,
            chord_profile=pre_process_outputs.chord_profile_exp,
            Cl=bem_implicit_outputs.Cl,
            Cd=bem_implicit_outputs.Cd,
            dr=pre_process_outputs.element_width,
            num_blades=num_blades,
            integration_scheme=self.integration_scheme,
        )

        bem_outputs.residual = bem_implicit_outputs.bem_residual

        return bem_outputs
    
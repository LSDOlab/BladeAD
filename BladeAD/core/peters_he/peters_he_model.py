from BladeAD.utils.var_groups import BEMInputs, RotorAnalysisOutputs
from BladeAD.core.preprocessing.compute_local_frame_velocity import compute_local_frame_velocities
from BladeAD.core.preprocessing.preprocess_variables import preprocess_input_variables
from BladeAD.core.peters_he.peters_he_inflow import solve_for_steady_state_inflow
import csdl_alpha as csdl


class PetersHeModel:
    def __init__(
        self, 
        num_nodes: int,
        airfoil_model,
        integration_scheme: str = "trapezoidal",
        Q: int = 3,
        M: int = 3,
    ) -> None:
        csdl.check_parameter(num_nodes, "num_nodes", types=int)
        csdl.check_parameter(integration_scheme, "integration_scheme", values=("Simpson", "Riemann", "trapezoidal"))
        csdl.check_parameter(Q, "Q", types=int)
        csdl.check_parameter(M, "M", types=int)

        if Q > 3:
            raise NotImplementedError("Q greater than 3 has not been tested yet and will likely not converge")

        self.num_nodes = num_nodes
        self.airfoil_model = airfoil_model
        self.integration_scheme = integration_scheme
        self.Q = Q
        self.M = M

    
    def evaluate(self, inputs: BEMInputs) -> RotorAnalysisOutputs:
        num_nodes = self.num_nodes
        num_radial = inputs.mesh_parameters.num_radial
        if self.integration_scheme == "Simpson" and (num_radial % 2) == 0:
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

        dynamic_inflow_outputs = solve_for_steady_state_inflow(
            shape=shape,
            psi=pre_process_outputs.azimuth_angle_exp,
            rpm=rpm,
            radius=radius,
            mu_z=local_frame_velocities.mu_z,
            mu=local_frame_velocities.mu,
            tangential_velocity=local_frame_velocities.tangential_velocity,
            frame_velocity=local_frame_velocities.local_frame_velocity, 
            radius_vec=pre_process_outputs.radius_vector_exp,
            chord_profile=pre_process_outputs.chord_profile_exp,
            twist_profile=pre_process_outputs.twist_profile_exp,
            airfoil_model=self.airfoil_model,
            atmos_states=inputs.atmos_states,
            num_blades=num_blades,
            dr=pre_process_outputs.element_width,
            integration_scheme=self.integration_scheme,
            M=self.M,
            Q=self.Q,
        )

        return dynamic_inflow_outputs

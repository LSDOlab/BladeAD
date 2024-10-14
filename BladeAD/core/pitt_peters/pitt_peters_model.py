from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorAnalysisOutputs
from BladeAD.core.preprocessing.compute_local_frame_velocity import compute_local_frame_velocities
from BladeAD.core.preprocessing.preprocess_variables import preprocess_input_variables
from BladeAD.core.pitt_peters.pitt_peters_inflow import solve_for_steady_state_inflow
from BladeAD.utils.post_process import compute_forces_and_moments
import csdl_alpha as csdl
import numpy as np
from typing import Union, List


class PittPetersModel:
    def __init__(
        self,
        num_nodes: int,
        airfoil_model,
        tip_loss: bool=True,
        integration_scheme: str = "trapezoidal",
        nl_solver_mode : str = "standard",
        use_frange_in_nl_solver : bool = True,
        rotation_direction : Union[str, None, List[str]] = None,
    ) -> None:
        csdl.check_parameter(num_nodes, "num_nodes", types=int)
        csdl.check_parameter(integration_scheme, "integration_scheme", values=("Simpson", "Riemann", "trapezoidal"))
        csdl.check_parameter(nl_solver_mode, "nl_solver_mode", values=("standard", "vectorized"))
        csdl.check_parameter(tip_loss, "tip_loss", types=bool)
        csdl.check_parameter(use_frange_in_nl_solver, "use_frange_in_nl_solver", types=bool)
        csdl.check_parameter(rotation_direction, "rotation_direction", types=(str, list), allow_none=True)
        
        self.num_nodes = num_nodes
        self.airfoil_model = airfoil_model
        self.integration_scheme = integration_scheme
        if tip_loss is True:
            self.tip_loss = 1
        else:
            self.tip_loss = 0
        self.nl_solver_mode = nl_solver_mode
        self.use_frange_in_nl_solver = use_frange_in_nl_solver
        if isinstance(rotation_direction, list):
            if len(rotation_direction) != num_nodes:
                raise ValueError(f"length of 'rotation_direction' is {len(rotation_direction)} but must be equal to 'num_nodes' ({num_nodes})")
            for string in rotation_direction:
                if not isinstance(string, str):
                    raise TypeError("'rotation_direction' must be a string ('cw' or 'ccw') or list of strings")
                if string not in ['ccw', 'cw']:
                    raise ValueError("'rotation_direction' can only be 'cw' or 'ccw' or list of those")
            self.rotation_direction = rotation_direction
        elif isinstance(rotation_direction, str):
            if rotation_direction not in ['ccw', 'cw']:
                raise ValueError("'rotation_direction' can only be 'cw' or 'ccw' or list of those")
            self.rotation_direction = [rotation_direction for i in range(num_nodes)]
        else:
            self.rotation_direction = None  


    def evaluate(
            self, inputs: RotorAnalysisInputs, 
            ref_point: Union[csdl.Variable, np.ndarray]=np.array([0., 0., 0.]),
            mirror_forces_and_moments: bool = False,
        ) -> RotorAnalysisOutputs:
        csdl.check_parameter(inputs, "inputs", types=RotorAnalysisInputs)
        csdl.check_parameter(ref_point, "ref_point", types=(csdl.Variable, np.ndarray))
        csdl.check_parameter(mirror_forces_and_moments, "mirror_forces_and_moments", types=bool)

        try:
            ref_point.reshape((3, ))
        except:
            raise Exception("reference point must have shape (3, )")
        num_nodes = self.num_nodes
        num_radial = inputs.mesh_parameters.num_radial
        if self.integration_scheme == "Simpson" and (num_radial % 2) == 0:
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
            theta_0=inputs.theta_0,
            theta_1_c=inputs.theta_1_c,
            theta_1_s=inputs.theta_1_s,
            xi_0=inputs.xi_0,
            xi_1_c=inputs.xi_1_c,
            xi_1_s=inputs.xi_1_s,
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
            norm_radius_vec=pre_process_outputs.norm_radius_exp,
            chord_profile=pre_process_outputs.chord_profile_exp,
            twist_profile=pre_process_outputs.twist_profile_exp,
            airfoil_model=self.airfoil_model,
            disk_inclination_angle=local_frame_velocities.disk_inclination_angle,
            # atmos_states=inputs.atmos_states,
            mu_atmos=pre_process_outputs.mu_exp,
            rho=pre_process_outputs.rho_exp,
            a=pre_process_outputs.a_exp,
            num_blades=num_blades,
            dr=pre_process_outputs.element_width,
            radius_vec_exp=pre_process_outputs.radius_vector_exp,
            hub_radius=pre_process_outputs.hub_radius,
            integration_scheme=self.integration_scheme,
            tip_loss=self.tip_loss,
            mode=self.nl_solver_mode,
            use_frange_in_nl_solver=self.use_frange_in_nl_solver,
        )

    
        thrust_vec_exp = pre_process_outputs.thrust_vector_exp
        thrust_origin_exp = pre_process_outputs.thrust_origin_exp

        thrust = dynamic_inflow_outputs.total_thrust
        torque = dynamic_inflow_outputs.total_torque

        forces, moments = compute_forces_and_moments(
            thrust=thrust,
            torque=torque,
            rotation_direction=self.rotation_direction,
            thrust_origin=thrust_origin_exp,
            thrust_vector=thrust_vec_exp,
            num_nodes=num_nodes,
            ac_states=inputs.ac_states,
            ref_point=ref_point,
            mirror_forces_and_moments=mirror_forces_and_moments,
        )

        dynamic_inflow_outputs.forces = forces
        dynamic_inflow_outputs.moments = moments

        return dynamic_inflow_outputs

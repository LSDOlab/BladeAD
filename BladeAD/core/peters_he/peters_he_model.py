from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorAnalysisOutputs
from BladeAD.core.preprocessing.compute_local_frame_velocity import compute_local_frame_velocities
from BladeAD.core.preprocessing.preprocess_variables import preprocess_input_variables
from BladeAD.core.peters_he.peters_he_inflow import solve_for_steady_state_inflow
import csdl_alpha as csdl
 
 
class PetersHeModel:
    """Peters--He dynamic inflow model.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    airfoil_model : object
        Airfoil model.
    tip_loss : bool, optional
        Include tip loss, by default True
    integration_scheme : str, optional
        Numerical integration scheme, by default "trapezoidal"
        Other options are "Simpson" and "Riemann".
    Q : int, optional
        Highest power of r/R, by default 3. Has to be less than or equal to 8.
    M : int, optional
        Highest harmonic number, by default 3. Has to be less than or equal to Q.
    """
    def __init__(
        self,
        num_nodes: int,
        airfoil_model,
        tip_loss: bool=True,
        integration_scheme: str = "trapezoidal",
        Q: int = 3,
        M: int = 3,
    ) -> None:
        csdl.check_parameter(num_nodes, "num_nodes", types=int)
        csdl.check_parameter(integration_scheme, "integration_scheme", values=("Simpson", "Riemann", "trapezoidal"))
        csdl.check_parameter(Q, "Q", types=int)
        csdl.check_parameter(M, "M", types=int)
        csdl.check_parameter(tip_loss, "tip_loss", types=bool)
 
        if num_nodes < 1:
            raise ValueError("num_nodes must be an integer greater or equal to 1")
 
        if M > Q:
            raise ValueError(f"M ({M}) (highest harmonic number) must not be greater than Q ({Q}) (highest power of r/R)")
 
        self.num_nodes = num_nodes
        self.airfoil_model = airfoil_model
        self.integration_scheme = integration_scheme
        self.Q = Q
        self.M = M
        self.tip_loss = tip_loss
 
   
    def evaluate(self, inputs: RotorAnalysisInputs) -> RotorAnalysisOutputs:
        """Evaluate the Peters--He dynamic inflow model.
        
        Parameters
        ----------
        inputs : RotorAnalysisInputs
            Rotor analysis inputs. Data class that stores rotor analysis 
            inputs such as rpm, mesh parameters, and mesh velocity, etc.
            
        Returns
        -------
        RotorAnalysisOutputs
            Rotor analysis outputs. Data class that stores rotor analysis
            outputs such as total thrust, total torque, etc.
        
        Raises
        ------
        ValueError
            If num_radial is even and integration scheme is Simpson.
        """
        num_nodes = self.num_nodes
        num_radial = inputs.mesh_parameters.num_radial
        if self.integration_scheme == "Simpson" and (num_radial % 2) == 0:
            raise ValueError("'num_radial' must be odd if integration scheme is Simpson.")
 
        num_azimuthal = inputs.mesh_parameters.num_azimuthal
        num_blades = inputs.mesh_parameters.num_blades
        norm_hub_radius = inputs.mesh_parameters.norm_hub_radius
 
        shape = (num_nodes, num_radial, num_azimuthal)
        thrust_vector = inputs.mesh_parameters.thrust_vector
        thrust_origin = inputs.mesh_parameters.thrust_origin
        mesh_velocity = inputs.mesh_velocity
        radius = inputs.mesh_parameters.radius
        chord_profile = inputs.mesh_parameters.chord_profile
        twist_profile = inputs.mesh_parameters.twist_profile
        rpm = inputs.rpm

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
 

        dynamic_inflow_outputs_1 = solve_for_steady_state_inflow(
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
            # atmos_states=inputs.atmos_states,
            rho_exp=pre_process_outputs.rho_exp,
            mu_exp=pre_process_outputs.mu_exp,
            a_exp=pre_process_outputs.a_exp,
            num_blades=num_blades,
            dr=pre_process_outputs.element_width,
            radius_vec_exp=pre_process_outputs.radius_vector_exp,
            norm_radius_vec_exp=pre_process_outputs.norm_radius_exp,
            disk_inclination_angle=local_frame_velocities.disk_inclination_angle,
            hub_radius=pre_process_outputs.hub_radius,
            integration_scheme=self.integration_scheme,
            tip_loss=self.tip_loss,
            M=self.M,
            Q=self.Q,
        )
 
        dynamic_inflow_outputs_2 = solve_for_steady_state_inflow(
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
            # atmos_states=inputs.atmos_states,
            rho_exp=pre_process_outputs.rho_exp,
            mu_exp=pre_process_outputs.mu_exp,
            a_exp=pre_process_outputs.a_exp,
            num_blades=num_blades,
            dr=pre_process_outputs.element_width,
            radius_vec_exp=pre_process_outputs.radius_vector_exp,
            norm_radius_vec_exp=pre_process_outputs.norm_radius_exp,
            disk_inclination_angle=local_frame_velocities.disk_inclination_angle,
            hub_radius=pre_process_outputs.hub_radius,
            integration_scheme=self.integration_scheme,
            tip_loss=self.tip_loss,
            M=self.M,
            Q=self.Q,
            initial_value=dynamic_inflow_outputs_1.states
        )
 
        return dynamic_inflow_outputs_2

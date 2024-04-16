import csdl_alpha as csdl
from typing import Union
from BladeAD.utils.var_groups import BEMInputs
from BladeAD.core.BEM.pre_processing import convert_caddee_velocities_into_rotor_frame


class BEMModel:
    def __init__(self, 
                 num_nodes : int,
                 num_radial :int,
                 num_azimuthal : int,
                 hub_radius: float = 0.2
                 ) -> None:
        self.num_nodes = num_nodes
        self.num_radial = num_radial
        self.num_azimuthal = num_azimuthal
        self.hub_radius = hub_radius

    def evaluate(self, inputs : BEMInputs): 
        num_nodes = self.num_nodes
        num_radial = self.num_radial
        num_azimuthal = self.num_azimuthal

        shape = (num_nodes, num_radial, num_azimuthal)
        thrust_vector = inputs.mesh_parameters.thrust_vector
        mesh_velocity = inputs.mesh_velocity
        radius = inputs.mesh_parameters.radius

        rotor_inflow, r_hub, dr = convert_caddee_velocities_into_rotor_frame(
            shape=shape,
            thrust_vector=thrust_vector,
            origin_velocity=mesh_velocity,
            radius=radius,

        )

        
import csdl_alpha as csdl
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class BEMMeshParameters(csdl.VariableGroup):
    thrust_vector : Union[np.ndarray, csdl.Variable]
    thrust_origin : Union[np.ndarray, csdl.Variable]
    chord_profile : Union[np.ndarray, csdl.Variable]
    twist_profile : Union[np.ndarray, csdl.Variable]
    radius : Union[float, int, csdl.Variable]
    num_radial: int
    num_azimuthal: int
    num_blades: int
    norm_hub_radius: float = 0.2


@dataclass
class AtmosStates(csdl.VariableGroup):
    density : Union[float, int, csdl.Variable] = 1.22518316 # 1.225
    speed_of_sound : Union[float, int, csdl.Variable] = 343
    temperature : Union[float, int, csdl.Variable] =  288.16
    pressure : Union[float, int, csdl.Variable] = 101325
    dynamic_viscosity : Union[float, int, csdl.Variable] = 1.735e-5


@dataclass
class BEMInputs(csdl.VariableGroup):
    atmos_states = AtmosStates()
    rpm: csdl.Variable = None
    ac_states: csdl.Variable = None
    mesh_velocity: csdl.Variable = None
    mesh_parameters: BEMMeshParameters = None



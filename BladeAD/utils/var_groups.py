import csdl_alpha as csdl
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class BEMMeshParameters(csdl.VariableGroup):
    thrust_vector : Union[np.ndarray, csdl.Variable]
    thrust_origin : Union[np.ndarray, csdl.Variable]
    blade_chord : Union[np.ndarray, csdl.Variable]
    blade_twist : Union[np.ndarray, csdl.Variable]
    radius : Union[float, int, csdl.Variable]


@dataclass
class BEMInputs(csdl.VariableGroup):
    ac_states = None
    atmos_states = None
    mesh_velocity = None
    mesh_parameters: BEMMeshParameters = None
from __future__ import annotations
import csdl_alpha as csdl
from dataclasses import dataclass
from typing import Union
import numpy as np
from typing import Optional


@dataclass
class RotorMeshParameters(csdl.VariableGroup):
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
    speed_of_sound : Union[float, int, csdl.Variable] = 343.
    temperature : Union[float, int, csdl.Variable] =  288.16
    pressure : Union[float, int, csdl.Variable] = 101325
    dynamic_viscosity : Union[float, int, csdl.Variable] = 1.735e-5



@dataclass
class AircaftStates(csdl.VariableGroup):
    u : Union[float, int, csdl.Variable, np.ndarray] = 0
    v : Union[float, int, csdl.Variable, np.ndarray] = 0
    w : Union[float, int, csdl.Variable, np.ndarray] = 0
    p : Union[float, int, csdl.Variable, np.ndarray] = 0
    q : Union[float, int, csdl.Variable, np.ndarray] = 0
    r : Union[float, int, csdl.Variable, np.ndarray] = 0
    phi : Union[float, int, csdl.Variable, np.ndarray] = 0
    theta : Union[float, int, csdl.Variable, np.ndarray] = 0
    psi : Union[float, int, csdl.Variable, np.ndarray] = 0
    x : Union[float, int, csdl.Variable, np.ndarray] = 0
    y : Union[float, int, csdl.Variable, np.ndarray] = 0
    z : Union[float, int, csdl.Variable, np.ndarray] = 0


@dataclass
class RotorAnalysisInputs(csdl.VariableGroup):
    """Rotor analysis inputs data class.

    Parameters
    ----------
    rpm : csdl.Variable
        The rpm of the rotor
    
    mesh_velocity : csdl.Variable 
        The mesh velocity (u_x, u_y, u_z) of the thrust origin
    
    mesh_parameters : RotorMeshParameters
        Instance of of RotorMeshParameters that stores information about the rotor geometry
    
    atmos_states : AtmosStates(), optional
        Instance of AtmosStates data class, by default AtmosStates() (states at sea-level)
    
    ac_states : AcStates(), optional
        Instance of AcStates() (if provided thrust will be decomposed into forces
        in the body-fixed reference frame based on aircraft orientation), by default None
    
    theta_0 : Union[csdl.Variable, int, float], optional
        Collective pitch at r/R=0.75; zero-th harmonic of blade pitch motion (radians); note 
        that it is up to the user to specify the blade twist such that it is zero at r/R=0.75, 
        by default 0
    
    theta_1_c : Union[csdl.Variable, int, float], optional
        Cyclic pitch angle: cosine coefficient for first harmonic (radians), by default 0
    
    theta_1_s : Union[csdl.Variable, int, float], optional
        Cyclic pitch angle: sine coefficient for first harmonic (radians), by default 0
    
    xi_0 : Union[csdl.Variable, int, float], optional
        Mean lag angle; zero-th harmonic of blade lag motion (radians), by default 0
    
    xi_1_c : Union[csdl.Variable, int, float], optional
        Cyclic lag angle; cosine coefficient for first harmonic (radians), by default 0
    
    xi_1_s : Union[csdl.Variable, int, float], optional
        Cyclic lag angle; sin coefficient for first harmonic (radians), by default 0
    """
    rpm: csdl.Variable
    mesh_velocity: csdl.Variable
    mesh_parameters: RotorMeshParameters
    atmos_states : Optional[AtmosStates] = AtmosStates()
    ac_states : Optional[AtmosStates] = None
    theta_0 : Union[csdl.Variable, int, float] = 0
    theta_1_c : Union[csdl.Variable, int, float] = 0
    theta_1_s : Union[csdl.Variable, int, float] = 0
    xi_0 : Union[csdl.Variable, int, float] = 0
    xi_1_c : Union[csdl.Variable, int, float] = 0
    xi_1_s : Union[csdl.Variable, int, float] = 0

@dataclass
class RotorAnalysisOutputs(csdl.VariableGroup):
    axial_induced_velocity: csdl.Variable
    tangential_induced_velocity: csdl.Variable
    sectional_thrust: csdl.Variable
    sectional_torque: csdl.Variable
    sectional_drag : csdl.Variable
    total_thrust: csdl.Variable
    total_torque: csdl.Variable
    total_power: csdl.Variable
    efficiency: csdl.Variable
    figure_of_merit: csdl.Variable
    thrust_coefficient: csdl.Variable
    torque_coefficient: csdl.Variable
    power_coefficient: csdl.Variable
    residual: csdl.Variable = None
    forces: csdl.Variable = None
    moments: csdl.Variable = None
    sectional_inflow_angle = None
    


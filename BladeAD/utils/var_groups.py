from __future__ import annotations
import csdl_alpha as csdl
from dataclasses import dataclass
from typing import Union
import numpy as np
from typing import Optional



@dataclass
class RotorMeshParameters(csdl.VariableGroup):
    """Data class for rotor mesh parameters.

    Parameters
    ----------
    thrust_vector : Union[np.ndarray, csdl.Variable]
        Thrust vector
    thrust_origin : Union[np.ndarray, csdl.Variable]
        Thrust origin
    chord_profile : Union[np.ndarray, csdl.Variable]
        Chord profile (m)
    twist_profile : Union[np.ndarray, csdl.Variable]
        Twist profile (radians)
    radius : Union[float, int, csdl.Variable]
        Radius of the rotor (m)
    num_radial : int
        Number of radial nodes
    num_azimuthal : int
        Number of azimuthal nodes
    num_blades : int
        Number of blades
    norm_hub_radius : float
        Normalmized hub radius (default=0.2)
    """
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
    """Data class for atmospheric states. Default values are for sea-level.

    Parameters
    ----------
    density : Union[float, int, csdl.Variable]
        Density of the atmosphere (kg/m^3)
    speed_of_sound : Union[float, int, csdl.Variable]
        Speed of sound (m/s)
    temperature : Union[float, int, csdl.Variable]
        Temperature (K)
    pressure : Union[float, int, csdl.Variable]
        Pressure (Pa)
    dynamic_viscosity : Union[float, int, csdl.Variable]
        Dynamic viscosity (Pa.s)
    """
    density : Union[float, int, csdl.Variable] = 1.22518316
    speed_of_sound : Union[float, int, csdl.Variable] = 343.
    temperature : Union[float, int, csdl.Variable] =  288.16
    pressure : Union[float, int, csdl.Variable] = 101325
    dynamic_viscosity : Union[float, int, csdl.Variable] = 1.735e-5

@dataclass
class AircaftStates(csdl.VariableGroup):
    """Data class for aircraft states.

    Parameters
    ----------
    u : Union[float, int, csdl.Variable, np.ndarray]
        u velocity component
    v : Union[float, int, csdl.Variable, np.ndarray]
        v velocity component
    w : Union[float, int, csdl.Variable, np.ndarray]
        w velocity component
    p : Union[float, int, csdl.Variable, np.ndarray]
        p angular velocity component (roll rate)
    q : Union[float, int, csdl.Variable, np.ndarray]
        q angular velocity component (pitch rate)
    r : Union[float, int, csdl.Variable, np.ndarray]
        r angular velocity component (yaw rate)
    phi : Union[float, int, csdl.Variable, np.ndarray]
        phi euler angle
    theta : Union[float, int, csdl.Variable, np.ndarray]
        theta euler angle
    psi : Union[float, int, csdl.Variable, np.ndarray]
        psi euler angle
    x : Union[float, int, csdl.Variable, np.ndarray]
        x position (not used in rotor analysis)
    y : Union[float, int, csdl.Variable, np.ndarray]
        y position (not used in rotor analysis)
    z : Union[float, int, csdl.Variable, np.ndarray]
        z position (not used in rotor analysis)
    """
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
    ac_states = None
    theta_0 : Union[csdl.Variable, int, float] = 0
    theta_1_c : Union[csdl.Variable, int, float] = 0
    theta_1_s : Union[csdl.Variable, int, float] = 0
    xi_0 : Union[csdl.Variable, int, float] = 0
    xi_1_c : Union[csdl.Variable, int, float] = 0
    xi_1_s : Union[csdl.Variable, int, float] = 0

@dataclass
class RotorAnalysisOutputs(csdl.VariableGroup):
    """Rotor analysis outputs data class.

    Parameters
    ----------
    axial_induced_velocity : csdl.Variable
        Axial induced velocity (at rotor disk)
    tangential_induced_velocity : csdl.Variable
        Tangential induced velocity (at rotor disk)
    sectional_thrust : csdl.Variable
        Sectional thrust dT (N)
    sectional_torque : csdl.Variable
        Sectional torque dQ (N.m)
    sectional_drag : csdl.Variable
        Sectional drag dD (N)
    total_thrust : csdl.Variable
        Total thrust (N)
    total_torque : csdl.Variable
        Total torque (N.m)
    total_power : csdl.Variable
        Total power (W)
    efficiency : csdl.Variable
        Efficiency
    figure_of_merit : csdl.Variable
        Figure of merit (hover efficiency metric)
    thrust_coefficient : csdl.Variable
        Thrust coefficient
    torque_coefficient : csdl.Variable
        Torque coefficient
    power_coefficient : csdl.Variable
        Power coefficient
    residual : csdl.Variable, optional
        Residual (if available), by default None
    forces : csdl.Variable, optional
        Forces (if available based on body-fixed reference 
        frame specified by ACStates), by default None
    moments : csdl.Variable, optional
        Moments (if available based on body-fixed reference 
        frame specified by ACStates) about specified reference point, 
        by default None
    sectional_inflow_angle : csdl.Variable, optional
        Sectional inflow angle (if available), by default None
    """

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
    


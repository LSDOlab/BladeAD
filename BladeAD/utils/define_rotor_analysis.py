import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
from BladeAD.core.peters_he.peters_he_model import PetersHeModel
import numpy as np
from typing import Union
from BladeAD.utils.var_groups import RotorAnalysisOutputs


def run_rotor_analysis(
    model : str,
    radius : Union[float, csdl.Variable],
    chord_profile : Union[np.ndarray, csdl.Variable],
    twist_profile : Union[np.ndarray, csdl.Variable],
    rpm : Union[float, csdl.Variable],
    mesh_velocity : Union[np.ndarray, csdl.Variable],
    thrust_vector : Union[np.ndarray, csdl.Variable] = np.array([1, 0, 0]),
    thrust_origin : Union[np.ndarray, csdl.Variable] =np.array([0., 0., 0.]),
    num_nodes : int = 1,
    num_radial : int = 35,
    num_azimuthal : int = 50,
    num_blades : int = 4,
    norm_hub_radius : float = 0.2,
    theta_0 : Union[float, csdl.Variable] = 0.0,
    theta_1_c : Union[float, csdl.Variable] = 0.0,
    theta_1_s : Union[float, csdl.Variable] = 0.0,
    xi_0 : Union[float, csdl.Variable] = 0.0,
    xi_1_c : Union[float, csdl.Variable] = 0.0,
    xi_1_s : Union[float, csdl.Variable] = 0.0,
    Q : int = 3,
    M : int = 3,
    tip_loss : bool = True,
    airfoil_model=NACA4412MLAirfoilModel(),
    integration_scheme='trapezoidal',
) -> RotorAnalysisOutputs:
    """Run rotor analysis using either BEM, Pitt-Peters, and Peters-He models.


    Parameters
    ----------
    model : str
        Model to run. Options are "bem", "pitt_peters", or "peters_he"
    radius : Union[float, csdl.Variable]
        Radius of the rotor
    chord_profile : Union[np.ndarray, csdl.Variable]
        Chord profile of the rotor
    twist_profile : Union[np.ndarray, csdl.Variable]
        Twist profile of the rotor
    rpm : Union[float, csdl.Variable]
        Rotor speed in RPM
    mesh_velocity : Union[np.ndarray, csdl.Variable]
        Mesh velocity
    thrust_vector : Union[np.ndarray, csdl.Variable], optional
        Thrust vector, by default np.array([1, 0, 0])
    thrust_origin : Union[np.ndarray, csdl.Variable], optional
        Thrust origin, by default np.array([0., 0., 0.])
    num_nodes : int, optional
        Number of evaluation points, by default 1
    num_radial : int, optional
        Number of radial sections, by default 35
    num_azimuthal : int, optional
        Number of azimuthal sections, by default 50
    num_blades : int, optional
        Number of blades, by default 4
    norm_hub_radius : float, optional
        Normalized hub radius, by default 0.2
    Q : int, optional
        Highest power of r/R in Peters-He model, by default 3
        only used if model is "peters_he"
    M : int, optional
        Highest harmonic number in Peters-He model, by default 3
        only used if model is "peters_he"
    airfoil_model : NACA4412MLAirfoilModel, optional
        Airfoil model, by default NACA4412MLAirfoilModel()  

    Returns
    -------
    RotorAnalysisOutputs
        Rotor analysis outputs. Data class that stores rotor analysis
        outputs such as total thrust, total torque, etc.
    """
    csdl.check_parameter(value=model, name="model", values=["bem", "pitt_peters", "peters_he"])
    csdl.check_parameter(value=radius, name="radius", types=(float, csdl.Variable))
    csdl.check_parameter(value=chord_profile, name="chord_profile", types=(np.ndarray, csdl.Variable))
    csdl.check_parameter(value=twist_profile, name="twist_profile", types=(np.ndarray, csdl.Variable))
    csdl.check_parameter(value=rpm, name="rpm", types=(float, csdl.Variable))
    csdl.check_parameter(value=mesh_velocity, name="mesh_velocity", types=(np.ndarray, csdl.Variable))
    csdl.check_parameter(value=thrust_vector, name="thrust_vector", types=(np.ndarray, csdl.Variable))
    csdl.check_parameter(value=thrust_origin, name="thrust_origin", types=(np.ndarray, csdl.Variable))
    csdl.check_parameter(value=num_nodes, name="num_nodes", types=(int, ))
    csdl.check_parameter(value=num_radial, name="num_radial", types=(int, ))
    csdl.check_parameter(value=num_azimuthal, name="num_azimuthal", types=(int, ))
    csdl.check_parameter(value=num_blades, name="num_blades", types=(int, ))

    # Rotor mesh parameters
    mesh_parameters = RotorMeshParameters(
        thrust_vector=thrust_vector,
        thrust_origin=thrust_origin,
        chord_profile=chord_profile,
        twist_profile=twist_profile, 
        radius=radius,
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
        norm_hub_radius=norm_hub_radius,
    )

    # Set up rotor analysis inputs
    rotor_analysis_inputs = RotorAnalysisInputs(
        rpm=rpm,
        mesh_velocity=mesh_velocity,
        mesh_parameters=mesh_parameters,
        theta_0=theta_0,
        theta_1_c=theta_1_c,
        theta_1_s=theta_1_s,
        xi_0=xi_0,
        xi_1_c=xi_1_c,
        xi_1_s=xi_1_s,
    )
    
    if model == "bem":
        bem_model = BEMModel(
            num_nodes=num_nodes,
            airfoil_model=airfoil_model,
            integration_scheme=integration_scheme,
            tip_loss=tip_loss,
        )
        outputs = bem_model.evaluate(rotor_analysis_inputs)
    
    elif model == "pitt_peters":
        pitt_peters_model = PittPetersModel(
            num_nodes=num_nodes,
            airfoil_model=airfoil_model,
            integration_scheme=integration_scheme,
            tip_loss=tip_loss,
        )
        outputs = pitt_peters_model.evaluate(rotor_analysis_inputs)
    
    else:
        peters_he_model = PetersHeModel(
            num_nodes=num_nodes,
            airfoil_model=airfoil_model,
            integration_scheme=integration_scheme,
            Q=Q,
            M=M,
            tip_loss=tip_loss,
        )
        outputs = peters_he_model.evaluate(rotor_analysis_inputs)
    
    return outputs
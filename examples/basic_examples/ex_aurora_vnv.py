"""Example aurora vnv"""
import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.peters_he.peters_he_model import PetersHeModel
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
import numpy as np


recorder = csdl.Recorder(inline=True)
recorder.start()

do_pitt_peters = True
do_peters_he = True

num_nodes = 1
num_radial = 30
num_azimuthal = 30 # NOTE: for edgewise flight this needs to be greater than 1

num_blades = 2

# 3-D airfoil model
naca_0012_3d = ThreeDAirfoilMLModelMaker(
    airfoil_name="naca_0012",
        aoa_range=np.linspace(-12, 16, 50), 
        reynolds_range=[2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 4e6], 
        mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
)

# 2-D airfoil model
naca_0012_2d_model = TwoDMLAirfoilModel(
    airfoil_name="naca_0012",
)

airfoil_model = naca_0012_2d_model #NACA4412MLAirfoilModel() # naca_0012_3d.get_airfoil_model(quantities=["Cl", "Cd"])

polar_parameters_1 = ZeroDAirfoilPolarParameters(
    alpha_stall_minus=-15.,
    alpha_stall_plus=15.,
    Cl_stall_minus=-1.2,
    Cl_stall_plus=1.2,
    Cd_stall_minus=0.02,
    Cd_stall_plus=0.02,
    Cl_0=0.0,
    Cd_0=0.005,
    Cl_alpha=6.28,
)
airfoil_model_1 = ZeroDAirfoilModel(
    polar_parameters=polar_parameters_1,
)

tilt_angles = np.deg2rad(np.array([0]))#, 10, 20, 45, 75, 90]))

bem_thrust = []
bem_torque = []
pp_thrust = []
pp_torque = []
ph_thrust = []
ph_torque = []

for tilt_angle in tilt_angles:
    radius = csdl.Variable(value=1.524)
    thrust_vector=csdl.Variable(value=np.array([1 * np.sin(tilt_angle), 0, -1 * np.cos(tilt_angle)])) # NOTE [0, 0, -1] would be up; [1, 0, 0] would be forward (toward nose of AC)
    thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
    chord_profile=csdl.Variable(value=np.linspace(0.3*1.524, 0.04*1.524, num_radial))
    twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(23), np.deg2rad(3), num_radial))

    mesh_vel_np = np.zeros((num_nodes, 3))
    mesh_vel_np[:, 0] = 0 #np.linspace(1, 50., num_nodes) # np.linspace(0, 10.0, num_nodes) #
    # mesh_vel_np[:, 2] = -10.
    mesh_velocity = csdl.Variable(value=mesh_vel_np)
    # rpm = csdl.Variable(value=4400 * np.ones((num_nodes,)))
    rpm = csdl.Variable(value=1100. * np.ones((num_nodes,)))


    bem_mesh_parameters = RotorMeshParameters(
        thrust_vector=thrust_vector,
        thrust_origin=thrust_origin,
        chord_profile=chord_profile,
        twist_profile=twist_profile,
        radius=radius,
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
        norm_hub_radius=0.2,
    )

    inputs = RotorAnalysisInputs(
        rpm=rpm,
        mesh_parameters=bem_mesh_parameters,
        mesh_velocity=mesh_velocity,
    )

    bem_model = BEMModel(
        num_nodes=num_nodes,
        airfoil_model=airfoil_model,
        integration_scheme='trapezoidal',
    )

    pitt_peters_model = PittPetersModel(
        num_nodes=num_nodes,
        airfoil_model=airfoil_model,
        integration_scheme='trapezoidal',
    )

    peters_he_model = PetersHeModel(
        num_nodes=num_nodes,
        airfoil_model=airfoil_model,
        integration_scheme='trapezoidal',
        Q=4,
        M=4,
    )

    bem_outputs = bem_model.evaluate(inputs=inputs)
    T_bem = bem_outputs.total_thrust.value
    J_bem = bem_outputs.advance_ratio.value
    eta_bem = bem_outputs.efficiency.value
    Q_bem = bem_outputs.total_torque.value
    print("thrust bem", T_bem)
    print("torque bem", Q_bem)
    bem_thrust.append(T_bem)
    bem_torque.append(Q_bem)

    if do_pitt_peters:
        pitt_peters_outputs = pitt_peters_model.evaluate(inputs=inputs)
        T_pitt_peters = pitt_peters_outputs.total_thrust.value
        Q_pitt_peters = pitt_peters_outputs.total_torque.value
        print("thrust pitt_peters", T_pitt_peters)
        print("torque pitt_peters", Q_pitt_peters)
        pp_thrust.append(T_pitt_peters)
        pp_torque.append(Q_pitt_peters)

    if do_peters_he:
        peters_he_outputs = peters_he_model.evaluate(inputs=inputs)
        T_peters_he = peters_he_outputs.total_thrust.value
        Q_peters_he = peters_he_outputs.total_torque.value
        print("thrust peters_he", T_peters_he)
        print("torque peters_he", Q_peters_he)
        ph_thrust.append(T_peters_he)
        ph_torque.append(Q_peters_he)


recorder.stop()



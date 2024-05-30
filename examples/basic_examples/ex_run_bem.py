import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.utils.plot import make_polarplot
import numpy as np
from time import time


recorder = csdl.Recorder(inline=True)
recorder.start()

num_nodes = 1
num_radial = 51
num_azimuthal = 30

num_blades = 2

thrust_vector=csdl.Variable(value=np.array([0., 0, -1]))
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
# chord_profile=csdl.Variable(value=np.linspace(0.25, 0.05, num_radial))
# twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(70), np.deg2rad(30), num_radial))
# radius = csdl.Variable(value=1.5)
chord_profile=csdl.Variable(value=np.linspace(0.031 ,0.012, num_radial))
twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_radial))
radius = csdl.Variable(value=0.3048 / 2)
mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = np.linspace(0, 10.0, num_nodes) # np.linspace(0, 50.06, num_nodes)
# mesh_vel_np[:, 2] = -10.
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=4400 * np.ones((num_nodes,)))

polar_parameters_1 = ZeroDAirfoilPolarParameters(
    alpha_stall_minus=-10.,
    alpha_stall_plus=15.,
    Cl_stall_minus=-1.,
    Cl_stall_plus=1.5,
    Cd_stall_minus=0.02,
    Cd_stall_plus=0.06,
    Cl_0=0.5,
    Cd_0=0.008,
    Cl_alpha=5.1566,
)

polar_parameters_2 = ZeroDAirfoilPolarParameters(
    alpha_stall_minus=-12.,
    alpha_stall_plus=15.,
    Cl_stall_minus=-1.,
    Cl_stall_plus=1.5,
    Cd_stall_minus=0.02,
    Cd_stall_plus=0.06,
    Cl_0=0.4,
    Cd_0=0.01,
    Cl_alpha=5.4566,
)

airfoil_model_1 = ZeroDAirfoilModel(
    polar_parameters=polar_parameters_1,
)

airfoil_model_2 = ZeroDAirfoilModel(
    polar_parameters=polar_parameters_2,
)

naca_4412_2d_model = TwoDMLAirfoilModel(
    airfoil_name="naca_4412",
)

clark_y_2d_model = TwoDMLAirfoilModel(
    airfoil_name="clark_y",
)

mh_117_2d_model = TwoDMLAirfoilModel(
    airfoil_name="mh_117",
)

airfoil_model = CompositeAirfoilModel(
    sections=[0., 0.5, 0.7, 1.],
    airfoil_models=[clark_y_2d_model, mh_117_2d_model, naca_4412_2d_model],
)


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

bem_inputs = RotorAnalysisInputs(
    rpm=rpm,
    mesh_parameters=bem_mesh_parameters,
    mesh_velocity=mesh_velocity,
)

bem_model = BEMModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
)

t1 = time()
outputs = bem_model.evaluate(inputs=bem_inputs)
t2 = time()
print("time", t2-t1)
T = outputs.total_thrust.value
Q = outputs.total_torque.value

print("thrust", T)
print("torque", Q)
print("eta", outputs.efficiency.value)

print(outputs.residual.value.max())
print(outputs.residual.value.min())

make_polarplot(outputs.sectional_thrust, quantity_name='dT', plot_contours=False, fig_size=(14, 14), azimuthal_offset=0)

recorder.stop()



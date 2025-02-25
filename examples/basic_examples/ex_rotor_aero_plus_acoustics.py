"""Example rotor aero + acoustics script"""
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
from lsdo_acoustics.core.models.broadband.GL.GL_model import GL_model, GLVariableGroup
from lsdo_acoustics.core.models.total_noise_model import total_noise_model
from lsdo_acoustics.core.models.tonal.Lowson.Lowson_model import Lowson_model, LowsonVariableGroup
from lsdo_acoustics import Acoustics


recorder = csdl.Recorder(inline=True)
recorder.start()

num_nodes = 1
num_radial = 30
num_azimuthal = 45 # NOTE: for edgewise flight (or dynamic inflow models) this needs to be greater than 1

do_bem = True
do_pitt_peters = False
do_peters_he = False

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

airfoil_model = naca_0012_2d_model# naca_0012_3d.get_airfoil_model(quantities=["Cl", "Cd"])

tilt_angle = 0
radius = csdl.Variable(value=1.524)
thrust_vector=csdl.Variable(value=np.array([1 * np.sin(tilt_angle), 0, -1 * np.cos(tilt_angle)])) # NOTE [0, 0, -1] would be up; [1, 0, 0] would be forward (toward nose of AC)
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
chord_profile=csdl.Variable(value=np.linspace(0.3*1.524, 0.04*1.524, num_radial))
twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(23), np.deg2rad(3), num_radial))

mesh_vel_np = np.zeros((num_nodes, 3))
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=1100. * np.ones((num_nodes,)))

mesh_parameters = RotorMeshParameters(
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
    mesh_parameters=mesh_parameters,
    mesh_velocity=mesh_velocity,
)

bem_model = BEMModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
)
if do_bem:
    rotor_aero_outputs = bem_model.evaluate(inputs)

pitt_peters_model = PittPetersModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
)
if do_pitt_peters:
    rotor_aero_outputs = pitt_peters_model.evaluate(inputs)

peters_he_model = PetersHeModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
    Q=4, # highest power of r/R
    M=4, # highest harmonic number
)
if do_peters_he:
    rotor_aero_outputs = peters_he_model.evaluate(inputs)

print("total thrust", rotor_aero_outputs.total_thrust.value)
print("total torque", rotor_aero_outputs.total_torque.value)
print("FOM", rotor_aero_outputs.figure_of_merit.value)


# -------------------------  ACOUSTICS -------------------------
observer_radius = 80
observer_angle = np.deg2rad(75)
# NOTE: can vary position of aircraft or observer (I suggest keeping the observer at [0, 0, 0])
broadband_acoustics = Acoustics(aircraft_position=np.array([observer_radius * np.sin(observer_angle) ,0., -observer_radius * np.cos(observer_angle)]))
broadband_acoustics.add_observer('obs', np.array([0., 0., 0.,]), time_vector=np.array([0.]))
observer_data = broadband_acoustics.assemble_observers()

gl_vg = GLVariableGroup(
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    rotor_radius=radius,
    CT=rotor_aero_outputs.thrust_coefficient,
    chord_profile=chord_profile,
    mach_number=0.,
    rpm=rpm,
    num_radial=num_radial,
    num_tangential=num_azimuthal, # same as num_azimuthal
    speed_of_sound=340, # m/s
)

gl_spl, gl_spl_A_weighted = GL_model(
    GLVariableGroup=gl_vg,
    observer_data=observer_data,
    num_blades=2,
    num_nodes=1,
    A_weighting=True,
)

r_nondim = csdl.linear_combination(0.2*radius+0.01, radius-0.01, num_radial)/radius

lowson_vg = LowsonVariableGroup(
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    RPM=rpm,
    speed_of_sound=340,
    rotor_radius=radius,
    mach_number=0., 
    density=1.225,
    num_radial=num_radial,
    num_tangential=num_azimuthal,
    dD=rotor_aero_outputs.sectional_drag,
    dT=rotor_aero_outputs.sectional_thrust,
    phi=rotor_aero_outputs.sectional_inflow_angle,
    chord_profile=chord_profile,
    nondim_sectional_radius=r_nondim.flatten(),
    thickness_to_chord_ratio=0.12 * chord_profile,
)

lowson_spl, lowson_spl_A_weighted  = Lowson_model(
    LowsonVariableGroup=lowson_vg,
    observer_data=observer_data,
    num_blades=2,
    num_nodes=1,
    modes=[1, 2, 3],
    A_weighting=True,
    toggle_thickness_noise=True,
)

total_spl = total_noise_model(SPL_list=[lowson_spl_A_weighted] + [gl_spl_A_weighted])

print("total A-weighted SPL", total_spl.value)

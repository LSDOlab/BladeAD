import csdl_alpha as csdl
from BladeAD.utils.plot import make_polarplot
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
do_pitt_peters = True
do_peters_he = True

num_blades = 4

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

airfoil_model = naca_0012_2d_model#  naca_0012_3d.get_airfoil_model(quantities=["Cl", "Cd"]) #  

tilt_angle = np.deg2rad(90.)
R = 1.372 #1.524 #
radius = csdl.Variable(value=R)
thrust_vector=csdl.Variable(value=np.array([1 * np.sin(tilt_angle), 0, -1 * np.cos(tilt_angle)])) # NOTE [0, 0, -1] would be up; [1, 0, 0] would be forward (toward nose of AC)
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
chord_profile=csdl.Variable(value=np.linspace(0.3*R, 0.04*R, num_radial))
twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(50), np.deg2rad(15), num_radial))

mesh_vel_np = np.zeros((num_nodes, 3))
mesh_vel_np[:, 0] = 30.87 #57.6 # 0. # 112 kts
mesh_velocity = csdl.Variable(value=mesh_vel_np)
rpm = csdl.Variable(value=1000. * np.ones((num_nodes,)))

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
    ac_states="hello",
)

bem_model = BEMModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
    tip_loss=True,
)
if do_bem:
    rotor_aero_outputs = bem_model.evaluate(inputs)
    print("\nBEM")
    print("total thrust", rotor_aero_outputs.total_thrust.value)
    print("total torque", rotor_aero_outputs.total_torque.value)
    print("FOM", rotor_aero_outputs.figure_of_merit.value)
    print("eta", rotor_aero_outputs.efficiency.value)
    print("CT", rotor_aero_outputs.thrust_coefficient.value * 4 / np.pi**3)
    print("CQ", rotor_aero_outputs.torque_coefficient.value * 8 / np.pi**3)
    print("\n")

pitt_peters_model = PittPetersModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
    tip_loss=True,
)
if do_pitt_peters:
    rotor_aero_outputs = pitt_peters_model.evaluate(inputs)
    print("\nPitt-Peters")
    print("total thrust", rotor_aero_outputs.total_thrust.value)
    print("total torque", rotor_aero_outputs.total_torque.value)
    print("FOM", rotor_aero_outputs.figure_of_merit.value)
    print("eta", rotor_aero_outputs.efficiency.value)
    print("CT", rotor_aero_outputs.thrust_coefficient.value * 4 / np.pi**3)
    print("CQ", rotor_aero_outputs.torque_coefficient.value * 8 / np.pi**3)
    print("\n")
    make_polarplot(rotor_aero_outputs.sectional_thrust, plot_contours=True)


peters_he_model = PetersHeModel(
    num_nodes=num_nodes,
    airfoil_model=airfoil_model,
    integration_scheme='trapezoidal',
    Q=4, # highest power of r/R
    M=4, # highest harmonic number
    tip_loss=True,
)
if do_peters_he:
    rotor_aero_outputs = peters_he_model.evaluate(inputs)
    print("\nPeters-He")
    print("total thrust", rotor_aero_outputs.total_thrust.value)
    print("total torque", rotor_aero_outputs.total_torque.value)
    print("FOM", rotor_aero_outputs.figure_of_merit.value)
    print("eta", rotor_aero_outputs.efficiency.value)
    print("CT", rotor_aero_outputs.thrust_coefficient.value * 4 / np.pi**3)
    print("CQ", rotor_aero_outputs.torque_coefficient.value * 8 / np.pi**3)
    print("\n")
    make_polarplot(rotor_aero_outputs.sectional_thrust, plot_contours=True)
exit()
# -------------------------  ACOUSTICS -------------------------
observer_radius = 20. # meters
observer_angle = np.deg2rad(1)
# NOTE: can vary position of aircraft or observer (I suggest keeping the observer at [0, 0, 0])
# broadband_acoustics = Acoustics(aircraft_position=np.array([observer_radius * np.sin(observer_angle) ,0., observer_radius * np.cos(observer_angle)]))
broadband_acoustics = Acoustics(aircraft_position=np.array([observer_radius,0.,1.])) # In rotor plane (slightly offset in z)
# broadband_acoustics = Acoustics(aircraft_position=np.array([0.,0.,observer_radius])) # Out of rotor plane (along rotor axis)
broadband_acoustics.add_observer('obs', np.array([0., 0., 0.,]), time_vector=np.array([0.]))
observer_data = broadband_acoustics.assemble_observers()

gl_vg = GLVariableGroup(
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    rotor_radius=radius,
    CT=rotor_aero_outputs.thrust_coefficient*4/np.pi**3, #rotor aero outputs is propeller coefficient; the acoustics code needs rotor coefficient.
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
    thickness_to_chord_ratio=0.12,
)

lowson_spl, lowson_spl_dBA, thickness_noise, thickness_noise_dBA, loading_noise, loading_noise_dBA = Lowson_model(
    LowsonVariableGroup=lowson_vg,
    observer_data=observer_data,
    num_blades=2,
    num_nodes=1,
    modes=[1, 2, 3, 4],
    A_weighting=True,
    toggle_thickness_noise=True,
)

thickness_spl = total_noise_model(SPL_list=[thickness_noise])
loading_spl = total_noise_model(SPL_list=[loading_noise])
total_spl = total_noise_model(SPL_list=[lowson_spl] + [gl_spl])

thickness_spl_dBA = total_noise_model(SPL_list=[thickness_noise_dBA])
loading_spl_dBA = total_noise_model(SPL_list=[loading_noise_dBA])
total_Aweighted_spl = total_noise_model(SPL_list=[lowson_spl_dBA] + [gl_spl_A_weighted])

print("Thickness SPL", thickness_spl.value)
print("Loading SPL", loading_spl.value)
print("total Overall SPL", total_spl.value)

print("Thickness A-weighted SPL", thickness_spl_dBA.value)
print("Loading A-weighted SPL", loading_spl_dBA.value)
print("total A-weighted SPL", total_Aweighted_spl.value)
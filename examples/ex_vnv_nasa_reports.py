"""V & V Script for NASA Inflow Measurement Reports"""
import csdl_alpha as csdl
import numpy as np
from BladeAD.utils.define_rotor_analysis import run_rotor_analysis
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD import NASA_INFLOW_DATA_FOLDER
import pandas as pd
from BladeAD.utils.verify_dynamic_inflow_models_plotting import plot_vnv_data


# start CSDL recorder
recorder = csdl.Recorder(inline=True)
recorder.start()

# Discretization & rotor geometry definition
num_nodes = 1
num_radial = 50
num_azimuthal = 100
num_blades = 4

radius = csdl.Variable(name="radius", value=0.860552)
chord_profile = csdl.Variable(
    name="chord_profile", 
    shape=(num_radial, ), 
    value=0.06604
)

# Note that the twist profile is linearly varying by 8 degrees
# and is such that the twist is zero at 3/4 of the blade radius
root_twist = np.deg2rad(8 - 2.470588235)
tip_twist = root_twist - np.deg2rad(8)
twist_profile = csdl.Variable(
    name="twist_profile", 
    shape=(num_radial, ), 
    value=np.linspace(root_twist, tip_twist, num_radial)
)

# The rotor blade uses a NACA 0012 airfoil
airfoil_model = TwoDMLAirfoilModel(
    airfoil_name="naca_0012",
)

# Thrust origin is the same for all test cases
thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))

# --------------------NASA test case 1--------------------
alpha = -3.00 # rotor disk tilt angle, i.e., angle of attack
tilt_angle = np.deg2rad(90 + alpha)
tx = np.cos(tilt_angle)
tz = np.sin(tilt_angle)
thrust_vector=csdl.Variable(value=np.array([tx, 0, -tz]))
mesh_velocity_np = np.zeros((num_nodes, 3))
mesh_velocity_np[:, 0] = 28.50 # Free stream velocity
mesh_velocity = csdl.Variable(name="mesh_velocity", shape=(num_nodes, 3), value=mesh_velocity_np)
rpm = csdl.Variable(name="rpm", shape=(num_nodes,), value=2113)

# Read in the NASA data
df_1 = pd.read_csv(f"{NASA_INFLOW_DATA_FOLDER}/nasa_report_rotor_inflow_data_mu_015.csv")

# ----- Run Pitt-Peters model -----
pitt_peters_ouputs = run_rotor_analysis(
    model="pitt_peters",
    radius=radius,
    chord_profile=chord_profile,
    twist_profile=twist_profile,
    rpm=rpm,
    mesh_velocity=mesh_velocity,
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    num_nodes=num_nodes,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    num_blades=num_blades,
    theta_0=np.deg2rad(9.37),
    theta_1_c=np.deg2rad(1.11),
    theta_1_s=np.deg2rad(-3.23),
    xi_0=np.deg2rad(-0.95),
    airfoil_model=airfoil_model,
)

inflow_pp = pitt_peters_ouputs.inflow
C_T_pp = pitt_peters_ouputs.thrust_coefficient.value

# Plot the inflow data for pitt-peters model
plot_vnv_data(
    nasa_data_df=df_1,
    nasa_C_T=0.0064,
    blade_ad_data=[inflow_pp.value],
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    model_name=["Pitt--Peters model"],
    model_C_T=[C_T_pp[0]],
)

# ----- Run Peters-He model for different number of flow states-----
def run_peters_he(Q, M):
    return run_rotor_analysis(
        model="peters_he",
        radius=radius,
        chord_profile=chord_profile,
        twist_profile=twist_profile,
        rpm=rpm,
        mesh_velocity=mesh_velocity,
        thrust_vector=thrust_vector,
        thrust_origin=thrust_origin,
        num_nodes=num_nodes,
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
        theta_0=np.deg2rad(9.37),
        theta_1_c=np.deg2rad(1.11),
        theta_1_s=np.deg2rad(-3.23),
        xi_0=np.deg2rad(-0.95),
        airfoil_model=airfoil_model,
        Q=Q,
        M=M,
    )

peters_he_outputs = [run_peters_he(Q, M) for Q, M in [(2, 1), (4, 3), (6, 5)]]

# Extract inflow and thrust coefficient values
inflow_ph = [output.inflow.value for output in peters_he_outputs]
C_T_ph = [output.thrust_coefficient.value[0] for output in peters_he_outputs]

# Plot Peters-He model results
plot_vnv_data(
    nasa_data_df=df_1,
    nasa_C_T=0.0064,
    blade_ad_data=inflow_ph,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    model_name=[
        "Peters--He w/ 6 flow states",
        "Peters--He w/ 15 flow states",
        "Peters--He w/ 28 flow states",
    ],
    model_C_T=C_T_ph,
)

# --------------------NASA test case 2--------------------
alpha = -3.04 # rotor disk tilt angle, i.e., angle of attack
tilt_angle = np.deg2rad(90 + alpha)
tx = np.cos(tilt_angle)
tz = np.sin(tilt_angle)
thrust_vector=csdl.Variable(value=np.array([tx, 0, -tz]))
mesh_velocity_np = np.zeros((num_nodes, 3))
mesh_velocity_np[:, 0] = 43.8605044 # Free stream velocity
mesh_velocity = csdl.Variable(name="mesh_velocity", shape=(num_nodes, 3), value=mesh_velocity_np)
rpm = csdl.Variable(name="rpm", shape=(num_nodes,), value=2113)

# Read in the NASA data
df_2 = pd.read_csv(f"{NASA_INFLOW_DATA_FOLDER}/nasa_report_rotor_inflow_data_mu_023.csv")

# ----- Run Pitt-Peters model -----
pitt_peters_ouputs = run_rotor_analysis(
    model="pitt_peters",
    radius=radius,
    chord_profile=chord_profile,
    twist_profile=twist_profile,
    rpm=rpm,
    mesh_velocity=mesh_velocity,
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    num_nodes=num_nodes,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    num_blades=num_blades,
    theta_0=np.deg2rad(8.16),
    theta_1_c=np.deg2rad(1.52),
    theta_1_s=np.deg2rad(-4.13),
    xi_0=np.deg2rad(-0.9),
    airfoil_model=airfoil_model,
)

inflow_pp = pitt_peters_ouputs.inflow
C_T_pp = pitt_peters_ouputs.thrust_coefficient.value

# Plot the inflow data for pitt-peters model
plot_vnv_data(
    nasa_data_df=df_2,
    nasa_C_T=0.0064,
    blade_ad_data=[inflow_pp.value],
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    model_name=["Pitt--Peters model"],
    model_C_T=[C_T_pp[0]],
)

# ----- Run Peters-He model for different number of flow states-----
def run_peters_he(Q, M):
    return run_rotor_analysis(
        model="peters_he",
        radius=radius,
        chord_profile=chord_profile,
        twist_profile=twist_profile,
        rpm=rpm,
        mesh_velocity=mesh_velocity,
        thrust_vector=thrust_vector,
        thrust_origin=thrust_origin,
        num_nodes=num_nodes,
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
        theta_0=np.deg2rad(8.16),
        theta_1_c=np.deg2rad(1.52),
        theta_1_s=np.deg2rad(-4.13),
        xi_0=np.deg2rad(-0.9),
        airfoil_model=airfoil_model,
        Q=Q,
        M=M,
    )

peters_he_outputs = [run_peters_he(Q, M) for Q, M in [(2, 2), (4, 4), (6, 6)]]

# Extract inflow and thrust coefficient values
inflow_ph = [output.inflow.value for output in peters_he_outputs]
C_T_ph = [output.thrust_coefficient.value[0] for output in peters_he_outputs]

# Plot Peters-He model results
plot_vnv_data(
    nasa_data_df=df_2,
    nasa_C_T=0.0064,
    blade_ad_data=inflow_ph,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    model_name=[
        "Peters--He w/ 6 flow states",
        "Peters--He w/ 15 flow states",
        "Peters--He w/ 28 flow states",
    ],
    model_C_T=C_T_ph,
)


# --------------------NASA test case 3--------------------
alpha = -5.7 # rotor disk tilt angle, i.e., angle of attack
tilt_angle = np.deg2rad(90 + alpha)
tx = np.cos(tilt_angle)
tz = np.sin(tilt_angle)
thrust_vector=csdl.Variable(value=np.array([tx, 0, -tz]))
mesh_velocity_np = np.zeros((num_nodes, 3))
mesh_velocity_np[:, 0] = 66.75 # Free stream velocity
mesh_velocity = csdl.Variable(name="mesh_velocity", shape=(num_nodes, 3), value=mesh_velocity_np)
rpm = csdl.Variable(name="rpm", shape=(num_nodes,), value=2113)

# Read in the NASA data
df_3 = pd.read_csv(f"{NASA_INFLOW_DATA_FOLDER}/nasa_report_rotor_inflow_data_mu_35.csv")

# ----- Run Pitt-Peters model -----
pitt_peters_ouputs = run_rotor_analysis(
    model="pitt_peters",
    radius=radius,
    chord_profile=chord_profile,
    twist_profile=twist_profile,
    rpm=rpm,
    mesh_velocity=mesh_velocity,
    thrust_vector=thrust_vector,
    thrust_origin=thrust_origin,
    num_nodes=num_nodes,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    num_blades=num_blades,
    theta_0=np.deg2rad(9.2),
    theta_1_c=np.deg2rad(0.3),
    theta_1_s=np.deg2rad(-6.80),
    xi_0=np.deg2rad(-1.3),
    airfoil_model=airfoil_model,
)

inflow_pp = pitt_peters_ouputs.inflow
C_T_pp = pitt_peters_ouputs.thrust_coefficient.value

# Plot the inflow data for pitt-peters model
plot_vnv_data(
    nasa_data_df=df_3,
    nasa_C_T=0.0064,
    blade_ad_data=[inflow_pp.value],
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    model_name=["Pitt--Peters model"],
    model_C_T=[C_T_pp[0]],
)

# ----- Run Peters-He model for different number of flow states-----
def run_peters_he(Q, M):
    return run_rotor_analysis(
        model="peters_he",
        radius=radius,
        chord_profile=chord_profile,
        twist_profile=twist_profile,
        rpm=rpm,
        mesh_velocity=mesh_velocity,
        thrust_vector=thrust_vector,
        thrust_origin=thrust_origin,
        num_nodes=num_nodes,
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
        theta_0=np.deg2rad(9.2),
        theta_1_c=np.deg2rad(0.3),
        theta_1_s=np.deg2rad(-6.80),
        xi_0=np.deg2rad(-1.3),
        airfoil_model=airfoil_model,
        Q=Q,
        M=M,
    )

peters_he_outputs = [run_peters_he(Q, M) for Q, M in [(2, 2), (4, 4), (6, 6)]]

# Extract inflow and thrust coefficient values
inflow_ph = [output.inflow.value for output in peters_he_outputs]
C_T_ph = [output.thrust_coefficient.value[0] for output in peters_he_outputs]

# Plot Peters-He model results
plot_vnv_data(
    nasa_data_df=df_3,
    nasa_C_T=0.0064,
    blade_ad_data=inflow_ph,
    num_radial=num_radial,
    num_azimuthal=num_azimuthal,
    model_name=[
        "Peters--He w/ 6 flow states",
        "Peters--He w/ 15 flow states",
        "Peters--He w/ 28 flow states",
    ],
    model_C_T=C_T_ph,
)
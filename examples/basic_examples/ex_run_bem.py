import csdl_alpha as csdl
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters
from BladeAD.core.airfoil.zero_d_airfoil_model import ZeroDAirfoilModel, ZeroDAirfoilPolarParameters
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel
from BladeAD.core.airfoil.ml_airfoil_models.NACA_4412.naca_4412_model import NACA4412MLAirfoilModel
from BladeAD.core.airfoil.xfoil.two_d_airfoil_model import TwoDMLAirfoilModel
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.peters_he.peters_he_model import PetersHeModel
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
from BladeAD.utils.plot import make_polarplot
import numpy as np
from time import time


recorder = csdl.Recorder(inline=True)
recorder.start()

do_pitt_peters = True
do_peters_he = True

num_nodes = 10
num_radial = 35
num_azimuthal = 40

num_blades = 2

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
    sections=[0., 0.35, 0.7, 1.],
    airfoil_models=[naca_4412_2d_model, clark_y_2d_model, mh_117_2d_model],
)



tilt_angles = np.deg2rad(np.array([0, 10, 20, 45, 75, 90]))

bem_thrust = []
bem_torque = []
pp_thrust = []
pp_torque = []
ph_thrust = []
ph_torque = []
pp_error_thrust = []
pp_error_torque = []
ph_error_thrust = []
ph_error_torque = []

for tilt_angle in tilt_angles:
    thrust_vector=csdl.Variable(value=np.array([1 * np.sin(tilt_angle), 0, -1 * np.cos(tilt_angle)]))
    thrust_origin=csdl.Variable(value=np.array([0. ,0., 0.]))
    chord_profile=csdl.Variable(value=np.linspace(0.25, 0.05, num_radial))
    twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(50), np.deg2rad(20), num_radial))
    radius = csdl.Variable(value=1.2)
    # chord_profile=csdl.Variable(value=np.linspace(0.031 ,0.012, num_radial))
    # twist_profile=csdl.Variable(value=np.linspace(np.deg2rad(21.5), np.deg2rad(11.1), num_radial))
    # radius = csdl.Variable(value=0.3048001 / 2)
    mesh_vel_np = np.zeros((num_nodes, 3))
    mesh_vel_np[:, 0] = np.linspace(1, 50., num_nodes) # np.linspace(0, 10.0, num_nodes) # 
    # mesh_vel_np[:, 2] = -10.
    mesh_velocity = csdl.Variable(value=mesh_vel_np)
    # rpm = csdl.Variable(value=4400 * np.ones((num_nodes,)))
    rpm = csdl.Variable(value=2000 * np.ones((num_nodes,)))


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
    airfoil_model=NACA4412MLAirfoilModel(),
    integration_scheme='trapezoidal',
)

    pitt_peters_model = PittPetersModel(
        num_nodes=num_nodes,
        airfoil_model=NACA4412MLAirfoilModel(),
        integration_scheme='trapezoidal',
    )

    peters_he_model = PetersHeModel(
        num_nodes=num_nodes,
        airfoil_model=NACA4412MLAirfoilModel(),
        integration_scheme='trapezoidal',
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

    print("\n")
    if do_pitt_peters:
        percent_error_thrust_pp = (T_pitt_peters - T_bem) / T_bem * 100
        percent_error_torque_pp = (Q_pitt_peters - Q_bem) / Q_bem * 100
        print("percent error thrust pp", percent_error_thrust_pp)
        print("percent error torque pp", percent_error_torque_pp)
        print("mean percent error thrust pp", abs(percent_error_thrust_pp).mean())
        print("mean percent error torque pp", abs(percent_error_torque_pp).mean())
        pp_error_thrust.append(percent_error_thrust_pp)
        pp_error_torque.append(percent_error_torque_pp)


    if do_peters_he:
        percent_error_thrust_ph = (T_peters_he - T_bem) / T_bem * 100
        percent_error_torque_ph = (Q_peters_he - Q_bem) / Q_bem * 100
        print("percent error thrust ph", percent_error_thrust_ph)
        print("percent error torque ph", percent_error_torque_ph)
        print("mean percent error thrust ph", abs(percent_error_thrust_ph).mean())
        print("mean percent error torque ph", abs(percent_error_torque_ph).mean())
        ph_error_thrust.append(percent_error_thrust_ph)
        ph_error_torque.append(percent_error_torque_ph)


import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3)

if do_pitt_peters:
    axs[0, 0].plot(bem_torque[0], bem_torque[0], color='k')
    axs[0, 0].scatter(pp_torque[0], bem_torque[0], label="Pitt--Peters")

    axs[0, 1].plot(bem_torque[1], bem_torque[1], color='k')
    axs[0, 1].scatter(pp_torque[1], bem_torque[1])
    
    axs[0, 2].plot(bem_torque[2], bem_torque[2], color='k')
    axs[0, 2].scatter(pp_torque[2], bem_torque[2])
    
    axs[1, 0].plot(bem_torque[3], bem_torque[3], color='k')
    axs[1, 0].scatter(pp_torque[3], bem_torque[3])
    
    axs[1, 1].plot(bem_torque[4], bem_torque[4], color='k')
    axs[1, 1].scatter(pp_torque[4], bem_torque[4])
    
    axs[1, 2].plot(bem_torque[5], bem_torque[5], color='k')
    axs[1, 2].scatter(pp_torque[5], bem_torque[5])

if do_peters_he:
    axs[0, 0].scatter(ph_torque[0], bem_torque[0], label="Peters--He")
    axs[0, 0].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[0])} (edgewise)."  + "\n" + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_torque[0]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_torque[0]).mean(), 3)}")
    axs[0, 0].legend()

    axs[0, 1].scatter(ph_torque[1], bem_torque[1])
    axs[0, 1].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[1])}." + "\n" + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_torque[1]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_torque[1]).mean(), 3)}")
    
    axs[0, 2].scatter(ph_torque[2], bem_torque[2])
    axs[0, 2].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[2])}." + "\n" + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_torque[2]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_torque[2]).mean(), 3)}")
    
    axs[1, 0].scatter(ph_torque[3], bem_torque[3])
    axs[1, 0].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[3])}." + "\n"  + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_torque[3]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_torque[3]).mean(), 3)}")
    
    axs[1, 1].scatter(ph_torque[4], bem_torque[4])
    axs[1, 1].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[4])}." + "\n"  +f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_torque[4]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_torque[4]).mean(), 3)}")
    
    axs[1, 2].scatter(ph_torque[5], bem_torque[5])
    axs[1, 2].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[5])} (axial)." + "\n"  + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_torque[5]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_torque[5]).mean(), 3)}")

fig.suptitle("Torque comparison to BEM")

fig_2, axs_2 = plt.subplots(2, 3)

if do_pitt_peters:
    axs_2[0, 0].plot(bem_thrust[0], bem_thrust[0], color='k')
    axs_2[0, 0].scatter(pp_thrust[0], bem_thrust[0], label="Pitt--Peters")

    axs_2[0, 1].plot(bem_thrust[1], bem_thrust[1], color='k')
    axs_2[0, 1].scatter(pp_thrust[1], bem_thrust[1])
    
    axs_2[0, 2].plot(bem_thrust[2], bem_thrust[2], color='k')
    axs_2[0, 2].scatter(pp_thrust[2], bem_thrust[2])
    
    axs_2[1, 0].plot(bem_thrust[3], bem_thrust[3], color='k')
    axs_2[1, 0].scatter(pp_thrust[3], bem_thrust[3])
    
    axs_2[1, 1].plot(bem_thrust[4], bem_thrust[4], color='k')
    axs_2[1, 1].scatter(pp_thrust[4], bem_thrust[4])
    
    axs_2[1, 2].plot(bem_thrust[5], bem_thrust[5], color='k')
    axs_2[1, 2].scatter(pp_thrust[5], bem_thrust[5])

if do_peters_he:
    axs_2[0, 0].scatter(ph_thrust[0], bem_thrust[0], label="Peters--He")
    axs_2[0, 0].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[0])} (edgewise)."  + "\n" + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_thrust[0]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_thrust[0]).mean(), 3)}")
    axs_2[0, 0].legend()

    axs_2[0, 1].scatter(ph_thrust[1], bem_thrust[1])
    axs_2[0, 1].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[1])}." + "\n" + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_thrust[1]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_thrust[1]).mean(), 3)}")
    
    axs_2[0, 2].scatter(ph_thrust[2], bem_thrust[2])
    axs_2[0, 2].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[2])}." + "\n" + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_thrust[2]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_thrust[2]).mean(), 3)}")
    
    axs_2[1, 0].scatter(ph_thrust[3], bem_thrust[3])
    axs_2[1, 0].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[3])}." + "\n"  + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_thrust[3]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_thrust[3]).mean(), 3)}")
    
    axs_2[1, 1].scatter(ph_thrust[4], bem_thrust[4])
    axs_2[1, 1].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[4])}." + "\n"  +f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_thrust[4]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_thrust[4]).mean(), 3)}")
    
    axs_2[1, 2].scatter(ph_thrust[5], bem_thrust[5])
    axs_2[1, 2].set_title(f"Rotor disk tile: {np.rad2deg(tilt_angles[5])} (axial)." + "\n"  + f"Mean percent difference Pitt--Peters to BEM {round(abs(pp_error_thrust[5]).mean(), 3)}" + "\n" + f"Mean percent difference Peters--He to BEM {round(abs(ph_error_thrust[5]).mean(), 3)}")
fig_2.suptitle("Thrust comparison to BEM")

plt.tight_layout()



#     axs[0, 0].scatter(T_peters_he, T_bem)
# axs[0].plot(T_bem, T_bem, color='k')
# # axs[0].plot(J_bem, eta_bem, color='k')
# # axs[0].plot([min_T, max_T], [min_T, max_T], color='red', linestyle='--', label='y=x')


# min_Q = min(Q_bem) #min(min(Q_bem), min(Q_pitt_peters), min(Q_peters_he))
# max_Q = max(Q_bem) # max(max(Q_bem), max(Q_pitt_peters), max(Q_peters_he))
# if do_pitt_peters:
#     axs[1].scatter(Q_pitt_peters, Q_bem)
# if do_peters_he:
#     axs[1].scatter(Q_peters_he, Q_bem)
# axs[1].plot(Q_bem, Q_bem, color='k')
# axs[1].plot([min_Q, max_Q], [min_Q, max_Q], color='red', linestyle='--', label='y=x')


plt.show()
# print("eta", outputs.efficiency.value)


# make_polarplot(peters_he_outputs.sectional_thrust, quantity_name='dT', plot_contours=False, fig_size=(14, 14), azimuthal_offset=-90)

recorder.stop()



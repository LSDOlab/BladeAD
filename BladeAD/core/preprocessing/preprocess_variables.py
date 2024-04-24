import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass


@dataclass
class PreProcessOutputs:
    thrust_vector_exp: csdl.Variable
    thrust_origin_vel_exp: csdl.Variable
    hub_radius: csdl.Variable
    element_width: csdl.Variable
    radius_vector_exp: csdl.Variable
    norm_radius_exp: csdl.Variable
    angular_speed_exp: csdl.Variable
    azimuth_angle_exp: csdl.Variable
    chord_profile_exp: csdl.Variable
    twist_profile_exp: csdl.Variable
    sigma: csdl.Variable


def preprocess_input_variables(
    shape: tuple,
    radius: csdl.Variable,
    chord_profile: csdl.Variable,
    twist_profile: csdl.Variable,
    rpm: csdl.Variable,
    norm_hub_radius: float,
    thrust_vector: csdl.Variable,
    origin_velocity: csdl.Variable,
    num_blades: int,
) -> PreProcessOutputs:
    # extract shape
    num_nodes = shape[0]
    num_radial = shape[1]
    num_azimuthal = shape[2]

    # expand thrust vector & origin velocity if necessary
    if thrust_vector.shape != (num_nodes, 3):
        thrust_vector_exp = csdl.expand(thrust_vector, (num_nodes, 3), action="j->ij")
    else:
        thrust_vector_exp = thrust_vector

    if origin_velocity.shape != (num_nodes, 3):
        thrust_origin_vel_exp = csdl.expand(origin_velocity, (num_nodes, 3))
    else:
        thrust_origin_vel_exp = origin_velocity

    # compute blade element width
    r_hub = norm_hub_radius * radius
    dr = (radius - 0.2 * radius) / (num_radial - 1)

    # compute normalized radius ]0, 1[
    # (the values correspond to the center of a blade element)
    norm_radius_linspace = 1.0 / num_radial / 2.0 + np.linspace(
        0.0, 1.0 - 1.0 / num_radial, num_radial
    )
    norm_radius_exp = csdl.Variable(
        value=np.einsum(
            "ik,j->ijk",
            np.ones((num_nodes, num_azimuthal)),
            norm_radius_linspace,
        )
    )

    # compute the radius vector (hub to tip)
    radius_vec_exp = r_hub + (radius - r_hub) * norm_radius_exp

    # convert rpm to rad / s
    if rpm.shape == (1, ):
        angular_speed_exp = csdl.expand(rpm, shape) / 60 * 2 * np.pi
    else:
        angular_speed_exp = csdl.expand(rpm, shape, action="i->ijk") / 60 * 2 * np.pi

    # define azimuthal discretization and normalized radius
    theta_linspace = np.linspace(
        0.0, 2 * np.pi - 2 * np.pi / num_azimuthal, num_azimuthal
    )
    theta_exp = csdl.Variable(
        value=np.einsum(
            "ij,k->ijk",
            np.ones((num_nodes, num_radial)),
            theta_linspace,
        )
    )

    # expand chord and twist profile
    chord_profile_exp = csdl.expand(chord_profile, out_shape=shape, action="j->ijk")
    twist_profile_exp = csdl.expand(twist_profile, out_shape=shape, action="j->ijk")

    # compute sectional blade solidity
    sigma = num_blades * chord_profile_exp / 2.0 / np.pi / radius_vec_exp

    pre_process_outputs = PreProcessOutputs(
        thrust_vector_exp=thrust_vector_exp,
        thrust_origin_vel_exp=thrust_origin_vel_exp,
        hub_radius=r_hub,
        element_width=dr,
        radius_vector_exp=radius_vec_exp,
        norm_radius_exp=norm_radius_exp,
        angular_speed_exp=angular_speed_exp,
        azimuth_angle_exp=theta_exp,
        chord_profile_exp=chord_profile_exp,
        twist_profile_exp=twist_profile_exp,
        sigma=sigma,
    )

    return pre_process_outputs

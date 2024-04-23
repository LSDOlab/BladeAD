import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass


@dataclass
class LocalFrameVelocities:
    local_frame_velocity: csdl.Variable
    tangential_velocity: csdl.Variable
    mu: csdl.Variable
    mu_z: csdl.Variable


def compute_local_frame_velocities(
    shape: tuple,
    thrust_vector: csdl.Variable,
    origin_velocity: csdl.Variable,
    angular_speed: csdl.Variable,
    azimuth_angle: csdl.Variable,
    radius_vec: csdl.Variable,
    radius: csdl.Variable,
) -> LocalFrameVelocities:
    
    # extract shape
    num_nodes = shape[0]
    num_radial = shape[1]
    num_azimuthal = shape[2]

    # set up (rotor) in-plane projection vector
    projection_vector = csdl.Variable(shape=(3,), value=np.array([0., 1., 0.]))

    # compute the rotor frame velocities (in-plane and axial)
    local_frame_velocity = csdl.Variable(shape=shape + (3,), value=np.zeros((shape + (3,))))

    mu_z = csdl.Variable(shape=(num_nodes, ), value=np.zeros((num_nodes, )))
    mu = csdl.Variable(shape=(num_nodes, ), value=np.zeros((num_nodes, )))

    for i in csdl.frange(num_nodes):
        normal_vec = thrust_vector[i, :]

        # project normal (i.e., thrust) vector along [0, 1, 0]
        thrust_vec_proj = (
            projection_vector - csdl.vdot(projection_vector, normal_vec) * normal_vec
        )

        # normalize projection
        in_plane_ey = thrust_vec_proj / csdl.norm(thrust_vec_proj)
        

        # compute second in-plane unit vector
        in_plane_ex = csdl.cross(normal_vec, in_plane_ey)
        

        # project origin velocity along in-plane vectors
        in_plane_ux = csdl.vdot(origin_velocity[i, :], in_plane_ex)
        in_plane_uy = csdl.vdot(origin_velocity[i, :], in_plane_ey)
        normal_uz = csdl.vdot(origin_velocity[i, :], normal_vec)

        # expand the in-plane and normal inflow to the right shape
        # axial frame velocity
        local_frame_velocity = local_frame_velocity.set(
            csdl.slice[i, :, :, 0],
            csdl.expand(
               normal_uz, (num_radial, num_azimuthal), 
            )
        )

        # in-plane frame velocity x
        local_frame_velocity = local_frame_velocity.set(
            csdl.slice[i, :, :, 1],
            csdl.expand(
               in_plane_ux, (num_radial, num_azimuthal), 
            )
        )

        # in-plane frame velocity y
        local_frame_velocity = local_frame_velocity.set(
            csdl.slice[i, :, :, 2],
            csdl.expand(
               in_plane_uy, (num_radial, num_azimuthal), 
            )
        )

        # compute mu & mu_z
        mu_z = mu_z.set(
            csdl.slice[i],
            normal_uz / angular_speed[i, 0, 0] / radius
        )

        mu = mu.set(
            csdl.slice[i],
            ((in_plane_ux**2 + in_plane_uy**2)**0.5) / angular_speed[i, 0, 0] / radius
        )


    # print(local_frame_velocity[:, :, :, 2].value)
    # exit()

    V_tangential = local_frame_velocity[:, :, :, 1] * csdl.sin(azimuth_angle) - \
                   local_frame_velocity[:, :, :, 2] * csdl.cos(azimuth_angle) + \
                    radius_vec * angular_speed
    

    local_frame_velocity = LocalFrameVelocities(
        local_frame_velocity=local_frame_velocity,
        tangential_velocity=V_tangential,
        mu=mu, 
        mu_z=mu_z,
    )

    return local_frame_velocity


if __name__ == "__main__":
    pass


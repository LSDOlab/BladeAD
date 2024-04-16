import csdl_alpha as csdl
import numpy as np


def expand_variables_and_convert_caddee_velocities_into_rotor_frame(
    shape: tuple,
    thrust_vector: csdl.Variable,
    origin_velocity: csdl.Variable,
    radius: csdl.Variable,
    norm_hub_radius: float,
    chord_profile : csdl.Variable,
    twist_profile : csdl.Variable,

):
    # extract shape
    num_nodes = shape[0]
    num_radial = shape[1]
    num_azimuthal = shape[2]

    # expand thrust vector & origin velocity if necessary
    if thrust_vector.shape != (num_nodes, 3):
        thrust_vector = csdl.expand(thrust_vector, (num_nodes, 3), action='j->ij')

    if origin_velocity.shape != (num_nodes, 3):
        origin_velocity = csdl.expand(origin_velocity, (num_nodes, 3))

    # compute blade element width
    r_hub = norm_hub_radius * radius
    dr = (radius - 0.2 * radius) / (num_radial - 1)

    _dr = csdl.expand()

    # set up (rotor) in-plane projection vector
    projection_vector = csdl.Variable(shape=(3,), value=np.array([0., 1., 0.]))

    # compute the rotor in-plane velocity
    rotor_inflow = csdl.Variable(shape=shape + (3,), value=np.zeros((shape + (3,))))

    for i in csdl.frange(num_nodes):
        normal_vec = -1 * thrust_vector[i, :]

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
        rotor_inflow = rotor_inflow.set(
            csdl.slice[i, :, :, 0],
            csdl.expand(
               normal_uz, (num_radial, num_azimuthal), 
            )
        )

        rotor_inflow = rotor_inflow.set(
            csdl.slice[i, :, :, 1],
            csdl.expand(
               -in_plane_ux, (num_radial, num_azimuthal), 
            )
        )

        rotor_inflow = rotor_inflow.set(
            csdl.slice[i, :, :, 2],
            csdl.expand(
               in_plane_uy, (num_radial, num_azimuthal), 
            )
        )



    return rotor_inflow, r_hub, dr


if __name__ == "__main__":
    import time
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    num_nodes = 2

    shape = (num_nodes, 30, 1)
    radius = csdl.Variable(value=1.5)
    thrust_vector = csdl.Variable(value=np.array([1/2**0.5, 0., -1/2**0.5]))
    origin_vel_np = np.zeros((num_nodes, 3))
    origin_vel_np[:, 0] = 40
    origin_vel_np[:, 1] = -10
    origin_vel_csdl = csdl.Variable(value=origin_vel_np)
    norm_hub_radius = 0.2

    t1 = time.time()
    rotor_inflow, r_hub, dr = convert_caddee_velocities_into_rotor_frame(
        shape=shape, 
        thrust_vector=thrust_vector, 
        origin_velocity=origin_vel_csdl,
        norm_hub_radius=norm_hub_radius,
        radius=radius,
    )

    x_dir_np = np.zeros((num_nodes, 3))
    x_dir_np[:, 0] = 1
    x_dir = csdl.Variable(shape=(num_nodes, 3), value=x_dir_np)

    x_dir_exp = csdl.expand(x_dir, shape + (3,), 'il->ijkl')

    _inflow_x = csdl.einsum(rotor_inflow, x_dir_exp, action='ijkl,ijkl->ijk')

    # csdl.einsum(rotor_inflow,)

    t2 = time.time()

    recorder.stop()

    print(rotor_inflow.value)
    print(_inflow_x.value.shape)
    # print(_inflow_x.shape)
    # print(rotor_inflow.shape)

    # print(dr.value)
    # print(r_hub.value)

    print("time" ,t2 - t1)



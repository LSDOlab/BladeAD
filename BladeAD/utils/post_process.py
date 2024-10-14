import csdl_alpha as csdl
import numpy as np


def compute_forces_and_moments(
        thrust, 
        torque,
        rotation_direction,
        thrust_vector, 
        thrust_origin,
        num_nodes,
        ac_states=None,
        ref_point=np.array([0., 0., 0.]),
        mirror_forces_and_moments=False,
):
    if mirror_forces_and_moments:
        forces = csdl.Variable(shape=(2 * num_nodes, 3), value=0.)
        moments = csdl.Variable(shape=(2* num_nodes, 3), value=0.)
    else:
        forces = csdl.Variable(shape=(num_nodes, 3), value=0.)
        moments = csdl.Variable(shape=(num_nodes, 3), value=0.)

    # transform forces into body-fixed frame
    if ac_states is not None:
        phi = ac_states.phi        
        theta = ac_states.theta
        psi = ac_states.psi

        if phi.shape != (num_nodes, ):
                phi = csdl.expand(phi, (num_nodes, ))

        if theta.shape != (num_nodes, ):
            theta = csdl.expand(theta, (num_nodes, ))

        if psi.shape != (num_nodes, ):
            psi = csdl.expand(psi, (num_nodes, ))
        
        # Rotate forces into body-fixed reference frame
        for i in range(num_nodes):
            L2B_mat = csdl.Variable(shape=(3, 3), value=0.)
            L2B_mat = L2B_mat.set(
                slices=csdl.slice[0, 0],
                value=csdl.cos(theta[i]) * csdl.cos(psi[i]),
            )

            L2B_mat = L2B_mat.set(
                slices=csdl.slice[0, 1],
                value=csdl.cos(theta[i]) * csdl.sin(psi[i]),
            )

            L2B_mat = L2B_mat.set(
                slices=csdl.slice[0, 2],
                value= -csdl.sin(theta[i]),
            )

            L2B_mat = L2B_mat.set(
                slices=csdl.slice[1, 0],
                value=csdl.sin(phi[i]) * csdl.sin(theta[i]) * csdl.cos(psi[i]) - csdl.cos(phi[i]) * csdl.sin(psi[i]),
            )

            L2B_mat = L2B_mat.set(
                slices=csdl.slice[1, 1],
                value=csdl.sin(phi[i]) * csdl.sin(theta[i]) * csdl.sin(psi[i]) + csdl.cos(phi[i]) * csdl.cos(psi[i]),
            )

            L2B_mat = L2B_mat.set(
                slices=csdl.slice[1, 2],
                value=csdl.sin(phi[i]) * csdl.cos(theta[i]),
            )

            L2B_mat = L2B_mat.set(
                slices=csdl.slice[2, 0],
                value=csdl.cos(phi[i]) * csdl.sin(theta[i]) * csdl.cos(psi[i]) + csdl.sin(phi[i]) * csdl.sin(psi[i]),
            )

            L2B_mat = L2B_mat.set(
                slices=csdl.slice[2, 1],
                value=csdl.cos(phi[i]) * csdl.sin(theta[i]) * csdl.sin(psi[i]) - csdl.sin(phi[i]) * csdl.cos(psi[i]),
            )

            L2B_mat = L2B_mat.set(
                slices=csdl.slice[2, 2],
                value=csdl.cos(phi[i]) * csdl.cos(theta[i]),
            )
            

            forces = forces.set(
                csdl.slice[i, :],
                csdl.matvec(
                    L2B_mat,
                    thrust[i] * thrust_vector[i, :],
                )
            )

            # print(num_nodes, i, rotation_direction)

            if rotation_direction is not None:
                rot_dir = rotation_direction[i]
                if rot_dir == "cw":
                    torque_moment = -thrust_vector[i, :] * torque[i]
                else:
                    torque_moment = thrust_vector[i, :] * torque[i]
            else:
                torque_moment = 0

            moments = moments.set(
                csdl.slice[i, :],
                csdl.cross(
                    csdl.matvec(
                        L2B_mat,
                        thrust_origin[i, :] - ref_point,
                    ),
                    forces[i, :]) + torque_moment
            )

            if mirror_forces_and_moments:
                thrust_vector_mirrored = thrust_vector[i, :] * np.array([1, -1, 1])
                thrust_origin_mirrored = thrust_origin[i, :] * np.array([1, -1, 1])
                forces = forces.set(
                    csdl.slice[num_nodes + i, :],
                    csdl.matvec(
                        L2B_mat,
                        thrust[i] * thrust_vector_mirrored,
                    )
                )

                if rotation_direction is not None:
                    rot_dir = rotation_direction[i]
                    if rot_dir == "cw":
                        torque_moment = thrust_vector_mirrored * torque[i]
                    else:
                        torque_moment = -thrust_vector_mirrored * torque[i]
                else:
                    torque_moment = 0

                moments = moments.set(
                    csdl.slice[num_nodes + i, :],
                    csdl.cross(
                        csdl.matvec(
                            L2B_mat,
                            thrust_origin_mirrored - ref_point,
                        ),
                        forces[num_nodes + i, :]) + torque_moment
                )

                
    else:
        for i in range(num_nodes):
            forces = forces.set(
                csdl.slice[i, :], thrust[i] * thrust_vector[i, :],
            )
            # print(num_nodes, i, rotation_direction)
            if rotation_direction is not None:
                rot_dir = rotation_direction[i]
                if rot_dir == "cw":
                    torque_moment = -thrust_vector[i, :] * torque[i]
                else:
                    torque_moment = thrust_vector[i, :] * torque[i]
            else:
                torque_moment = 0

            moments = moments.set(
                csdl.slice[i, :],
                csdl.cross(
                    thrust_origin[i, :] - ref_point,
                    forces[i, :]) + torque_moment
            )
            
            if mirror_forces_and_moments:
                thrust_vector_mirrored = thrust_vector[i, :] * np.array([1, -1, 1])
                thrust_origin_mirrored = thrust_origin[i, :] * np.array([1, -1, 1])

                forces = forces.set(
                    csdl.slice[num_nodes + i, :], thrust[i] * thrust_vector_mirrored,
                )

                if rotation_direction is not None:
                    rot_dir = rotation_direction[i]
                    if rot_dir == "cw":
                        torque_moment = thrust_vector_mirrored * torque[i]
                    else:
                        torque_moment = -thrust_vector_mirrored * torque[i]
                else:
                    torque_moment = 0

                moments = moments.set(
                    csdl.slice[num_nodes + i, :],
                    csdl.cross(
                        thrust_origin_mirrored - ref_point,
                        forces[num_nodes + i, :]) + torque_moment
                )
            
    return forces, moments

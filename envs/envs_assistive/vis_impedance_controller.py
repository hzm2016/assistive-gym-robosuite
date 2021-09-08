# System imports

# Package imports
import mujoco_py
from mujoco_py.generated import const
import numpy as np
np.set_printoptions(precision=5)

# Local imports
from envs.gym_kuka_mujoco.controllers import iMOGVIC
from envs.gym_kuka_mujoco.utils.transform_utils import *

import transforms3d as transforms3d
import matplotlib.pyplot as plt
import seaborn as sns
import commentjson
import os
from envs.gym_kuka_mujoco import kuka_asset_dir

# sns.set_theme(font_scale=1.5)


def render_frame(viewer, pos, euler):
    viewer.add_marker(pos=pos,
                      label='',
                      type=const.GEOM_SPHERE,
                      size=[.01, .01, .01])
    
    # mat = quat2mat(quat)
    mat = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2], 'sxyz')
    cylinder_half_height = 0.02
    pos_cylinder = pos + mat.dot([0.0, 0.0, cylinder_half_height])
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[.005, .005, cylinder_half_height],
                      mat=mat)


def render_point(viewer, pos):
    viewer.add_marker(pos=pos,
                      label='',
                      type=const.GEOM_SPHERE,
                      size=[.01, .01, .01])


def vis_impedance_random_sawyer_setpoint(collision=False):
    options = dict()
    num_waypoints = 3

    param_dir = '/home/zhimin/code/5_thu/rl-robotic-assembly-control/code/pytorch/LAMPO/params/'
    param_file = 'IMOGICAssitiveJaco.json'
    
    param_file = os.path.join(param_dir, param_file)
    with open(param_file) as f:
        params = commentjson.load(f)
    
    # options['model_path'] = 'a_sawyer_test.xml'
    #
    # options['rot_scale'] = .3
    # options['stiffness'] = np.array([1., 1., 1., 3., 3., 3.])
    #
    # options['controlled_joints'] = ["robot0_right_j0", "robot0_right_j1",
    #                                 "robot0_right_j2", "robot0_right_j3",
    #                                 "robot0_right_j4", "robot0_right_j5",
    #                                 "robot0_right_j6"]
    #
    # options['num_waypoints'] = 3
    # options['null_space_damping'] = 1.0
    
    model_path = os.path.join(params["controller_options"]['model_root'], params["controller_options"]["model_path"])
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    
    controller = iMOGVIC(sim, **params["controller_options"])
    
    frame_skip = 50
    high = np.array([.1, .1, .1, 2, 2, 2])
    low = -np.array([.1, .1, .1, 2, 2, 2])
    
    viewer = mujoco_py.MjViewer(sim)

    # set parameters :::
    scale = np.array([8.0, 0.0, 0.0])
    scale_list = scale.repeat([6, 6, 6], axis=0).reshape(num_waypoints, 6)
    # print("scale_list :::", scale_list)

    stiffness_list = np.array([[20., 4., 4., 4., 4., 4.],
                               [0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0.]])

    control_scale = np.ones_like(stiffness_list) * 100
    stiffness_list = control_scale * stiffness_list
    damping_list = scale_list * np.sqrt(stiffness_list)
    print("stiffness_list :::", stiffness_list)
    print("damping_list :::", damping_list)
    
    weight_list = np.array([1.0, 0.05, 0.01])
    controller.set_params_direct(stiffness_list, damping_list, weight_list)

    qpos = np.array([0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 1.6482, 1.9628])
    # qpos = np.random.uniform(-1., 1., size=7)
    
    # controller.update_initial_joints(qpos)
    
    qvel = np.zeros(7)

    sim_state = sim.get_state()
    sim_state.qpos[:7] = qpos
    sim_state.qvel[:7] = qvel
    sim.set_state(sim_state)
    sim.forward()

    controller.update_state()
    print("current ee_pose :::", controller.ee_pose)
    
    target_pos, target_mat = controller.get_pose_site("ee_site")
    target_pos, target_mat = controller.ee_pose[:3] + np.array([0.0, 0.0, 0.2]), target_mat
    print("target ee_pose :::", target_pos, target_mat)
    
    target_euler = transforms3d.euler.mat2euler(target_mat, 'sxyz')
    
    # while True:
    #     viewer.render()

    # set way_points :::
    initial_state = np.array([-0.60349109, 0.09318907, 0.27348721,
                              1.9265446606661554, -0.40240192959667226, -1.541555812071902])
    optimal_state = np.array([-0.80349109, 0.09318907, 0.07348721,
                              1.9265446606661554, -0.40240192959667226, -1.541555812071902])
    state_scale = initial_state - optimal_state

    # pos_set_list = np.array([[0.1,  0.,  1.2],
    #                          [0.1,  0.,  1.2],
    #                          [0.1,  0.,  1.2]])
    # quat_set_list = np.array([[-3.128104, 0.00437383, -2.08817412],
    #                           [-3.128104, 0.00437383, -2.08817412],
    #                           [-3.128104, 0.00437383, -2.08817412]])
    
    pos_set_list = np.zeros((3, 3))
    quat_set_list = np.zeros((3, 3))
    for i in range(num_waypoints):
        pos_set_list[i, :] = target_pos
        quat_set_list[i, :] = target_euler

    way_points_list = np.concatenate((pos_set_list, quat_set_list), axis=1)
    print("way_point_list :::", way_points_list)
    controller.set_way_points(way_points_list)
    print("reference list :::", controller.reference_list)

    optimal_pos = pos_set_list[0, :]
    velocity_list = []
    position_list = []
    stiffness_matrix = []
    damping_matrix = []
    energy_list = []

    for i in range(1):
        qpos = np.random.uniform(-1., 1., size=7)
        # qpos = np.array([0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 1.6482, 1.9628])
        # qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        # qpos = np.array([-0.5538, -0.8208, 0.4155, 0.8409, -0.4955, 0.6482, 1.9628])
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])
    
        sim_state = sim.get_state()
        sim_state.qpos[:7] = qpos
        sim_state.qvel[:7] = qvel
        sim.set_state(sim_state)
        sim.forward()

        controller.update_state()
        print("current ee_pose :::", controller.ee_pose)
        for j in range(3500):
            controller.update_state()
            torque, V, pose_err, vel_err, stiffness_eqv, damping_eqv = controller.update_vic_torque()
            energy_list.append(V)
            position_list.append(pose_err)
            velocity_list.append(vel_err)
            stiffness_matrix.append(stiffness_eqv)
            damping_matrix.append(damping_eqv)
            # torque = controller.get_euler_torque(way_points_list)
            # torque = controller.update_torque(way_points_list)
            print("final state", np.linalg.norm(controller.state, ord=2))
            sim.data.ctrl[:] = torque[:7]
            sim.step()
            render_frame(viewer, pos_set_list[0, :], quat_set_list[0, :])
            viewer.render()
            
    return position_list, velocity_list, energy_list, optimal_pos, stiffness_matrix, damping_matrix

    
def plot_3d_trajectory(position_list, velocity_list):
    LINEWIDTH = 4.0
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(projection='3d')
    
    plt.title('Position')
    ax.plot(position_list[:, 0],
            position_list[:, 1],
            position_list[:, 2], linewidth=LINEWIDTH)
     
    # ax_1.plot(position_list[:, 0], position_list[:, 2], linewidth=LINEWIDTH)
    # # plt.xticks([0.0, 0.1, 0.2])
    # plt.yticks([0.0, 0.1, 0.2])
    # ax_1.set_xticks([0.0, 0.1, 0.2], minor=True)
    
    # scale = 15
    # for i in range(len(index_list)):
    #     print("stiffness matrix :::", stiffness_matrix[index_list[i]])
    #     el = Ellipse((position_list[index_list[i], 0], position_list[index_list[i], 2]),
    #                  stiffness_matrix[index_list[i]][0] / np.linalg.norm(
    #                      (stiffness_matrix[index_list[i]][0], stiffness_matrix[index_list[i]][2]), ord=2) / scale,
    #                  stiffness_matrix[index_list[i]][2] / np.linalg.norm(
    #                      (stiffness_matrix[index_list[i]][0], stiffness_matrix[index_list[i]][2]), ord=2) / scale,
    #                  alpha=0.3, facecolor='green', edgecolor='black', linewidth=2.0)
    #     ax_1.add_patch(el)
    #     ax_1.scatter(position_list[index_list[i], 0], position_list[index_list[i], 2], s=200, marker='^', c='red')
    #     ax_1.text(position_list[index_list[i], 0] - 0.02, position_list[index_list[i], 2], index_list[i], fontsize=15)
    
    # plt.scatter(position_list[0, 0] - optimal_pos[0],
    # 			position_list[0, 1] - optimal_pos[1],
    # 			position_list[0, 2] - optimal_pos[2],
    # 			s=100, marker='^', c='g')
    # plt.scatter(position_list[-1, 0] - optimal_pos[0],
    # 			position_list[-1, 1] - optimal_pos[1],
    # 			position_list[-1, 2] - optimal_pos[2],
    # 			s=100, marker='^', c='r')
    # plt.xticks([-0.2, -0.1, 0.0])
    # plt.yticks([-0.2, -0.1, 0.0])

    ax.set_xlabel('$X$(m)')  # fontsize=20, rotation=150
    ax.set_ylabel('$Y$(m)')
    ax.set_zlabel('$Z$(m)')  # fontsize=30, rotation=60
    # ax.yaxis._axinfo['label']['space_factor'] = 3.0
    plt.show()
    
    # ax_2 = plt.subplot(1, 4, 2)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.gca(projection='3d')
    
    plt.title('Velocity')
    ax.plot(velocity_list[:, 0],
            velocity_list[:, 1],
            velocity_list[:, 2], linewidth=3.5)
    # ax_2.plot(velocity_list[:, 0], velocity_list[:, 2], linewidth=LINEWIDTH)
    
    # scale = 5
    # for i in range(len(index_list)):
    #     print("damping_matrix :::", damping_matrix[index_list[i]])
    #     el_2 = Ellipse((velocity_list[index_list[i], 0], velocity_list[index_list[i], 2]),
    #                    damping_matrix[index_list[i]][0] / np.linalg.norm(
    #                        (damping_matrix[index_list[i]][0], damping_matrix[index_list[i]][2]), ord=2) / scale,
    #                    damping_matrix[index_list[i]][2] / np.linalg.norm(
    #                        (damping_matrix[index_list[i]][0], damping_matrix[index_list[i]][2]), ord=2) / scale,
    #                    alpha=0.3, facecolor='green', edgecolor='black', linewidth=2.0)
    #     ax_2.add_patch(el_2)
    #
    #     ax_2.scatter(velocity_list[index_list[i], 0], velocity_list[index_list[i], 2], s=200, marker='^', c='red')
    
    ax.set_xlabel('$X$(m/s)')  # fontsize=20, rotation=150
    ax.set_ylabel('$Y$(m/s)')
    ax.set_zlabel('$Z$(m/s)')  # fontsize=30, rotation=60
    plt.show()
    

def plot_energy(energy_list, index_list=None):
    LINEWIDTH = 4
    plt.figure(figsize=(10, 10), dpi=500)
    plt.title('Energy')

    plt.plot(np.array(energy_list), linewidth=LINEWIDTH)
    
    for i in range(len(index_list)):
        plt.scatter(index_list[i], energy_list[index_list[i]], s=200, marker='^', c='red')
    
    plt.xlabel('steps')
    plt.ylabel('r$V(s,\dot{s})$')
    plt.show()


def plot_results(position_list, velocity_list, energy_list, stiffness_matrix, damping_matrix):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Ellipse
    
    FONT_SIZE = 24
    LINEWIDTH = 4
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['font.weight'] = 'bold'
    
    position_list = np.array(position_list)
    velocity_list = np.array(velocity_list)
    energy_list = np.array(energy_list)
    stiffness_matrix = stiffness_matrix
    damping_matrix = np.array(damping_matrix)
    
    index_list = [3, 100, 600, 950]
    plt.figure(figsize=(24, 6))
    
    ax_1 = plt.subplot(1, 4, 1)
    # fig = plt.figure(figsize=(10, 8), dpi=300)
    # ax = fig.add_subplot(projection='3d')
    # ax = fig.gca(projection='3d')
    plt.title('Position')
    # ax.plot(position_list[:, 0],
    #         position_list[:, 1],
    #         position_list[:, 2], linewidth=3.5)
    ax_1.plot(position_list[:, 0], position_list[:, 2], linewidth=LINEWIDTH)
    # plt.xticks([0.0, 0.1, 0.2])
    plt.yticks([0.0, 0.1, 0.2])
    ax_1.set_xticks([0.0, 0.1, 0.2], minor=True)

    scale = 15
    for i in range(len(index_list)):
        print("stiffness matrix :::", stiffness_matrix[index_list[i]])
        el = Ellipse((position_list[index_list[i], 0], position_list[index_list[i], 2]),
                      stiffness_matrix[index_list[i]][0]/np.linalg.norm((stiffness_matrix[index_list[i]][0], stiffness_matrix[index_list[i]][2]), ord=2)/scale,
                      stiffness_matrix[index_list[i]][2]/np.linalg.norm((stiffness_matrix[index_list[i]][0], stiffness_matrix[index_list[i]][2]), ord=2)/scale,
                      alpha=0.3, facecolor='green', edgecolor='black', linewidth=2.0)
        ax_1.add_patch(el)
        ax_1.scatter(position_list[index_list[i], 0], position_list[index_list[i], 2], s=200, marker='^', c='red')
        ax_1.text(position_list[index_list[i], 0] - 0.02, position_list[index_list[i], 2], index_list[i], fontsize=15)
    
    # plt.scatter(position_list[0, 0] - optimal_pos[0],
    # 			position_list[0, 1] - optimal_pos[1],
    # 			position_list[0, 2] - optimal_pos[2],
    # 			s=100, marker='^', c='g')
    # plt.scatter(position_list[-1, 0] - optimal_pos[0],
    # 			position_list[-1, 1] - optimal_pos[1],
    # 			position_list[-1, 2] - optimal_pos[2],
    # 			s=100, marker='^', c='r')
    # plt.xticks([-0.2, -0.1, 0.0])
    # plt.yticks([-0.2, -0.1, 0.0])
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    # plt.zlabel('Z(m)')
    # plt.show()
    
    ax_2 =plt.subplot(1, 4, 2)
    # fig = plt.figure(figsize=(10, 8), dpi=300)
    # ax = fig.gca(projection='3d')
    plt.title('Velocity')
    # ax.plot(velocity_list[:, 0],
    #         velocity_list[:, 1],
    #         velocity_list[:, 2], linewidth=3.5)
    ax_2.plot(velocity_list[:, 0], velocity_list[:, 2], linewidth=LINEWIDTH)

    scale = 5
    for i in range(len(index_list)):
        print("damping_matrix :::", damping_matrix[index_list[i]])
        el_2 = Ellipse((velocity_list[index_list[i], 0], velocity_list[index_list[i], 2]),
                     damping_matrix[index_list[i]][0] / np.linalg.norm(
                         (damping_matrix[index_list[i]][0], damping_matrix[index_list[i]][2]), ord=2) / scale,
                     damping_matrix[index_list[i]][2] / np.linalg.norm(
                         (damping_matrix[index_list[i]][0], damping_matrix[index_list[i]][2]), ord=2) / scale,
                     alpha=0.3, facecolor='green', edgecolor='black', linewidth=2.0)
        ax_2.add_patch(el_2)
    
        ax_2.scatter(velocity_list[index_list[i], 0], velocity_list[index_list[i], 2], s=200, marker='^', c='red')
        
    plt.xlabel('X(m/s)')
    plt.ylabel('Y(m/s)')
    # plt.zlabel('Z(m)')
    # plt.show()

    ax_3 = plt.subplot(1, 4, 3)
    # plt.figure(figsize=(10, 8), dpi=300)
    plt.title('Energy')
    # print("energy_list :::", energy_list)
    # velocity_list = np.array(velocity_list)
    ax_3.plot(energy_list, linewidth=LINEWIDTH)

    for i in range(len(index_list)):
        ax_3.scatter(index_list[i], energy_list[index_list[i]], s=200, marker='^', c='red')
    
    plt.xlabel('steps')
    plt.ylabel('r$V(\cdot,\cdot)$')

    ax_4 = plt.subplot(1, 4, 4)
    # plt.figure(figsize=(10, 8), dpi=300)
    plt.title('ee pose')
    # print("energy_list :::", energy_list)
    # velocity_list = np.array(velocity_list)
    ax_4.plot(position_list[:, 0], linewidth=LINEWIDTH, label='x')
    ax_4.plot(position_list[:, 1], linewidth=LINEWIDTH, label='y')
    ax_4.plot(position_list[:, 2], linewidth=LINEWIDTH, label='z')
    ax_4.plot(position_list[:, 3], linewidth=LINEWIDTH, label=r'$\alpha$')
    ax_4.plot(position_list[:, 4], linewidth=LINEWIDTH, label=r'$\beta$')
    ax_4.plot(position_list[:, 5], linewidth=LINEWIDTH, label=r'$\gamma$')
    ax_4.legend()

    plt.xlabel('steps')
    plt.ylabel('$V(\cdot,\cdot)$')
    
    plt.show()


if __name__ == '__main__':
    position_list, velocity_list, energy_list, optimal_pos, stiffness_matrix, damping_matrix \
        = vis_impedance_random_sawyer_setpoint(collision=True)

    # plot_3d_trajectory(np.array(position_list), np.array(velocity_list))
    # plot_results(mode='3d')
    # vis_impedance_random_sawyer_setpoint(collision=False)

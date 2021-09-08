# System imports

# Package imports
import mujoco_py
from mujoco_py.generated import const
import numpy as np
import time
np.set_printoptions(precision=5)

# Local imports
from envs.gym_kuka_mujoco.controllers import ImpedanceControllerV2
from envs.gym_kuka_mujoco.controllers import iMOGVIC

from envs.gym_kuka_mujoco.utils.kinematics import forwardKinSite
from envs.gym_kuka_mujoco.utils.quaternion import mat2Quat, quat2Mat
from envs.gym_kuka_mujoco.controllers.test.common import create_sim
from envs.gym_kuka_mujoco.utils.transform_utils import *

import transforms3d as transforms3d
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# def render_frame(viewer, pos, quat, euler):
#     viewer.add_marker(pos=pos,
#                       label='',
#                       type=const.GEOM_SPHERE,
#                       size=[.01, .01, .01])
#
#     # mat = quat2mat(quat)
#     mat = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2], 'sxyz')
#     cylinder_half_height = 0.02
#     pos_cylinder = pos + mat.dot([0.0, 0.0, cylinder_half_height])
#     viewer.add_marker(pos=pos_cylinder,
#                       label='',
#                       type=const.GEOM_CYLINDER,
#                       size=[.005, .005, cylinder_half_height],
#                       mat=mat)


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
    
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[cylinder_half_height, .005, .005],
                      mat=mat)
    
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[.005, cylinder_half_height, .005],
                      mat=mat)


def render_point(viewer, pos):
    viewer.add_marker(pos=pos,
                      label='',
                      type=const.GEOM_SPHERE,
                      size=[.01, .01, .01])


def vis_impedance_fixed_setpoint(collision=False):
    options = dict()
    options['model_path'] = 'a_sawyer_test.xml'
    options['rot_scale'] = .3
    options['controlled_joints'] = ["robot0_right_j0", "robot0_right_j1",
                                    "robot0_right_j2", "robot0_right_j3",
                                    "robot0_right_j4", "robot0_right_j5",
                                    "robot0_right_j6"]
    
    options['stiffness'] = np.array([1000., 1000., 1000., 1000., 1000., 1000.])
    import os
    from envs.gym_kuka_mujoco import kuka_asset_dir

    model_path = os.path.join(kuka_asset_dir(), 'a_sawyer_test.xml')
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    # controller = ImpedanceControllerV2(sim, **options)
    
    controller = iMOGVIC(sim, **options)
    
    # sim = create_sim(collision=collision)
    # controller = ImpedanceControllerV2(sim, **options)

    viewer = mujoco_py.MjViewer(sim)
    for i in range(10):
        
        # Set a random state to get a random feasible setpoint.
        # qpos = np.random.uniform(-1., 1, size=7)
        qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])
        
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()
        
        controller.set_action(np.array([0., 0., -0.1, 0.0, 0.0, 0.]))
        print("pos_set", controller.pos_set)
        print("quat_set", controller.quat_set)
        pos_set = np.array([0.08143538, -0.3, 1.2090885])
        # quat_set = np.array([0.00200338, 0.02710148, 0.99960754, - 0.00680102])
        # euler = mat2euler(quat2mat(np.array([0.00200338, 0.02710148, 0.99960754, - 0.00680102])))
        
        euler = np.array([0.0, 1.0, 1.57])
        
        # print("euler :::", euler)
        quat_set = mat2quat(euler2mat(euler))
        print("quat_set_new", euler2mat(euler))
        controller.set_target(pos_set, euler)
        
        # Set a different random state and run the controller.
        # qpos = np.random.uniform(-1., 1., size=7)
        qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        controller.update_initial_joints(qpos)
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])
        
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()
        
        # get initial pose :::
        controller.update_state()
        print('ee_pos :::', controller.ee_pos)
        print('ee_ori_mat :::', controller.ee_ori_mat)
        
        # set way_points :::
        pos_set_list = np.array([[0.08143538, -0.3, 1.2090885],
                                 [0.08143538, -0.0, 0.2090885]])
        quat_set_list = np.array([[0.0, 1.57, 1.0],
                                  [0.0, 1.0, 1.0]])
        way_points_list = np.concatenate((pos_set_list, quat_set_list), axis=1)
        controller.set_waypoints(way_points_list)
        # print("way_points_list", way_points_list)
        print("optimal pose:::", controller.optimal_pose)
        print("reference_list", controller.reference_list)
        
        # set parameters :::
        scale = 2.0
        stiffness_list = np.array([[1000., 1000., 1000., 1000., 1000., 1000.],
                                   [1000., 1000., 1000., 1000., 1000., 1000.]])
        
        damping_list = scale * np.sqrt(stiffness_list)
        weight_list = np.array([1.0, 0.9])
        controller.set_params_direct(stiffness_list, damping_list, weight_list)
        controller.update_state()
        
        # print('ee_ori :::', mat2euler(controller.ee_ori_mat))
        # state = np.concatenate((controller.ee_pos, mat2euler(controller.ee_ori_mat)))
        
        # omega_weights, _ = controller.get_non_linear_weight(state, stiffness_list, weight_list)
        # print("omega_weights ::::", omega_weights)
        
        # while True:
        #     controller.update_calculation()
        #     print('ee_pos :::', controller.ee_pos)
        #     print('ee_ori_mat :::', controller.ee_ori_mat)
        #     viewer.render()
        
        # for i in range(3000):
        #     # torque = controller.get_torque()
        #     controller.update_state()
        #     torque = controller.update_torque()
        #     print("torque ::::", torque)
        #     sim.data.ctrl[:] = torque[:7]
        #     sim.step()
        #
        #     render_frame(viewer, controller.pos_set, np.array(controller.quat_set).astype(np.float64), )
        #     viewer.render()
        
        optimal_quat = transforms3d.euler.euler2quat(quat_set_list[0, 0], quat_set_list[0, 1], quat_set_list[0, 2], 'sxyz')
        for i in range(3000):
            # torque = controller.get_torque()
            controller.update_state()
            # torque = controller.update_vic_torque()
            torque = controller.get_euler_torque(way_points_list)
            # print("torque ::::", torque)
            sim.data.ctrl[:] = torque[:7]
            sim.step()

            render_frame(viewer, controller.pos_set, np.array(optimal_quat).astype(np.float64), np.array(quat_set_list[0, :]))
            viewer.render()


def vis_impedance_random_setpoint(collision=False):
    options = dict()
    options['model_path'] = 'full_kuka_no_collision.xml'
    options['rot_scale'] = 1.0
    options['pos_scale'] = 1.0
    options['stiffness'] = np.array([3., 3., 3., 3., 3., 3.])
    options['controlled_joints'] = ["kuka_joint_1", "kuka_joint_2",
                                    "kuka_joint_3", "kuka_joint_4",
                                    "kuka_joint_5", "kuka_joint_6",
                                    "kuka_joint_7"]

    sim = create_sim(collision=collision)
    controller = iMOGVIC(sim, **options)

    frame_skip = 50
    high = np.array([.1, .1, .1, 2, 2, 2])
    low = -np.array([.1, .1, .1, 2, 2, 2])

    viewer = mujoco_py.MjViewer(sim)
    for i in range(10):

        # Set a different random state and run the controller.
        # qpos = np.random.uniform(-1., 1., size=7)
        qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])
        
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()
        
        controller.update_state()
        controller.set_target(controller.ee_pos, transforms3d.quaternions.mat2quat(controller.ee_ori_mat))

        qpos = np.random.uniform(-1., 1., size=7)
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()
        
        # controller.set_action()
        # controller.set_action(np.array([-0.2, 0.0, 0.2, 0.0, 1.57, 0.0]))
        for i in range(1000):
            # controller.set_action(np.random.uniform(high, low))
            for i in range(frame_skip):
                controller.update_state()
                sim.data.ctrl[:] = controller.get_torque_reset()
                sim.step()
                euler = transforms3d.euler.mat2euler(transforms3d.quaternions.quat2mat(controller.quat_set), 'sxyz')
                render_frame(viewer, controller.pos_set, euler)
                viewer.render()


def vis_impedance_sawyer_setpoint(collision=False):
    import os
    from envs.gym_kuka_mujoco.envs.assets import kuka_asset_dir
    
    num_waypoints = 3
    
    options = dict()
    
    # options['model_path'] = 'full_peg_insertion_experiment_no_gravity_moving_hole_id=030.xml'
    options['model_path'] = 'full_kuka_no_collision.xml'
    options['rot_scale'] = .3
    options['num_waypoints'] = num_waypoints
    options['null_space_damping'] = 1.0
    options['stiffness'] = np.array([3., 3., 3., 3., 3., 3.])
    options['controlled_joints'] = ["kuka_joint_1", "kuka_joint_2",
                                    "kuka_joint_3", "kuka_joint_4",
                                    "kuka_joint_5", "kuka_joint_6",
                                    "kuka_joint_7"]
    
    # sim = create_sim(collision=collision)
    model_path = os.path.join(kuka_asset_dir(), options['model_path'])
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)
    controller = iMOGVIC(sim, **options)
    
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
    print("damping_list :::", stiffness_list)
    damping_list = scale_list * np.sqrt(stiffness_list)
    
    print("damping_list :::", damping_list)
    weight_list = np.array([1.0, 0.05, 0.01])
    controller.set_params_direct(stiffness_list, damping_list, weight_list)

    # Set a different random state and run the controller.
    # qpos = np.random.uniform(-1., 1., size=7)
    qpos = np.array([0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
    controller.update_initial_joints(qpos)
    qvel = np.zeros(7)

    sim_state = sim.get_state()
    sim_state.qpos[:] = qpos
    sim_state.qvel[:] = qvel
    sim.set_state(sim_state)
    sim.forward()
 
    # controller.update_state()
    # print("current ee_pose :::", controller.ee_pose)
    #
    # random_hole_file = 'random_reachable_holes_small_randomness.npy'
    # reachable_holes = np.load(os.path.join(kuka_asset_dir(), random_hole_file), allow_pickle=True)
    # hole_data = np.random.choice(reachable_holes)
    # good_states = hole_data['good_poses']
    # print("good_states", good_states)
    # sim.data.set_mocap_pos('hole', hole_data['hole_pos'])
    # sim.data.set_mocap_quat('hole', hole_data['hole_quat'])
    # qpos = good_states[1]
    # sim_state = sim.get_state()
    # sim_state.qpos[:] = qpos
    # sim_state.qvel[:] = qvel
    # sim.set_state(sim_state)
    # sim.forward()
    #
    # while True:
    #     viewer.render()
    
    # set way_points :::
    initial_state = np.array([-0.60349109, 0.09318907, 0.27348721,
                              1.9265446606661554, -0.40240192959667226, -1.541555812071902])
    optimal_state = np.array([-0.80349109, 0.09318907, 0.07348721,
                              1.9265446606661554, -0.40240192959667226, -1.541555812071902])
    state_scale = initial_state - optimal_state

    pos_set_list = np.array([[-0.80349109, 0.09318907, 0.07348721],
                             [-0.78349109, 0.09318907, 0.07348721],
                             [-0.80349109, 0.09318907, -0.03348721]])
    quat_set_list = np.array([[1.9265446606661554, -0.40240192959667226, -1.541555812071902],
                              [1.9265446606661554, -0.40240192959667226, -1.541555812071902],
                              [1.9265446606661554, -0.40240192959667226, -1.541555812071902]])

    way_points_list = np.concatenate((pos_set_list, quat_set_list), axis=1)
    print("way_point_list :::", way_points_list)
    controller.set_way_points(way_points_list)
    print("reference list :::", controller.reference_list)

    optimal_pose = pos_set_list[0, :]
    velocity_list = []
    position_list = []
    stiffness_matrix = []
    damping_matrix = []
    energy_list = []

    for i in range(1):
        # qpos = np.random.uniform(-1., 1., size=7)
        qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        # qpos = np.array([-0.5538, -0.8208, 0.4155, 0.8409, -0.4955, 0.6482, 1.9628])
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])

        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()
        for j in range(1000):
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

    return position_list, velocity_list, energy_list, optimal_pose, stiffness_matrix, damping_matrix


def vis_impedance_random_sawyer_setpoint(collision=False):
    import os
    from envs.gym_kuka_mujoco import kuka_asset_dir
    
    options = dict()
    num_waypoints = 3
    
    options['model_root'] = "/home/zhimin/code/5_thu/rl-robotic-assembly-control/envs/envs_assistive/assets/sawyer/"
    options['model_path'] = "sawyer_new.xml"
    options['controlled_joints'] = ["right_j0", "right_j1",
                                    "right_j2", "right_j3",
                                    "right_j4", "right_j5",
                                    "right_j6"]
    
    # options['model_root'] = kuka_asset_dir()
    # options['model_path'] = 'a_sawyer_test.xml'
    # options['controlled_joints'] = ["robot0_right_j0", "robot0_right_j1",
    #                                 "robot0_right_j2", "robot0_right_j3",
    #                                 "robot0_right_j4", "robot0_right_j5",
    #                                 "robot0_right_j6"]
    
    options['rot_scale'] = .3
    options['stiffness'] = np.array([1., 1., 1., 3., 3., 3.])

    options['num_waypoints'] = 3
    options['null_space_damping'] = 1.0
    
    model_path = os.path.join(options['model_root'], options['model_path'])
    model = mujoco_py.load_model_from_path(model_path)
    sim = mujoco_py.MjSim(model)

    controller = iMOGVIC(sim, **options)
    # controller = ImpedanceControllerV2(sim, **options)
    
    frame_skip = 50
    high = np.array([.1, .1, .1, 2, 2, 2])
    low = -np.array([.1, .1, .1, 2, 2, 2])
    
    viewer = mujoco_py.MjViewer(sim)

    # set parameters :::
    scale = np.array([8.0, 0.0, 0.0])
    scale_list = scale.repeat([6, 6, 6], axis=0).reshape(num_waypoints, 6)
    # print("scale_list :::", scale_list)

    stiffness_list = np.array([[10., 4., 4., 4., 4., 4.],
                               [0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0.]])

    control_scale = np.ones_like(stiffness_list) * 100
    stiffness_list = control_scale * stiffness_list
    print("damping_list :::", stiffness_list)
    damping_list = scale_list * np.sqrt(stiffness_list)

    print("damping_list :::", damping_list)
    weight_list = np.array([1.0, 0.05, 0.01])
    controller.set_params_direct(stiffness_list, damping_list, weight_list)

    # Set a different random state and run the controller.
    # qpos = np.random.uniform(-1., 1., size=7)
    qpos = np.array([0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
    controller.update_initial_joints(qpos)
    qvel = np.zeros(7)

    sim_state = sim.get_state()
    sim_state.qpos[:7] = qpos
    sim_state.qvel[:7] = qvel
    sim.set_state(sim_state)
    sim.forward()
    
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
    
    # pos_set_list = np.array([[0.1,  0.,  1.2],
    #                          [0.1,  0.,  1.2],
    #                          [0.1,  0.,  1.2]])


    # optimal_pose = pos_set_list[0, :]
    velocity_list = []
    position_list = []
    stiffness_matrix = []
    damping_matrix = []
    energy_list = []
 
    for i in range(1):
        # qpos = np.random.uniform(-1., 1., size=7)
        # qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        qpos = np.array([-0.76486, -0.29371, -0.80882, 0.95799, -0.63627, -1.34402, -1.78878])
        qvel = np.zeros(7)
        state = np.concatenate([qpos, qvel])
    
        sim_state = sim.get_state()
        sim_state.qpos[:7] = qpos
        sim_state.qvel[:7] = qvel
        sim.set_state(sim_state)
        sim.forward()

        controller.update_state()
        print("initial ee_pose :::", controller.ee_pose)
        initial_ee_pose = controller.ee_pose.copy()

        # target_pos, target_mat = controller.get_pose_site("target_ee_site")
        # print("target ee_pose :::", target_pos)

        target_pos, target_euler = controller.ee_pose[:3] + np.array([0.20295, 0.27486, -0.1]), \
                                   controller.ee_pose[3:] + np.array([-0.0, -0.0, 0.3])
        original_target_pos, original_target_euler = controller.ee_pose[:3], controller.ee_pose[3:].copy()
        
        pos_set_list = np.tile(np.array(target_pos), (3, 1))
        quat_set_list = np.tile(np.array(target_euler), (3, 1))
        print("quat_set_list :", quat_set_list)

        way_points_list = np.concatenate((pos_set_list, quat_set_list), axis=1)
        print("way_point_list :::", way_points_list)
        controller.set_way_points(way_points_list)
        print("reference list :::", controller.reference_list)
        
        for j in range(1500):
            controller.update_state()
            torque, V, pose_err, vel_err, stiffness_eqv, damping_eqv = controller.update_vic_torque()
            energy_list.append(V)
            position_list.append(pose_err)
            velocity_list.append(vel_err)
            stiffness_matrix.append(stiffness_eqv)
            damping_matrix.append(damping_eqv)
            # torque = controller.get_euler_torque(way_points_list)
            # torque = controller.update_torque(way_points_list)

            sim.data.ctrl[:7] = torque[:7]
            sim.step()
            
            render_frame(viewer, pos_set_list[0, :], quat_set_list[0, :])
            
            viewer.render()
        
        # while True:
        #     # render_frame(viewer, pos_set_list[0, :], quat_set_list[0, :])
        #     render_frame(viewer, target_pos, original_target_euler)
        #     viewer.render()
        
        print("final state", np.linalg.norm(controller.state, ord=2))
        print("final pose :", controller.ee_pose)
        print("final error :", controller.ee_pose - initial_ee_pose)


def plot_results(mode='3d'):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Ellipse
    
    FONT_SIZE = 18
    LINEWIDTH = 4
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['font.weight'] = 'bold'
    
    position_list, velocity_list, energy_list, optimal_pos, stiffness_matrix, damping_matrix \
        = vis_impedance_sawyer_setpoint(collision=True)
    
    position_list = np.array(position_list)
    velocity_list = np.array(velocity_list)
    energy_list = np.array(energy_list)
    stiffness_matrix = stiffness_matrix
    damping_matrix = np.array(damping_matrix)
    
    index_list = [3, 100, 600, 950]
    plt.figure(figsize=(24, 6))
    if mode == '3d':
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
    # vis_impedance_fixed_setpoint()
    # vis_impedance_fixed_setpoint(collision=True)
    
    # vis_impedance_random_setpoint()
    # vis_impedance_random_setpoint(collision=True)
    
    vis_impedance_random_sawyer_setpoint(collision=False)
    # vis_impedance_sawyer_setpoint(collision=False)
    
    # plot_results(mode='3d')

    # vis_impedance_random_setpoint(collision=False)

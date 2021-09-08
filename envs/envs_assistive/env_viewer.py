import gym, sys, argparse
import numpy as np
np.set_printoptions(precision=5)
import transforms3d as transforms3d
import matplotlib.pyplot as plt
import seaborn as sns
import mujoco_py
from mujoco_py.generated import const
import pybullet as p

from envs.gym_kuka_mujoco.controllers import iMOGVIC
from envs.gym_kuka_mujoco.utils.transform_utils import *
from envs.envs_assistive.feeding_envs import *
from envs.envs_assistive.drinking_envs import *
from envs.envs_assistive.scratch_itch_envs import *
from code.pytorch.LAMPO.core.rl_bench_box import *

import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import time

# from .learn import make_env
# import assistive_gym

import imageio

import commentjson

from code.pytorch.LAMPO.core.rl_bench_box import Mujoco_model, Mujoco_RL_model, AssistiveDRL
from envs.robosuite.robosuite.controllers import *

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()


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
        
        
def vis_impedance_random_sawyer_setpoint(initial_angles=None):
    options = dict()
    num_waypoints = 3
    
    options['model_path'] = 'a_sawyer_test.xml'
    options['rot_scale'] = .3
    options['stiffness'] = np.array([1., 1., 1., 3., 3., 3.])
    
    options['controlled_joints'] = ["robot0_right_j0", "robot0_right_j1",
                                    "robot0_right_j2", "robot0_right_j3",
                                    "robot0_right_j4", "robot0_right_j5",
                                    "robot0_right_j6"]
    
    options['num_waypoints'] = 3
    options['null_space_damping'] = 1.0
    
    import os
    from envs.gym_kuka_mujoco import kuka_asset_dir
    
    model_path = os.path.join(kuka_asset_dir(), 'a_sawyer_test.xml')
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
    
    stiffness_list = np.array([[20., 4., 4., 4., 4., 4.],
                               [0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0.]])
    
    control_scale = np.ones_like(stiffness_list) * 100
    stiffness_list = control_scale * stiffness_list
    print("stiffness_list :::", stiffness_list)
    damping_list = scale_list * np.sqrt(stiffness_list)
    print("damping_list :::", damping_list)
    weight_list = np.array([1.0, 0.05, 0.01])
    controller.set_params_direct(stiffness_list, damping_list, weight_list)
    
    # Set a different random state and run the controller.
    # qpos = np.random.uniform(-1., 1., size=7)
    
    # qpos = np.array([0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
    qpos = initial_angles
    controller.update_initial_joints(qpos)
    qvel = np.zeros(7)
    
    sim_state = sim.get_state()
    sim_state.qpos[:] = qpos
    sim_state.qvel[:] = qvel
    sim.set_state(sim_state)
    sim.forward()
    
    controller.update_state()
    print("current ee_pose :::", controller.ee_pose)
    
    target_pos, target_mat = controller.get_pose_site("target_ee_site")
    print("target ee_pose :::", target_pos)
    
    while True:
        viewer.render()
    
    # set way_points :::
    initial_state = np.array([-0.60349109, 0.09318907, 0.27348721,
                              1.9265446606661554, -0.40240192959667226, -1.541555812071902])
    optimal_state = np.array([-0.80349109, 0.09318907, 0.07348721,
                              1.9265446606661554, -0.40240192959667226, -1.541555812071902])
    state_scale = initial_state - optimal_state
    
    pos_set_list = np.array([[0.1, 0., 1.2],
                             [0.1, 0., 1.2],
                             [0.1, 0., 1.2]])
    quat_set_list = np.array([[-3.128104, 0.00437383, -2.08817412],
                              [-3.128104, 0.00437383, -2.08817412],
                              [-3.128104, 0.00437383, -2.08817412]])
    
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
    
    # for i in range(1):
    #     # qpos = np.random.uniform(-1., 1., size=7)
    #     qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
    #     # qpos = np.array([-0.5538, -0.8208, 0.4155, 0.8409, -0.4955, 0.6482, 1.9628])
    #     qvel = np.zeros(7)
    #     state = np.concatenate([qpos, qvel])
    #
    #     sim_state = sim.get_state()
    #     sim_state.qpos[:] = qpos
    #     sim_state.qvel[:] = qvel
    #     sim.set_state(sim_state)
    #     sim.forward()
    #     for j in range(1000):
    #         controller.update_state()
    #         torque, V, pose_err, vel_err, stiffness_eqv, damping_eqv = controller.update_vic_torque()
    #         energy_list.append(V)
    #         position_list.append(pose_err)
    #         velocity_list.append(vel_err)
    #         stiffness_matrix.append(stiffness_eqv)
    #         damping_matrix.append(damping_eqv)
    #         # torque = controller.get_euler_torque(way_points_list)
    #         # torque = controller.update_torque(way_points_list)
    #         print("final state", np.linalg.norm(controller.state, ord=2))
    #         sim.data.ctrl[:] = torque[:7]
    #         sim.step()
    #         render_frame(viewer, pos_set_list[0, :], quat_set_list[0, :])
    #         viewer.render()

 
def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        # module = importlib.import_module('assistive_gym.envs')
        env_class = globals()[env_name.split('-')[0] + 'Env']
        print(env_class)
        env = env_class()
    env.seed(seed)
    return env


def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()
    

def viewer(env_name):
    coop = 'Human' in env_name
    # env = make_env(env_name, coop=True) if coop else gym.make(env_name)
    
    env = FeedingSawyerHumanEnv()
    # env = DrinkingSawyerHumanEnv()
    
    options = dict()
    
    options['model_path'] = 'a_sawyer_test.xml'
    options['rot_scale'] = .3
    options['stiffness'] = np.array([1., 1., 1., 3., 3., 3.])

    options['controlled_joints'] = ["robot0_right_j0",
                                    "robot0_right_j1",
                                    "robot0_right_j2",
                                    "robot0_right_j3",
                                    "robot0_right_j4",
                                    "robot0_right_j5",
                                    "robot0_right_j6"]

    options['num_waypoints'] = 3
    # options['frame_skip'] = 10
    
    options['null_space_damping'] = 1.0

    param_dir = '/home/zhimin/code/5_thu/rl-robotic-assembly-control/code/pytorch/LAMPO/params/'
    param_file = 'IMOGICAssitive.json'

    param_file = os.path.join(param_dir, param_file)
    with open(param_file) as f:
        set_params = commentjson.load(f)
     
    mujoco_model = Mujoco_model(set_params["controller_options"], render=True)
    
    while True:
        done = False
        env.render()
        
        # observation, spoon_pos, spoon_orient = env.reset()
        observation = env.reset()
        # print("Initial observation :", observation)
        print('+' * 100)
        # spoon_pos_inital, spoon_orient_initial = env.get_tool_pose()
        print("robot_joint_angles :", observation['robot_joint_angles'])
        # print("spoon pos :", spoon_pos, "spoon orient :", spoon_orient)
        # print("spoon orient :", spoon_orient)
        
        # set way points : target pose
        pos, ori = env.robot.get_ee_pose()
        start_euler = transforms3d.euler.quat2euler(ori, 'sxyz')
        print("EE pos :", pos, "ee euler :", start_euler)
        
        # target_pose = env.get_context()
        # print("target pos :", env.target_pos)
        # # print("target ori :", env.target_orient)
        # print("target euler :", transforms3d.euler.quat2euler(env.target_orient, 'sxyz'))
        # # p.getQuaternionFromEuler()
        
        target_euler = transforms3d.euler.quat2euler(env.target_orient, 'sxyz')
        print("target pos :", env.target_pos, "target euler :", target_euler)

        delta_pos = env.target_pos - spoon_pos
        
        # delta_pos = np.array([0., 0.3, 0.1])
        # print("des_pos", pos + delta_pos)
        # print('+' * 100)
        print("delta_pos :", delta_pos)
        
        ee_pose = mujoco_model.reset(observation['robot_joint_angles'])
        print("initial_pose :", np.array(ee_pose))
        
        print('+' * 100)
        mujoco_model.set_waypoints(ee_pose[:3] + delta_pos, ee_pose[3:])
        # mujoco_model.set_waypoints(ee_pose[:3] + np.array([0., -0.1, 0.2]), transforms3d.euler.quat2euler(ee_pose[3:], 'sxyz'))
        
        # set impedance params
        mujoco_model.set_impedance_params(params=None)
        
        # time.sleep(10)
        joint_list = []
        joint_last = observation['robot_joint_angles']
        time_steps = 0
        while not done:
            # action = sample_action(env, coop)
            joint = mujoco_model.step(np.zeros(7))
            # print("robot joints :", joint[0])

            human_action = np.zeros(env.action_human_len)

            action = {'robot': joint[0].copy() - joint_last, 'human': human_action}  # env.action_space_human.sample()
            joint_list.append(joint[0].copy())

            # print("sample_action :", action)
            observation, reward, done, info = env.step(action)
            # print('robot joints pybullet:', observation['robot_joint_angles'])
            if coop:
                done = done['__all__']

            # print('Robot reward:', reward['robot'], 'Human reward:', reward['human'])

            # time.sleep(0.1)

            joint_last = observation['robot_joint_angles']
            time_steps += 1

        print('+' * 100)
        print("target pos :", env.target_pos)
        print("target euler :", transforms3d.euler.quat2euler(env.target_orient, 'sxyz'))
        spoon_pos, spoon_orient = env.get_tool_pose()
        print("spoon pos :", spoon_pos)
        print("spoon orient :", spoon_orient)
        
        # set way points : target pose
        pos, ori = env.robot.get_ee_pose()
        start_euler = transforms3d.euler.quat2euler(ori, 'sxyz')
        print("EE pos :", pos, "ee euler :", start_euler)
        print("time_steps :", time_steps)
        
        print("Mujoco ee pose :", mujoco_model.get_ee_pose())
        
        print("Error :", mujoco_model.get_ee_pose() - ee_pose)
        print("Pybullet error :", spoon_pos - spoon_pos_inital)


def mujoco_eval(env_name):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)

    env.render()
    # env.reset()
    observation = env.reset()

    print("obs :", observation['robot_joint_angles'])
    
    param_dir = '/home/zhimin/code/5_thu/rl-robotic-assembly-control/code/pytorch/LAMPO/params/'
    param_file = 'VICESAssitiveItch.json'
    
    param_file = os.path.join(param_dir, param_file)
    with open(param_file) as f:
        params = commentjson.load(f)
    
    mujoco_model = Mujoco_RL_model(params["controller_options"], render=True)
    
    # qpos = np.array([0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 1.6482, 1.9628])
    # qpos = np.array([1.73155, 1.91932, 1.47255, -2.29171, 0.42262, 1.13446, 1.75369])
    qpos = np.array([[1.73155, 1.91932, 1.47255, 3.99147, 0.42262, 1.13446, 1.75369]])
    ee_pose = mujoco_model.reset(qpos)
    
    # # ee_pose = mujoco_model.reset(np.array([2.95, 4.07, -0.06, 1.44171, -6.2, 3.7, -0.35369]))
    # # print("ee_pose :", ee_pose)
    # # # print("goal_pose :", mujoco_model.controller.goal_pos, mujoco_model.controller.goal_ori)
    # # # # ee_pose = mujoco_model.reset(np.zeros(7))
    # # # ee_euler = np.array(transforms3d.euler.euler2mat(ee_pose[3], ee_pose[4], ee_pose[5], 'sxyz'))
    #
    # while True:
    #     mujoco_model.viewer_render()
    #
    #     action = np.array([-0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    #     joint = mujoco_model.step(action)
    #     # # joint = mujoco_model.step(action, set_pos=ee_pose[:3], set_ori=ee_euler)
    #     print("ee_pose :", mujoco_model.get_ee_pose())
    
    # while True:
    #     action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #     target_euler = transforms3d.euler.mat2euler(mujoco_model.controller.goal_ori, 'sxyz')
    #     print("target pos :", mujoco_model.controller.goal_pos, "target euler :", target_euler)
    #     joint = mujoco_model.step(action)
    #     print("ee_pose :", mujoco_model.get_ee_pose())
    #     mujoco_model.viewer_render()
    
    # done = False
    # while not done:
    #     # action = sample_action(env, coop)
    #     action = np.array([0., 0.00, 0.0, 0.0, 0.0, 0.])
    #     print("goal_pos :", mujoco_model.controller.goal_pos)
    #     print("ee_pos :", mujoco_model.get_ee_pose())
    #     joint = mujoco_model.step(action, set_pos=ee_pose[:3], set_ori=ee_euler)
        
    while True:
        done = False
        env.render()

        print('+' * 100)
        observation = env.reset()

        if observation['robot_joint_angles'] is not None:
            print("Done !!!")

        # observation, spoon_pos, spoon_orient = env.reset()
        # print("Initial observation :", observation)

        # spoon_pos_inital, spoon_orient_initial = env.get_tool_pose()
        # print("robot_joint_angles :", observation['robot_joint_angles'])
        # print("spoon pos :", spoon_pos, "spoon orient :", spoon_orient)
        # print("spoon orient :", spoon_orient)

        human_action = np.zeros(env.action_human_len)

        # action = {'robot': np.array([0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.0]), 'human': human_action}  # env.action_space_human.sample()
        #
        # # print("sample_action :", action)
        # observation, reward, done, info = env.step(action)
        print("robot_joint_angles :", observation['robot_joint_angles'])
        done = False

        # set way points : target pose
        pos, ori = env.robot.get_ee_pose()
        start_euler = transforms3d.euler.quat2euler(ori, 'sxyz')
        start_ori = transforms3d.euler.euler2mat(start_euler[0], start_euler[1], start_euler[2], 'sxyz')
        print("EE pos :", pos, "ee euler :", start_euler)

        # target_pose = env.get_context()
        # print("target pos :", env.target_pos)
        # # print("target ori :", env.target_orient)
        # print("target euler :", transforms3d.euler.quat2euler(env.target_orient, 'sxyz'))
        # # p.getQuaternionFromEuler()

        # target_euler = transforms3d.euler.quat2euler(env.target_orient, 'sxyz')
        # print("target pos :", env.target_pos, "target euler :", target_euler)

        # delta_pos = env.target_pos - spoon_pos
        # # delta_pos = np.array([0., 0.3, 0.1])
        # # print("des_pos", pos + delta_pos)
        # # print('+' * 100)
        # print("delta_pos :", delta_pos)

        initial_angle = observation['robot_joint_angles']
        # initial_angle[6] = 1.534
        ee_pose = mujoco_model.reset(initial_angle)
        print("initial_pose :", np.array(ee_pose))
        ee_euler = np.array(transforms3d.euler.euler2mat(ee_pose[3], ee_pose[4], ee_pose[5], 'sxyz'))

        print('+' * 100)
        # time.sleep(10)
        joint_list = []
        joint_last = observation['robot_joint_angles']
        time_steps = 0
        
        # action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # joint_last = mujoco_model.step(action, set_pos=ee_pose[:3], set_ori=start_ori)

        # time.sleep(2)
        while not done:
            # action = sample_action(env, coop)
            action = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.])
            joint = mujoco_model.step(action)

            print("Robot Joints :", joint)

            human_action = np.zeros(env.action_human_len)

            action = {'robot': joint.copy() - joint_last, 'human': human_action}  # env.action_space_human.sample()
            joint_list.append(joint.copy())

            # print("sample_action :", action)
            observation, reward, done, info = env.step(action)
            # print('robot joints pybullet:', observation['robot_joint_angles'])

            if coop:
                done = done['__all__']

            # print('Robot reward:', reward['robot'], 'Human reward:', reward['human'])

            # time.sleep(0.1)

            joint_last = observation['robot_joint_angles']
            time_steps += 1

        # print('+' * 100)
        # print("target pos :", env.target_pos)
        # print("target euler :", transforms3d.euler.quat2euler(env.target_orient, 'sxyz'))
        # spoon_pos, spoon_orient = env.get_tool_pose()
        # print("spoon pos :", spoon_pos)
        # print("spoon orient :", spoon_orient)

        # set way points : target pose
        pos, ori = env.robot.get_ee_pose()
        start_euler = transforms3d.euler.quat2euler(ori, 'sxyz')
        print("EE pos :", pos, "ee euler :", start_euler)
        print("time_steps :", time_steps)

        print("Mujoco ee pose :", mujoco_model.get_ee_pose())

        print("Error :", mujoco_model.get_ee_pose() - ee_pose)
        # print("Pybullet error :", spoon_pos - spoon_pos_inital)


def viewer_mujoco(env_name, params):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)

    env = AssistiveDRL(env, params, logdir='')
    
    # Grab name of this rollout combo
    video_name = "{}-{}-{}".format(
        "env_test", "".join("jaco"), "controller_osc").replace("_", "-")
    
    # Calculate appropriate fps
    fps = int(10)
    
    # Define video writer
    video_writer = imageio.get_writer("{}.mp4".format(video_name), fps=fps)
    
    while True:
        done = False
        env.render()
        observation = env.reset()
        action = sample_action(env, coop)
        print(observation)
        
        env.setup_camera(camera_width=1920 // 2, camera_height=1080 // 2)
        if coop:
            print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:',
                  np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:',
                  np.shape(action['human']))
        else:
            print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))
        
        while not done:
            observation, reward, done, info = env.step(sample_action(env, coop))
            img, _ = env.get_camera_image_depth()
            video_writer.append_data(img)
            env.render()
            if coop:
                done = done['__all__']
                

def viewer_pybullet(env_name, params):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)

    # task = globals()[params["alg_options"]["task_class"]](args, env, '')
    
    env = AssistiveDRL(env, params, logdir='')
    # task.reset()
    # while True:
    #     task._env.render()
    
    # task.send_movement(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    
    env.reset()
    while True:
        # env.view_render()
        env.step(np.array([0.0, 0.0, 1, 0.0, 0.0, 0.0]))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env',
                        # default="ScratchItchJacoHuman-v1",
                        default="DrinkingSawyerHuman-v1",
                        # default="FeedingSawyerHuman-v1",
                        help='Environment to test (default: ScratchItchJaco-v1)')

    parser.add_argument('--video_record',
                        type=bool,
                        default=False,
                        help='the parameter file to use')

    parser.add_argument('--video_path',
                        type=str,
                        default='IMOGICAssitive.json',
                        help='the parameter file to use')
    
    args = parser.parse_args()

    param_dir = '/home/zhimin/code/5_thu/rl-robotic-assembly-control/code/pytorch/LAMPO/params/'
    # param_file = 'IMOGICAssitiveJaco.json'
    param_file = 'VICESAssitive.json'

    param_file = os.path.join(param_dir, param_file)
    with open(param_file) as f:
        params = commentjson.load(f)

    # mujoco_eval(args.env)
    # viewer_mujoco(args.env, params)
    viewer_pybullet(args.env, params)
    
    # env = DrinkingSawyerHumanEnv()
    # done = False
    # env.render()
    # observation, spoon_pos, spoon_orient = env.reset()
    #
    # # time.sleep(10)
    # joint_list = []
    # # joint_last = observation['robot_joint_angles']
    # time_steps = 0
    # coop = True
    # while not done:
    #     action = sample_action(env, coop)
    #
    #     # print("sample_action :", action)
    #     observation, reward, done, info = env.step(action)
    #     # print('robot joints pybullet:', observation['robot_joint_angles'])
    #     if coop:
    #         done = done['__all__']
    #
    #     # print('Robot reward:', reward['robot'], 'Human reward:', reward['human'])
    #
    #     # time.sleep(0.1)
    #
    #     # joint_last = observation['robot_joint_angles']
    #     time_steps += 1
        
    # viewer(args.env)
    # mujoco_eval(args.env)
    # viewer_mujoco(args.env)
    
    # print(p.getQuaternionFromEuler([0, 1.57, 0]))

    # controller_name = "OSC_POSE"
    # controller_path = os.path.join('/home/zhimin/code/5_thu/rl-robotic-assembly-control/envs/robosuite/robosuite/',
    #                                'controllers/config/{}.json'.format(controller_name.lower()))
    # controller_config = load_controller_config(custom_fpath=controller_path)
    # print("controller_config :", controller_config)
    
    # controller_config['sim'] = self.sim
    # controller_config["eef_name"] = "ee_site"

    # controller_config["joint_indexes"] = {
    #     "joints": self.joint_indexes,
    #     "qpos": self._ref_joint_pos_indexes,
    #     "qvel": self._ref_joint_vel_indexes
    # }

    # controller_config["impedance_mode"] = "variable"
    # controller_config["kp_limits"] = [0, 300]
    # controller_config["damping_limits"] = [0, 10]

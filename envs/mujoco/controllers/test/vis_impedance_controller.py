# System imports
# Package imports
import mujoco_py
import mujoco_py as mjp
import numpy as np
from mujoco_py.generated import const
import time

# Local imports
from envs.mujoco.controllers import ImpedanceControllerV2, RLVIC
from envs.mujoco.controllers.test.common import create_sim
from envs.mujoco.utils.quaternion import *
import copy as cp

from envs.mujoco.utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices, \
	get_joint_indices

from envs.mujoco.utils.transformations import quaternion_from_euler
import numpy as np

from envs.mujoco.mujoco_config import MujocoConfig


def render_frame(viewer, pos, quat):
	viewer.add_marker(pos=pos,
					  label='',
					  type=const.GEOM_SPHERE,
					  size=[.01, .01, .01])
	mat = quat2Mat(quat)
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


def vis_impedance_fixed_setpoint(collision=False):
	options = dict()
	options['model_path'] = 'full_kuka_mesh_collision.xml'
	options['rot_scale'] = .3
	options['stiffness'] = np.array([10., 10., 10., 10., 10., 10.])
	
	sim = create_sim(collision=collision)
	controller = ImpedanceControllerV2(sim, **options)
	
	viewer = mujoco_py.MjViewer(sim)
	for i in range(10):
		
		# Set a random state to get a random feasible setpoint.
		qpos = np.random.uniform(-1., 1, size=7)
		qvel = np.zeros(7)
		state = np.concatenate([qpos, qvel])
		
		sim_state = sim.get_state()
		sim_state.qpos[:] = qpos
		sim_state.qvel[:] = qvel
		sim.set_state(sim_state)
		sim.forward()
		
		controller.set_action(np.zeros(6))
		
		# Set a different random state and run the controller.
		qpos = np.random.uniform(-1., 1., size=7)
		qvel = np.zeros(7)
		state = np.concatenate([qpos, qvel])
		
		sim_state = sim.get_state()
		sim_state.qpos[:] = qpos
		sim_state.qvel[:] = qvel
		sim.set_state(sim_state)
		sim.forward()
		
		for i in range(3000):
			sim.data.ctrl[:] = controller.get_torque()
			sim.step()
			render_frame(viewer, controller.pos_set, controller.quat_set)
			print('pos_set:::', controller.pos_set)
			viewer.render()


def vis_impedance_random_setpoint(collision=False):
	import os
	from envs.mujoco import kuka_asset_dir
	
	options = dict()
	options['model_path'] = 'full_kuka_mesh_collision.xml'
	# options['model_path'] = "full_peg_insertion_experiment_no_gravity_moving_hole_id=025.xml"
	options['rot_scale'] = .3
	
	options['stiffness'] = np.array([40, 40., 2., 5., 5., 5.])
	controlled_joints = ["kuka_joint_1", "kuka_joint_2", "kuka_joint_3", "kuka_joint_4",
						 "kuka_joint_5", "kuka_joint_6", "kuka_joint_7"]
	
	model_path = os.path.join(kuka_asset_dir(), 'full_kuka_mesh_collision.xml')
	model = mujoco_py.load_model_from_path(model_path)
	sim = mujoco_py.MjSim(model)
	# sim = create_sim(collision=collision)
	# controller = ImpedanceControllerV2(sim, **options)
	
	controller = RLVIC(sim, **options)
	
	frame_skip = 5
	high = np.array([.1, .1, .1, 2, 2, 2])
	low = -np.array([.1, .1, .1, 2, 2, 2])
	
	viewer = mujoco_py.MjViewer(sim)
	
	# mujoco_config_kuka = MujocoConfig('/home/zhimin/code/5_tsinghua_assembly_projects/rl-robotic-assembly-control/envs/mujoco/envs/assets/full_kuka_mesh_collision')
	# mujoco_config_kuka._connect(sim,
	# 							joint_pos_addrs=get_qpos_indices(sim.model, controlled_joints),
	# 							joint_vel_addrs=get_qvel_indices(sim.model, controlled_joints),
	# 							joint_dyn_addrs=get_joint_indices(sim.model, controlled_joints))
	#
	# M = mujoco_config_kuka.M(np.array([-2.71426487,  0.73422411, -0.12319505,  1.81347569, -3.42125145, 0.9523561, 0.71809284]))
	#
	# print("inertia Matrix ::", M.shape)
	# sim_state = sim.get_state()
	# print("sim_state_qpos :::", sim_state.qpos[:])
	
	for i in range(1):
		
		# Set a different random state and run the controller.
		# qpos = np.random.uniform(-1., 1., size=7)
		# qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
		# qpos = np.array([-3.0690824, 0.48358201, -0.22983445, 2.14603292, -3.71142297,
		# 			  1.71805615, 0.88579455])
		qpos = np.array([-2.71426487,  0.73422411, -0.12319505,  1.81347569, -3.42125145, 0.9523561, 0.71809284])
		qvel = np.zeros(7)
		state = np.concatenate([qpos, qvel])
		
		sim_state = sim.get_state()
		print('sim_state :::', sim_state)
		sim_state.qpos[:] = qpos
		sim_state.qvel[:] = qvel
		sim.set_state(sim_state)
		sim.forward()
		
		init_pos, init_quat, init_vel = controller.get_state_cartersian_sapce()
		
		print('pos :::', init_pos)
		print('quat :::', init_quat)
		print('angle vel :::', quat2Vel(init_quat))
		print('angle :::', quat2Vel(init_quat))
		
		optimal_pos, optimal_quat = init_pos - np.array([-0.0, 0.0, 0.2]), init_quat
		
		position_list = np.array([[0., 0., 0.],
								  # [0.08, 0.0, 0.0],
								  # [0.0, 0.0, -0.04]
								  ])
		# quat_list = np.array([[0., 0., 0., 0.,], init_quat, init_quat])
		quat_list = np.array([[0., 0., 0., 0.]])
		
		# weights = np.array([1.0, 0.1, 0.5])
		weights = np.array([1.0])
		controller.set_waypoints(
			pos_list=position_list,
			quat_list=quat_list,
			pos_optimal_point=optimal_pos,
			quat_optimal_point=optimal_quat
		)
		
		stiffness_list = np.array([[4, 0, 1, 0, 0, 0],
								   # [10, 1, 0, 0, 0, 0],
								   # [0, 0, 10, 0, 0, 0]
								   ])
		
		damping_list = np.array([[4, 0, 2, 0, 0, 0],
								 # [0, 0, 0, 0, 0, 0],
								 # [0, 0, 25, 0, 0, 0]
								 ])
		
		velocity_list = []
		position_list = []
		stiffness_plannar = []
		energy_list = []
		
		for index in range(1000):
			# controller.set_action(np.array([0.0, 0.0, -0.0, 0.0, 0.0, 0.0]))
			for i in range(frame_skip):
				sim.data.ctrl[:], V, stiffness_eqv, damping_eqv, state_vel = controller.get_torque_vic(
					stiffness=stiffness_list,
					damping=damping_list,
					weights=weights
				)
				sim.step()
				render_frame(viewer, controller.pos_set, controller.quat_set)
				viewer.render()
				energy_list.append(cp.deepcopy(V))

			pos, quat, xvel = controller.get_state_cartersian_sapce()
			# print('each pose ::: ', pos)
			# print('each vel:::', xvel)
			position_list.append(cp.deepcopy(pos))
			velocity_list.append(cp.deepcopy(state_vel[:3]))

			if index % 5 == 0:
				print("Stiffness Append :::")
				stiffness_plannar.append(cp.deepcopy([pos, np.diag(stiffness_eqv)[0], np.diag(stiffness_eqv)[2]]))

		sim_state = sim.get_state()
		print('sim_state final :::', sim_state)
		final_pos, final_quat, final_vel = controller.get_state_cartersian_sapce()
		print('final_pos :::', final_pos)
		print('final_quat :::', final_quat)

		return position_list, velocity_list, stiffness_plannar, energy_list, optimal_pos


def vis_impedance_3d_random_setpoint(collision=False):
	options = dict()
	options['model_path'] = 'full_kuka_mesh_collision.xml'
	# options['model_path'] = "full_peg_insertion_experiment_no_gravity_moving_hole_id=025.xml"
	options['rot_scale'] = .3
	
	options['stiffness'] = np.array([40, 100., 2., 5., 5., 5.])
	controlled_joints = ["kuka_joint_1", "kuka_joint_2", "kuka_joint_3", "kuka_joint_4",
						 "kuka_joint_5", "kuka_joint_6", "kuka_joint_7"]
	
	sim = create_sim(collision=collision)
	# controller = ImpedanceControllerV2(sim, **options)
	controller = RLVIC(sim, **options)
	
	frame_skip = 5
	high = np.array([.1, .1, .1, 2, 2, 2])
	low = -np.array([.1, .1, .1, 2, 2, 2])
	
	viewer = mujoco_py.MjViewer(sim)
	
	for i in range(1):
		#####
		# reset state and target state
		#####
		# Set a different random state and run the controller.
		# qpos = np.random.uniform(-1., 1., size=7)
		# qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
		# qpos = np.array([-3.0690824, 0.48358201, -0.22983445, 2.14603292, -3.71142297,
		# 			  1.71805615, 0.88579455])
		qpos = np.array([-2.71426487, 0.73422411, -0.12319505, 1.81347569, -3.42125145, 0.9523561, 0.71809284])
		qvel = np.zeros(7)
		# state = np.concatenate([qpos, qvel])
		
		sim_state = sim.get_state()
		sim_state.qpos[:] = qpos
		sim_state.qvel[:] = qvel
		sim.set_state(sim_state)
		sim.forward()
		
		init_pos, init_quat, init_vel = controller.get_state_cartersian_sapce()
		
		print('pos :::', init_pos)
		print('quat :::', init_quat)
		print('angle vel :::', quat2Vel(init_quat))
		print('angle :::', quat2Vel(init_quat))
		
		optimal_pos, optimal_quat = init_pos - np.array([-0.5, 0.1, 0.6]), init_quat
		
		position_list = np.array([[0., 0., 0.],
								  # [0.08, 0.0, 0.0],
								  # [0.0, 0.0, -0.04]
								  ])
		
		# quat_list = np.array([[0., 0., 0., 0.,], init_quat, init_quat])
		quat_list = np.array([[0., 0., 0., 0.]])
		
		# weights = np.array([1.0, 0.1, 0.5])
		weights = np.array([1.0])
		controller.set_waypoints(
			pos_list=position_list,
			quat_list=quat_list,
			pos_optimal_point=optimal_pos,
			quat_optimal_point=optimal_quat
		)
		
		stiffness_list = np.array([[4, 4, 1, 0, 0, 0],
								   # [10, 1, 0, 0, 0, 0],
								   # [0, 0, 10, 0, 0, 0]
								   ])
		
		damping_list = np.array([[4, 4, 2, 0, 0, 0],
								 # [0, 0, 0, 0, 0, 0],
								 # [0, 0, 25, 0, 0, 0]
								 ])
		
		velocity_list = []
		position_list = []
		stiffness_plannar = []
		energy_list = []
		
		for index in range(500):
			for i in range(frame_skip):
				sim.data.ctrl[:], V, stiffness_eqv, damping_eqv, state_vel = controller.get_torque_impedance(
					stiffness=stiffness_list,
					damping=damping_list,
					weights=weights
				)
				sim.step()
				render_frame(viewer, controller.pos_set, controller.quat_set)
				viewer.render()
				energy_list.append(cp.deepcopy(V))
			
			pos, quat, xvel = controller.get_state_cartersian_sapce()
			# print('each pose ::: ', pos)
			# print('each vel:::', xvel)
			position_list.append(cp.deepcopy(pos))
			velocity_list.append(cp.deepcopy(state_vel[:3]))
			
			if index % 5 == 0:
				print("Stiffness Append :::")
				stiffness_plannar.append(cp.deepcopy([pos, np.diag(stiffness_eqv)[0], np.diag(stiffness_eqv)[2]]))
		
		return position_list, velocity_list, stiffness_plannar, energy_list, optimal_pos


def plot_results(mode='3d'):
	from mpl_toolkits.mplot3d import Axes3D
	FONT_SIZE = 20
	plt.rcParams['font.family'] = 'Times New Roman'
	plt.rcParams['font.size'] = FONT_SIZE
	
	position_list, velocity_list, stiffness_planner, energy_list, optimal_pos \
		= vis_impedance_random_setpoint(collision=True)
	
	# print("stiffness_planner", stiffness_planner)
	# print(position_list)
	# print(position_list[0])
	# print('init_pos', init_pos)
	
	position_list = np.array(position_list)
	velocity_list = np.array(velocity_list)
	energy_list = np.array(energy_list)
	
	if mode=='3d':
		fig = plt.figure(figsize=(10, 8), dpi=300)
		# ax = fig.add_subplot(projection='3d')
		ax = fig.gca(projection='3d')
		plt.title('Position')
		ax.plot(position_list[:, 0] - optimal_pos[0],
				 position_list[:, 1] - optimal_pos[1],
				 position_list[:, 2] - optimal_pos[2], linewidth=3.5)
		
		# plt.scatter(position_list[0, 0] - optimal_pos[0],
		# 			position_list[0, 1] - optimal_pos[1],
		# 			position_list[0, 2] - optimal_pos[2],
		# 			s=100, marker='^', c='g')
		# plt.scatter(position_list[-1, 0] - optimal_pos[0],
		# 			position_list[-1, 1] - optimal_pos[1],
		# 			position_list[-1, 2] - optimal_pos[2],
		# 			s=100, marker='^', c='r')
		plt.xlabel('X(m)')
		plt.ylabel('Y(m)')
		# plt.zlabel('Z(m)')
		plt.show()
		
		plt.figure(figsize=(10, 8), dpi=300)
		plt.title('Energy')
		print("energy_list :::", energy_list)
		# velocity_list = np.array(velocity_list)
		plt.plot(energy_list, linewidth=3.5)
		plt.show()
	else:
		plt.figure(figsize=(10, 8), dpi=300)
		position_list = np.array(position_list)
		plt.title('Position')
		plt.plot(position_list[:, 0] - optimal_pos[0], position_list[:, 2] - optimal_pos[2], linewidth=3.5)
		plt.scatter(position_list[0, 0] - optimal_pos[0], position_list[0, 2] - optimal_pos[2],
					s=100, marker='^', c='g')
		plt.scatter(position_list[-1, 0] - optimal_pos[0], position_list[-1, 2] - optimal_pos[2],
					s=100, marker='^', c='r')
		plt.xlabel('X(m)')
		plt.ylabel('Z(m)')
		plt.show()
	
		plt.figure(figsize=(10, 8), dpi=300)
		plt.title('Velocity')
		plt.plot(velocity_list[:, 0], velocity_list[:, 2], linewidth=3.5)
		plt.scatter(velocity_list[0, 0], velocity_list[0, 2], s=100, marker='^', c='g')
		plt.scatter(velocity_list[-1, 0], velocity_list[-1, 2], s=100, marker='^', c='r')
		plt.scatter(velocity_list[100, 0], velocity_list[100, 2], s=100, marker='^', c='black')
		plt.xlabel('X(m)')
		plt.ylabel('Z(m)')
		plt.show()
		
		plt.figure(figsize=(10, 8), dpi=300)
		plt.title('Energy')
		print("energy_list :::", energy_list)
		# velocity_list = np.array(velocity_list)
		plt.plot(energy_list, linewidth=3.5)
		plt.show()
	
		# fig = plt.figure(figsize=(10, 8), dpi=300)
		# ax = fig.add_subplot(111, aspect='auto')
		# position_list = np.array(position_list)
		# # position_list = np.array([[1, 2], [0.5, 1]])
		# # stiffness_list = np.array([[4, 1], [4, 2]])
		# plt.title('Position')
		# # plt.plot(position_list[:, 0], position_list[:, 1], linewidth=3.5)
		# plt.plot(position_list[:, 0], position_list[:, 2], linewidth=3.5)
		# # plt.scatter(position_list[0, 0] - init_pos[0] - 0.2, position_list[0, 2] - init_pos[2] + 0.4,
		# # 			s=100, marker='^', c='g')
		# # plt.scatter(position_list[-1, 0] - init_pos[0] - 0.2, position_list[-1, 2] - init_pos[2] + 0.4,
		# # 			s=100, marker='^', c='r')
		# scale = 4
		# for i in range(len(stiffness_planner)):
		# 	el = patches.Ellipse((stiffness_planner[i][0][0], stiffness_planner[i][0][2]),
		# 						  stiffness_planner[i][1]/np.linalg.norm((stiffness_planner[i][1], stiffness_planner[i][2]), ord=2)/scale,
		# 						  stiffness_planner[i][2]/np.linalg.norm((stiffness_planner[i][1], stiffness_planner[i][2]), ord=2)/scale,
		# 						  alpha=0.5, edgecolor='black', linewidth=3.5)
		# 	ax.add_patch(el)
		# plt.xlim([0, 0.6])
		# plt.ylim([0.2, 1])
		# plt.show()


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	import numpy as np
	from matplotlib import patches
	
	plot_results(mode='3d')
	
# for i in range(100):
# 	# controller.set_action(np.array([0.0, 0.0, -0.0, 0.0, 0.0, 0.0]))
# 	for i in range(frame_skip):
# 		sim.data.ctrl[:] = controller.get_torque()
# 		sim.step()
# 		render_frame(viewer, controller.pos_set, controller.quat_set)
# 		viewer.render()
# 	pos, quat, xvel = controller.get_state_cartersian_sapce()
# 	# print('each pose ::: ', pos)
# 	# print('each vel:::', xvel)
# 	position_list.append(cp.deepcopy(pos))
# 	velocity_list.append(cp.deepcopy(xvel))

# stiffness_1 = np.array([4, 4, 1, 1, 1, 1])
# stiffness_2 = np.array([1, 1, 4, 1, 1, 1])
# stiffness_list = np.array([[4, 4, 0.1, 1, 1, 1],
# 						   [0.5, 0.5, 8, 1, 1, 1]])

# controller.set_target(np.array([0.2, 0.0, 0.8]), quaternion_from_euler(-np.pi/4, 0., - np.pi/2))
# controller.set_target(init_pos - np.array([-0.2, 0.0, 0.2]), init_quat)

# while True:
# 	render_frame(viewer, controller.pos_set, controller.quat_set)
# 	viewer.render()

# weights = np.array([0.9, 0.1])
# # weights = np.array([1.0])
# total_reward = 0.0
# while True:
# 	# obs, total_reward, done, info, target_obs = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
# 	obs, reward, done, info, target_obs, action = controller.get_torque_im(stiffness_list, weights)
# 	print("single_step_reward :::", reward)
# 	print("action :::", action)
# 	total_reward += reward
# 	target_distance = np.linalg.norm((target_obs[0] - target_obs[1]), ord=2)
# 	# print("Done or not !!!", done)
# 	# print("target_distance :::", target_distance)
#
# 	env.render()
# 	time.sleep(0.1)

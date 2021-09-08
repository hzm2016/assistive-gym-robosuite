# System imports

# Package imports
import mujoco_py
from mujoco_py.generated import const
import numpy as np

# Local imports
from envs.mujoco.controllers import ImpedanceControllerV2, FullImpedanceController
from envs.mujoco.utils.kinematics import forwardKinSite, forwardVelKinSite
from envs.mujoco.utils.quaternion import mat2Quat, quat2Mat
from envs.mujoco.controllers.test.common import create_sim
from envs.mujoco.utils import transformations
from envs.mujoco.utils.mujoco_utils import get_qpos_indices, get_qvel_indices, get_actuator_indices, get_joint_indices

# path planner
from envs.mujoco import path_planners

import argparse
import imageio
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)


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


def generate_reference_path(start_pos, start_orientation,
                       target_pos, target_orientation, n_timesteps=500):
    """
    generate reference path
    :param start_pos:
    :param starting_orientation:
    :param target_pos:
    :param target_orientation:
    :return:
    """
    # n_timesteps = 500
    position_planner = path_planners.SecondOrderDMP(
        error_scale=1,
        n_timesteps=n_timesteps
    )
    orientation_path = path_planners.Orientation()
    
    position_planner.generate_path(position=start_pos,
                                   target_position=target_pos)
    
    orientation_path.match_position_path(
        orientation=start_orientation,
        target_orientation=target_orientation,
        position_path=position_planner.position_path,
    )
    return position_planner, orientation_path, target_pos, target_orientation


def plot_result(ee_track, ee_angles_track, ee_vel_track,
                ee_ref_track, ee_ref_angles_track, ee_ref_vel_track,
                object_track, object_angles_track):
    print("Simulation terminated...")
    ctrlr_dof = [True, True, True, True, True, True]
    dof_labels = ["x", "y", "z", "a", "b", "g"]
    
    ee_track = np.array(ee_track)
    ee_angles_track = np.array(ee_angles_track)
    ee_vel_track = np.array(ee_vel_track)
    
    ee_ref_track = np.array(ee_ref_track)
    ee_ref_angles_track = np.array(ee_ref_angles_track)
    ee_ref_vel_track = np.array(ee_ref_vel_track)
    
    object_track = np.array(object_track)
    object_angles_track = np.array(object_angles_track)
    
    if ee_track.shape[0] > 0:
        # plot distance from target and 3D trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d  # pylint: disable=W0611
        
        fig = plt.figure(figsize=(8, 12))
        
        ax1 = fig.add_subplot(411)
        ax1.set_ylabel("3D position (m)")
        for ii, controlled_dof in enumerate(ctrlr_dof[:3]):
            if controlled_dof:
                ax1.plot(ee_track[:, ii], label=dof_labels[ii])
                ax1.plot(ee_ref_track[:, ii], "--", label=dof_labels[ii] + "_ref")
        ax1.legend()
        
        ax2 = fig.add_subplot(412)
        for ii, controlled_dof in enumerate(ctrlr_dof[3:]):
            if controlled_dof:
                ax2.plot(ee_angles_track[:, ii], label=dof_labels[ii + 3])
                ax2.plot(ee_ref_angles_track[:, ii], "--", label=dof_labels[ii + 3] + "_ref")
        ax2.set_ylabel("3D orientation (rad)")
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        
        ax3 = fig.add_subplot(413, projection="3d")
        ax3.set_title("End-Effector Trajectory")
        ax3.plot(ee_track[:, 0], ee_track[:, 1], ee_track[:, 2], label="ee_xyz")
        ax3.plot(ee_ref_track[:, 0], ee_ref_track[:, 1], ee_ref_track[:, 2], label="ref_xyz")
        ax3.scatter(
            object_track[:, 0],
            object_track[:, 1],
            object_track[:, 2],
            label="object",
            c="r",
        )
        ax3.legend()
        # ax3.scatter(
        #     object_track[:, 0],
        #     object_track[:, 1],
        #     object_track[:, 2],
        #     label="object",
        #     c="g",
        # )
        # for j in range(len(iterative_list)):
        #     ax3.scatter(
        #         release_xyz_track_list[j][0][0],
        #         release_xyz_track_list[j][0][1],
        #         release_xyz_track_list[j][0][2],
        #         label="release_" + str(j),
        #         marker='p',
        #         s=60,
        #         c="r",
        #     )
        
        ax4 = fig.add_subplot(414)
        ax4.set_ylabel("3D velocity (m/s)")
        for ii, controlled_dof in enumerate(ctrlr_dof[:3]):
            if controlled_dof:
                ax4.plot(ee_vel_track[:, ii], label=dof_labels[ii])
                ax4.plot(ee_ref_vel_track[:, ii], "--", label=dof_labels[ii] + "_ref")
                ax4.plot(object_angles_track[:, ii], "--", label=dof_labels[ii] + "_object")
        ax4.legend()
        
        fig.tight_layout()
        # plt.savefig("./video/release_position.png")
        plt.show()


def send_target_angles(sim, idx, q):
    """
        Move the robot to the specified configuration.
        Parameters
        ----------
        q: np.array
            configuration to move to [radians]
    """
    sim.data.qpos[idx] = np.copy(q)
    sim.forward()


def set_object_initial_pose(sim, joint_name, q):
    """
        Move the robot to the specified configuration.
        Parameters
        ----------
        q: np.array
            configuration to move to [radians]
    """
    sim.data.set_joint_qpos(joint_name, np.copy(q))
    sim.forward()


def vis_impedance_fixed_setpoint(collision=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="SawyerStack")
    parser.add_argument("--video_path", type=str, default="video_our_robot.mp4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=10)
    args = parser.parse_args()
    
    options = dict()
    options['model_path'] = 'dual_arms.xml'
    options['rot_scale'] = .3
    options['stiffness'] = 50.
    options['null_space_damping'] = 1
    options['site_name'] = 'ee_site_l'
    dual_arms_control = True
    path_planning = True
    save_video = True
    
    if dual_arms_control:
        options['controlled_joints'] = \
            [
                "LJ_1", "LJ_2", "LJ_3", "LJ_4", "LJ_5", "LJ_6",
                "RJ_1", "RJ_2", "RJ_3", "RJ_4", "RJ_5", "RJ_6"
            ]
    else:
        options['controlled_joints'] = \
            [
                "LJ_1", "LJ_2", "LJ_3", "LJ_4", "LJ_5", "LJ_6",
            ]
    
    # sim = create_sim(collision=collision)
    sim = create_sim(model_filename=options['model_path'], collision=False)
    controller = ImpedanceControllerV2(sim, **options)
    
    if save_video is True:
        viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=0)
        viewer.cam.trackbodyid = 0
        viewer.cam.distance = sim.model.stat.extent * 2.0
        viewer.cam.azimuth = 180
        viewer.cam.lookat[2] = 0.1
        viewer.cam.elevation = -15
    else:
        viewer = mujoco_py.MjViewer(sim)

    for i in range(1):
        # Set a random state to get a random feasible setpoint.
        # qpos = np.random.uniform(-1., 1, size=7)
        # qvel = np.zeros(7)
        # qpos = np.random.uniform(-1., 1, size=12)
        # qpos = np.zeros(12)
        qpos = -1 * np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        qvel = np.zeros(12)
        state = np.concatenate([qpos, qvel])
        
        sim_state = sim.get_state()
        sim_state.qpos[:] = qpos
        sim_state.qvel[:] = qvel
        sim.set_state(sim_state)
        sim.forward()
        
        pos_l, quat_l, pos_r, quat_r = controller.get_site_pose('ee_site_l', 'ee_site_r')
        print("pos_set_r :::", pos_r)
        print("pos_set_l :::", pos_l)
        
        # while True:
        #     viewer.render()
        
        # ==============================================
        #  plan manipulation trajectory based on DMP   #
        # ==============================================
        target_pos_r, target_orientation_r = pos_r + np.array([0.2, 0., 0.5]), quat_r
        start_pos_r, start_orientation_r = pos_r, quat_r
        target_pos_l, target_orientation_l = pos_l + np.array([0.2, 0., 0.5]), quat_l
        start_pos_l, start_orientation_l = pos_l, quat_l
        
        (
            position_planner_r,
            orientation_path_r,
            target_position_r,
            target_orientation_r,
        ) = generate_reference_path(
            start_pos_r, start_orientation_r,
            target_pos_r, target_orientation_r,
            n_timesteps=500
        )
        
        (
            position_planner_l,
            orientation_path_l,
            target_position_l,
            target_orientation_l,
        ) = generate_reference_path(
            start_pos_l, start_orientation_l,
            target_pos_l, target_orientation_l,
            n_timesteps=500
        )

        writer = imageio.get_writer(args.video_path, fps=10)
        # ============================================================
        #  Track manipulation trajectory with impedance controller   #
        # ============================================================
        for i in range(500):
            if dual_arms_control:
                if path_planning:
                    pos_r, vel_r = position_planner_r.next()
                    orient = orientation_path_r.next()
                    target_r = np.hstack([pos_r,
                                          transformations.quaternion_from_euler(orient[0],
                                                                                orient[1],
                                                                                orient[2],
                                                                                "rxyz")]
                                         )
                    
                    pos_l, vel_l = position_planner_l.next()
                    orient = orientation_path_l.next()
                    target_l = np.hstack([pos_l,
                                          transformations.quaternion_from_euler(orient[0],
                                                                                orient[1],
                                                                                orient[2],
                                                                                "rxyz")]
                                         )
                    
                    controller.set_dual_arms_target_pose(target_r, target_l)
                else:
                    controller.set_dual_arms_action(np.array([0.0, 0.0, 0.0, 0., 0., 0.]),
                                                    np.array([0.0, 0.0, 0.0, 0., 0., 0.]))
            else:
                controller.set_action(np.array([0.01, 0., 0., 0., 0., 0.]))
                print("pos_set :::", controller.pos_set)
            print("pos_set_r :::", controller.pos_set_r)
            print("pos_set_l :::", controller.pos_set_l)
            
            for j in range(10):
                if dual_arms_control:
                    torque_1 = controller.get_site_torque(site_name="ee_site_l", target_vel=None)[:6]
                    torque_2 = controller.get_site_torque(site_name="ee_site_r", target_vel=None)[6:]
                    torque = np.hstack([torque_1, torque_2])
                else:
                    torque = controller.get_torque()
                sim.data.ctrl[:] = torque
                sim.step()
        
            if save_video:
                viewer.render(width=1024, height=1024)
                img = viewer.read_pixels(width=1024, height=1024, depth=False)
    
                # if mode == 'rgb_array':
                #     self._get_viewer(mode).render(width, height)
                #     # window size used for old mujoco-py:
                #     data = self._get_viewer(mode).read_pixels(width, height, depth=False)
                # dump a frame from every K frames
                if i % args.skip_frame == 0:
                    frame = img[::-1, :, :]
                    # print("frame :::", frame)
                    writer.append_data(frame)
                    # print("Saving frame #{}".format(i))
                # render_frame(viewer, controller.pos_set, controller.quat_set)
                # print("pos_set :::", controller.pos_set)
            else:
                viewer.render()
            # viewer.render()


if __name__ == '__main__':
    # vis_impedance_fixed_setpoint(collision=False)
    sim = create_sim(collision=False)
    save_video = True
    if save_video is True:
        viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=0)
        viewer.cam.trackbodyid = 0
        viewer.cam.distance = sim.model.stat.extent * 2.0
        viewer.cam.azimuth = 180
        viewer.cam.lookat[2] = 0.1
        viewer.cam.elevation = -15
    else:
        viewer = mujoco_py.MjViewer(sim)
    sim.step()

    writer = imageio.get_writer("test.mp4", fps=10)
    for i in range(1000):
        if save_video:
            viewer.render(width=1024, height=1024)
            img = viewer.read_pixels(width=1024, height=1024, depth=False)
            skip_frame = 10
            if i % skip_frame == 0:
                frame = img[::-1, :, :]
                # print("frame :::", frame)
                writer.append_data(frame)
                # print("Saving frame #{}".format(i))
            # render_frame(viewer, controller.pos_set, controller.quat_set)
            # print("pos_set :::", controller.pos_set)
        else:
            viewer.render()
    
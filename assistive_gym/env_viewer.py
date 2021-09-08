import gym, sys, argparse
import numpy as np
from assistive_gym.learn import make_env
# import assistive_gym
import imageio
import time


if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()


def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()


def viewer(env_name):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)
    
    # Grab name of this rollout combo
    video_name = "{}-{}-{}".format(
        "env_test", "".join("jaco"), "controller_osc").replace("_", "-")

    # Calculate appropriate fps
    fps = int(10)
    
    # Define video writer
    video_writer = imageio.get_writer("{}.mp4".format(video_name), fps=fps)

    # while True:
    for i in range(5):
        done = False
        env.render()
        observation = env.reset()
        print(observation)
        action = sample_action(env, coop)
        print("action :", action)
        time.sleep(1)
        env.setup_camera(camera_width=1920//2, camera_height=1080//2)
        if coop:
            print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))
        else:
            print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))
        
        # while not done:
        #     observation, reward, done, info = env.step(sample_action(env, coop))
        #     img, _ = env.get_camera_image_depth()
        #     video_writer.append_data(img)
        #     env.render()
        #     if coop:
        #         done = done['__all__']
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assistive Gym Environment Viewer')
    parser.add_argument('--env',
                        # default='DressingSawyerHuman-v1',
                        default='ScratchItchJaco-v1',
                        help='Environment to test (default: ScratchItchJaco-v1)')
    
    args = parser.parse_args()
    
    viewer(args.env)

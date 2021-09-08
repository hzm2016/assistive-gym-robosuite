import gym, assistive_gym

env = gym.make('DrinkingSawyer-v1')
# env = gym.make('FeedingSawyer-v1')

env.render()
observation = env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    print("action :::", action)
    observation, reward, done, info = env.step(action)
    print("observation :::", observation)
    print("reward :::", reward)

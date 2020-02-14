import gym
env = gym.make("FrozenLake-v0",is_slippery=False)
actions=[2,2,1,1,1,2,1,2]
env.reset()
for i in range(len(actions)):

    env.render()
    env.step(actions[i]) # take a random action
env.close()
import gym
from agent import DQNAgent

max_frames=1000
batch_size=4

env = gym.make("MountainCar-v0")
agent = DQNAgent(env)
state = env.reset()


for frame in range(max_frames):
  env.render()
  action = agent.get_action(state)
  next_state, reward, done, _ = env.step(action)
  agent.replay_buffer.push(state,action,reward,next_state,done)

  if len(agent.replay_buffer) > batch_size:
    agent.update(batch_size)

  if done:
    state = env.reset()


env.close()
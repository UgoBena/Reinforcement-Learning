import torch
import numpy as np
from torch import nn
import time
import os

###Simple neural net
class SimpleNet(nn.Module):
	def __init__(self,input_dim,output_dim):
		super(SimpleNet,self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.fc = nn.Linear(self.input_dim,self.output_dim)

	def forward(self,state):
		return self.fc(state)

class NetAgent:
	def __init__(self,env,learning_rate=1e-3,gamma=0.99):
		self.env=env
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.model = SimpleNet(env.observation_space.n,env.action_space.n)

		self.optimizer=torch.optim.Adam(self.model.parameters())
		self.MSE_loss=nn.MSELoss()

	def get_action(self,state,eps=0.2):
		state_tensor = torch.zeros(self.env.observation_space.n)
		state_tensor[state] = 1
		qvals = self.model.forward(state_tensor)
		action = np.argmax(qvals.detach().numpy())

		if np.random.rand() < eps :
			return self.env.action_space.sample()

		return action

	def compute_loss(self,state,action,reward,next_state):
		state_tensor = torch.zeros(self.env.observation_space.n)
		state_tensor[state] = 1
		next_state_tensor = torch.zeros(self.env.observation_space.n)
		next_state_tensor[next_state] = 1

		curr_Q = self.model.forward(state_tensor)[action]
		next_Q = self.model.forward(next_state_tensor)
		max_next_Q = torch.max(next_Q).item()
		expected_Q = torch.FloatTensor([reward + self.gamma * max_next_Q])

		loss = self.MSE_loss(curr_Q,expected_Q)
		return loss

	def update(self,state,action,reward,next_state):
		loss = self.compute_loss(state,action,reward,next_state)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


class SimpleAgent:
	def __init__(self,env,learning_rate=0.81,gamma=0.96,epsilon=0.3):
		self.env=env
		self.gamma = gamma
		self.learning_rate=learning_rate
		self.Q = np.zeros((env.observation_space.n,env.action_space.n))
		self.epsilon=epsilon

	def get_action(self,state):
		if np.random.rand() < self.epsilon:
			return self.env.action_space.sample()
		return np.argmax(self.Q[state])

	def update(self,state,action,reward,next_state,done):
		if done:
			self.Q[state][action] = (1-self.learning_rate)*self.Q[state][action] + self.learning_rate*reward
		else:
			self.Q[state][action] = (1-self.learning_rate)*self.Q[state][action] + self.learning_rate *(
				reward + self.gamma * np.max(self.Q[next_state]))



import gym

env=gym.make("FrozenLake-v0",is_slippery=True)
agent=SimpleAgent(env)
max_episode_len = 100

state=env.reset()

attempt = 0
print("===============\n" + "Training Phase " + "\n===============\n")
agent.epsilon=0.5
while attempt<5000:
	for t in range(max_episode_len):
		env.render()
		action = agent.get_action(state)
		next_state, reward, done, _ = env.step(action)
		agent.update(state,action,reward,next_state,done)
		state = next_state
		if done:
			state=env.reset()
			attempt += 1
			break

print("===============\n" + "Testing Phase " + "\n===============\n")
agent.epsilon=0
attempt=0
total_reward = 0

while attempt<20:
	print("===============\nAttempt n° " + str(attempt) + "\n===============\n")
	print("Current Points :", total_reward)
	env.render()
	action = agent.get_action(state)
	next_state, reward, done, _ = env.step(action)
	state=next_state

	if reward==1:
		total_reward +=1

	if done:
		env.render()
		time.sleep(2)
		state=env.reset()
		attempt += 1
	time.sleep(0.3)
	os.system('clear')


env.close()

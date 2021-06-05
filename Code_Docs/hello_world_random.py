import gym
import minerl
import matplotlib.pyplot as plt

env = gym.make('MineRLNavigateDense-v0')

obs = env.reset()

done = False
timesteps = 0

# Lists for plotting
timeList = []
rewardList = []
compassAngleList = []

while not done:
	action = env.action_space.sample()
	obs, reward, done, _ = env.step(action)
	timeList.append(timesteps)
	rewardList.append(reward)
	compassAngleList.append(obs["compassAngle"])
	timesteps += 1

plt.subplot(1, 2, 1)
plt.plot(timeList, rewardList)
plt.xlabel('Time Step')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(timeList, compassAngleList)
plt.xlabel('Time Step')
plt.ylabel('Compass Angle')

plt.show()
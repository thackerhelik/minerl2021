import minerl
import gym
env = gym.make('MineRLNavigateDense-v0')


obs  = env.reset()
done = False
net_reward = 0
timesteps = 0

# Lists for plotting
timeList = []
rewardList = []
compassAngleList = []

while not done:
    action = env.action_space.noop()

    action['camera'] = [0, 0.03*obs["compassAngle"]]
    action['back'] = 0
    action['forward'] = 1
    action['jump'] = 1
    action['attack'] = 1

    obs, reward, done, info = env.step(action)

    timeList.append(timesteps)
    rewardList.append(reward)
    compassAngleList.append(obs["compassAngle"])
    timesteps += 1

    net_reward += reward

print("Total reward: ", net_reward)

plt.subplot(1, 2, 1)
plt.plot(timeList, rewardList)
plt.xlabel('Time Step')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(timeList, compassAngleList)
plt.xlabel('Time Step')
plt.ylabel('Compass Angle')

plt.show()
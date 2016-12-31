import gym
from gym.wrappers.monitoring import Monitor
from gym.monitoring.tests import helpers


with helpers.tempdir() as temp:

    env = gym.make('CartPole-v0')
    # 모니터 래핑
    env = Monitor(temp)(env)
    #env.monitor.start(temp)
    #env.monitor.start('/tmp/cartpole-experiment-1')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(observation,reward,done,info)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()
#env.monitor.close()
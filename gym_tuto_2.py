# https://gym.openai.com/docs 여기 실습을 따라 했습니다.
# env.step()정보 확

import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print("observation : ",observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print('| reward : ',reward,'| done :',done,'| info :',info,'|')
        if done:
            print("에피소드가 {} 타임셋 이후 종료됨".format(t+1))
            break
# https://gym.openai.com/docs 여기 실습을 따라 했습니다.
# 기본 작동 구조 실습
import gym
env = gym.make('CartPole-v0')
env.reset()
for i in range(1000):
    env.render()
    env.step(env.action_space.sample())
    print(i)


env.reset()
env = gym.make('SpaceInvaders-v0')
for i in range(1000):
    env.render()
    env.step(env.action_space.sample())
    print(i)
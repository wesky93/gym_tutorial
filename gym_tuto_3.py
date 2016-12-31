
import gym
env = gym.make('CartPole-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)
print(env.observation_space.high)
#> [  4.80000000e+00   3.40282347e+38   4.18879020e-01   3.40282347e+38]
print(env.observation_space.low)
#> [ -4.80000000e+00  -3.40282347e+38  -4.18879020e-01  -3.40282347e+38]



from gym import spaces
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8
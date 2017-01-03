import numpy as np
import tensorflow as tf
import math
from pprint import pprint

import gym

env = gym.make( 'CartPole-v0' )

env.reset( )
random_episodes = 0
reward_sum = 0

# print("랜덤 액션일경우 플레이 모습")
# while random_episodes < 10 :
#     env.render( )
#     observation, reward, done, _ = env.step( np.random.randint( 0, 2 ) )
#     reward_sum += reward
#     if done :
#         random_episodes += 1
#         print( "이번 에피소드의 보상은 {}".format( reward_sum ) )
#         reward_sum = 0
#         env.reset( )

# 가설
H = 5000  # 숨은 레이어의 뉴런수
batch_size = 4  # 한 에피소드당 몇개씩 학습을 진행할지 결정
learning_rate = 1e-2
gamma = 0.99  # 보상을 위해 행위자의 가치를 내

D = 4  # 입력할 차원

# 그래프 구성
tf.reset_default_graph( )

# 네트워크는 env의 관측자료에서 부터 시작되며 어떠한 행동을 취해야 할지에 대한 확률을 도출한다.
observations = tf.placeholder( tf.float32, [ None, D ], name="input_x" )
W1 = tf.get_variable( "W1", shape=[ D, H ], initializer=tf.contrib.layers.xavier_initializer( ) )
layer1 = tf.nn.relu( tf.matmul( observations, W1 ) )

W2 = tf.get_variable( "W2", shape=[ H, 1 ], initializer=tf.contrib.layers.xavier_initializer( ) )
score = tf.matmul( layer1, W2 )
probability = tf.nn.sigmoid( score )

# 학습하기 좋은 정책을 결정
tvars = tf.trainable_variables( )
input_y = tf.placeholder( tf.float32, [ None, 1 ], name="input_y" )
advantages = tf.placeholder( tf.float32, name="reward_signal" )

# 손실 함수
loglik = tf.log( input_y * (input_y - probability) + (1 - input_y) * (input_y + probability) )
loss = -tf.reduce_mean( loglik * advantages )
newGrads = tf.gradients( loss, tvars )

adam = tf.train.AdamOptimizer( learning_rate=learning_rate )
W1Grad = tf.placeholder( tf.float32, name="batch_grad1" )
W2Grad = tf.placeholder( tf.float32, name="batch_grad2" )
batchGrad = [ W1Grad, W2Grad ]
updateGrads = adam.apply_gradients( zip( batchGrad, tvars ) )


def discount_rewards( r ) :
    """
    보상점수가 담긴 1차원 배열을 받아 보상을 깎음
    :param r:
    :return:
    """
    discounted_r = np.zeros_like( r )
    runnig_add = 0
    for t in reversed( range( 0, r.size ) ) :
        runnig_add = runnig_add * gamma + r[ t ]
        discounted_r[ t ] = runnig_add
    return discounted_r


# 행위자와 환경을 실행
xs, hs, dlogps, drs, ys, tfps = [ ], [ ], [ ], [ ], [ ], [ ]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 200
init = tf.global_variables_initializer( )

# 그래프 실행
with tf.Session( )as sess :
    rendering = False
    sess.run(init)
    observation = env.reset()

    # 그래디언트를 초기화
    # 정책 네트워크를 업데이트 할수 있을때 까지 gradeBuffer을 이용하여 그래디언트를 수집하고 0으로 초기화 시킵니다
    gradBuffer = sess.run(tvars)
    pprint( "초기화 이전\n{}".format( gradBuffer ) )
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad*0

    pprint("초기화 이후\n{}".format(gradBuffer))

    print("학습 시작")
    while episode_number <= total_episodes:
        # 렌더링은 학습 속도를 늦추기 때문에 일정 점수이상 나와야 렌더링 하도록 함
        if reward_sum/batch_size > 400 or rendering == True:
            # env.render()
            rendering = True

        x = np.reshape(observation,[1,D])

        # 정책 네트워크를 돌려서 다음에 행해야할 행동을 알아냄
        tfporb = sess.run(probability,feed_dict={observations:x})
        action = 1 if np.random.uniform() < tfporb else 0

        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        # env에 다음 행동을 투입
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        # 보상을 기록함
        drs.append(reward)

        # 게임이 끝났을 경우 죽었을 경우
        if done:
            episode_number += 1

            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]

            discount_epr = discount_rewards(epr)
            discount_epr -= np.mean(discount_epr)
            discount_epr /= np.std(discount_epr)


            tGrad = sess.run(newGrads,feed_dict={observations: epx,input_y:epy,advantages:discount_epr})

            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0:
                sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],W2Grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad*0

                running_reward = reward_sum if running_reward is None else running_reward*0.99+reward_sum*0.01
                print("에피소드 {}번의 평균 보상은 {} 이며, 전체 보상의 평균은 {} 입니다".format(episode_number,reward_sum/batch_size,running_reward/batch_size))

                # 목표 점수에 도달시 학습 종료
                # if reward_sum/batch_size > 500:
                #     print("{}번쨰 에피소드에서 작업을 해결함".format(episode_number))
                #     break

                # 최대 점수를 구해보기

                reward_sum = 0
            observation = env.reset()

print(episode_number,"에피소드 진행 완료")
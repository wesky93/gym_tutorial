# https://github.com/awjuliani/DeepRL-Agents/blob/master/Simple-Policy.ipynb 튜토리얼을 따라했습니다.

import tensorflow as tf
import numpy as np

# 뽑기 기계에 대한 정의
bandits = [ 0.2, 0, -0.2, -5 ]
# 뽑기 기계 수량
num_bandits = len( bandits )


def pullBandit( bandit ) :
    # 임의의 숫자 뽑기
    result = np.random.randn( 1 )
    # 램덤한 숫자와 뽑기 기계의 숫자를 비교하여 보상을 줌
    if result > bandit :  # 뽑기기계가 졌을 경우 + 보상을 줌
        return 1
    else :  # 뽑기가 이겼을 때는 - 보상을 줌
        return -1


# 행위자에 대한 정의
tf.reset_default_graph( )

# 각 기계의 결과에 대한 가중치
weights = tf.Variable( tf.ones( [ num_bandits ] ) )
# 이전의 결과에서 가장 확률이 높은 기계를 선택
chosen_action = tf.argmax( weights, 0 )

reward_holder = tf.placeholder( shape=[ 1 ], dtype=tf.float32 )
action_holder = tf.placeholder( shape=[ 1 ], dtype=tf.int32 )
responsible_weight = tf.slice( weights, action_holder, [ 1 ] )
loss = -(tf.log( responsible_weight ) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer( learning_rate=0.001 )
update = optimizer.minimize( loss )

# 텐서플로우로 강화학습하기

total_ep = 1000  # 실행할 에피소드 횟수
total_reward = np.zeros( num_bandits )  # 각 뽑기기계의 전체 보상 초기화
e = 0.3  # 랜덤한 행동을 할 확률

init = tf.global_variables_initializer()

print("학습 시작")
with tf.Session( ) as sess :
    sess.run( init )
    i = 0
    while i < total_ep :
        if np.random.rand( 1 ) < e : # 랜덤으로 무작위 뽑기 기계를 실행한다
            action = np.random.randint( num_bandits )
        else :
            action = sess.run( chosen_action )

        reward = pullBandit( bandits[ action ] )

        # 게임 플레이 결과
        print('{count} : 뽑기기계 {action}를 실행한 결과 {reward}'.format(count=i+1,action=action,reward=reward))
        # 이를 이용한 강화학습
        _, resp, ww = sess.run( [ update, responsible_weight, weights ],
                                feed_dict={ reward_holder : [ reward ], action_holder : [ action ] } )
        total_reward[ action ] += reward
        if i % 50 == 0 :
            print( "{}대의 뽑기기계에 대한 보상결과 : {}".format( num_bandits, total_reward ) )
        i += 1
        print("그것으로 학습한 결과 {}".format(ww))
print("학습결과  {} 뽑기 기계가 가장 확률이 높은 뽑기라고 판단된다".format(np.argmax(ww)+1))

if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("그리고 그것은 정확하게 학습이 된것으로 보인다.")
else:
    print("하지만 실제로는 그렇지 않다")


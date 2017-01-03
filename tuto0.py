import gym
import gym.wrappers
import random
import tensorflow as tf
import numpy as np

# 기본적인 Q-테이블 학습

# 게임 환경 불러오기
env = gym.make('FrozenLake-v0')
env = gym.wrappers.Monitor(directory='monitor',force=True,video_callable=None)(env)

# Q-테이블 학습 알고리즘
tf.reset_default_graph()
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# 초기화
init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 10
#create lists to contain total rewards and steps per episode
stepList = []
rList = []

print("학습 시작")
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        title = "|{}번 에피소드 플레이|".format(i+1)
        bar = "{}".format('='*len(title))
        print("{bar}\n{title}\n{bar}".format(bar=bar,title=title))
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        step = 0
        #The Q-Network
        while step < 99:
            step+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            env.render()
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        stepList.append(step)
        rList.append(rAll)
print ("성공한 에피소드의 확률:{}".format(sum(rList)/num_episodes))
env.close()

#!/usr/bin/env python3
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
#from src.turtlebot3_dqn.turtlebot3_env import TurtleBot3Env

import gym
import math
import gym_turtlebot3
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gym import spaces
from gym.utils import seeding
#from gym_turtlebot3.envs.mytf import euler_from_quaternion
#from gym_turtlebot3.envs import Respawn
import tensorflow
from std_msgs.msg import Float32MultiArray
import keras
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import RMSprop
# from keras.optimizersv2 import rmsprop as RMSprop
from keras.layers import Dense, Dropout, Activation

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


time.sleep(4)
os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + 1)
rospy.init_node('TurtleBot3_Circuit_Simple-v0'.replace('-', '_') + "_w{}".format(1))
env = gym.make('TurtleBot3_Circuit_Simple-v0')
time.sleep(4)

observation = env.reset()


EPISODES = 2000

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath    = os.path.dirname(os.path.realpath(__file__))
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 1000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=50000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()
        
        if self.load_model:
            self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

            with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
   
    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer= RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model   
   	
    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())


    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(np.array(state, dtype=np.float64).reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in range(self.batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])
        
        states = np.array(states).reshape(self.batch_size, self.state_size)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states).reshape(self.batch_size, self.state_size)
        dones = np.array(dones)
        
        q_value = self.model.predict(states)
        self.q_value = q_value

        if target:
            next_target = self.target_model.predict(next_states)
 #           rospy.loginfo('If target')

        else:
            next_target = self.model.predict(next_states)
   #         rospy.loginfo('Else')

        next_q_value = []
        
        for i in range(self.batch_size):
            next_q_value.append(self.getQvalue(rewards[i], next_target[i], dones[i]))
        	
        next_q_value = np.array(next_q_value)
        
        '''
        X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
        Y_sample = q_value.copy()

        Y_sample[0][actions] = next_q_value
        Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

        if dones:
            X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
            Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)
		'''
        X_batch = states
        Y_batch = next_q_value
        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)
        
 #       rospy.loginfo('------Finish training---------')
if __name__ == '__main__':
#    rospy.init_node('turtlebot3_dqn_stage_env')	
   # time.sleep(5)
  #  os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + 1)
  #  rospy.init_node('TurtleBot3_Circuit_Simple-v0'.replace('-', '_') + "_w{}".format(1))
   # env = gym.make('TurtleBot3_Circuit_Simple-v0', observation_mode=0, continuous=False)
  #  time.sleep(5)

    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 26
    action_size = 5
    
    agent = ReinforceAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    
    det = {0: -1.5, 1: -0.75, 2: 0, 3: 0.75, 4: 1.5}

 #   observation = env.reset()

    for e in range(agent.load_episode + 1, EPISODES):
        done = False
    #    action = env.action_space.sample()
        state = env.reset()
 #       action = agent.getAction(state)
        score = 0
        
  #      observation, reward, done, info = env.step(action)
  
        for t in range(agent.episode_step):
        
            action = int(agent.getAction(state))
            # print(f'{action} and type action: {type(action)}')
            # print(det[action])

            next_state, reward, done, info = env.step(np.array([det[action], 0.15], dtype=np.float64))
            
  #          with open("rewardTB3_teste3.txt", 'a') as arq:
  #          	arq.write('\n ')
  #          	arq.write(str(reward))
  #          arq.close()
            
            if t >= 500:
                print("Time out!!")
                done = True
            
            agent.appendMemory(state, action, reward, next_state, done)
		
            if len(agent.memory) >= agent.train_start:
       #         rospy.loginfo('------Start training---------')
                if global_step <= agent.target_update:

                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            if e % 10 == 0:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)


            if done:
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                
                with open("scoreTB_teste4.txt", 'a') as arq:
                     arq.write('\n ')
                     arq.write(str(score))
                arq.close()

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d', e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")
                agent.updateTargetModel()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
                    
  #      if done:    
   #         observation = env.reset()
   #         print('chegou a meta')


 #   env.close()

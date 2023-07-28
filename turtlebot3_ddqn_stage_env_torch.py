#!/usr/bin/env python3

# Authors: Linda Moraes, adapted from Tien Tran and ROBOTIS 
# Tien Tran mail: quang.tran@fh-dortmund.de
# Linda Dotto mail: lindadotto@yahoo.com.br

import rospy
import os
import numpy as np
import time
import sys
import csv

from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32MultiArray
#from src.turtlebot3_dqn.environment_stage_thesis import Env

import gym
import math
import gym_turtlebot3
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gym import spaces
from gym.utils import seeding
from std_msgs.msg import Float32MultiArray

EPISODES = 3000

import torch as T

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

time.sleep(4)
os.environ['ROS_MASTER_URI'] = "http://localhost:{}/".format(11310 + 1)
rospy.init_node('TurtleBot3_Circuit_Simple-v0'.replace('-', '_') + "_w{}".format(1))
env = gym.make('TurtleBot3_Circuit_Simple-v0')
time.sleep(4)

observation = env.reset()

if not os.path.exists('logs'):
	os.makedirs('logs')
	
writer = SummaryWriter('logs')


class ReplayBuffer():
    def __init__(self, max_size, input_dims, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0 #keep track position of position of first unsave mem
        self.batch_size = batch_size
        #mem for states unpack input dims as a list *
        self.state_memory  = np.zeros((self.mem_size, *input_dims),
                                      dtype = np.float32)
        #mem for state trasition
        self.new_state_memory = np.zeros ((self.mem_size, *input_dims),
                                          dtype =np.float32)
        #mem for action   
        self.action_memory = np.zeros (self.mem_size, dtype =np.int32)
        #reward memory
        
        self.reward_memory = np.zeros (self.mem_size, dtype =np.float32)
        #mem for terminal
        self.terminal_memory = np.zeros (self.mem_size, dtype =bool)
    
    #add the trasition to the memory buffer
    def store_transition (self, state, action, reward, new_state , done):
        
        #check the unocupied mem position
        index = self.mem_cntr % self.mem_size
        
        #update the replay buffer
        self.state_memory[index]= state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory [index] = reward
        self.terminal_memory[index] = done  #want to multiply the reward by 0 (can also do it in learn function) Tien 1 -int (done)
        self.mem_cntr+=1 # increase the memor
        
    #function to sample from the memmory from the batch size    
    def sample_buffer (self, batch_size):
        #have we fill up the mem or not? 
        max_mem = min (self.mem_cntr, self.mem_size) #check if mem is full
        
        #select randomly from batch
        batch = np.random.choice(max_mem, batch_size, replace = False)#array of random. wont select again with replay False
        # book keeping for proper batch slicing
        batch_index = np.arange (self.batch_size, dtype=np.int32)
        
        
        #return the data in batch
        states = self.state_memory[batch]
        states_=  self.new_state_memory[batch]
        rewards =  self.reward_memory[batch]
        actions =  self.action_memory[batch] 
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, terminal, batch_index

class LinearDeepQNetwork(nn.Module):
    
    #constructor 
    def __init__(self, lr, n_actions, input_dims):
        # run the constructor of the parent class.
        super(LinearDeepQNetwork,self).__init__()
        
        # 3 steps:
        # 1: define the layers
        # 2: define the optimizer, and lose funtion: torch.optim, torch.nn
        # 3: define the training devices and send it to the device.: torch.device
        
        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear (256, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear (256, n_actions)
        #choose gradient decent method for backward propergation
        self.optimizer = optim.Adam(self.parameters(),lr = lr)
        self.loss = nn.MSELoss()
        
       # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        #send the network to the device
        self.to(self.device)
        
        
    #forward propergation: activation function
    def forward(self, state):
        
        layer1 = F.relu (self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        
        # Apply dropout
        drop_out = self.dropout(layer2)
        
        out_actions = self.fc3(drop_out)
        
        return out_actions

class ReinforceAgent():
    def __init__(self, state_size, action_size, writer):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.Path = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.Path.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/torch_model/stage_4_')
        self.resultPATH = self.Path.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/result/result.csv')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.observation_space = (self.state_size,)

        self.action_size = action_size
        self.action_space = np.arange(0,5,1) # for similar to gym (action index for scling)
        
        #for loging only, enter the day of run
        self.day = 1905
        
        self.writer = writer
        
        '''Hyperparameters'''
        
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0 # for exploration and exploitation dilema
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = self.batch_size # start learning as soon as fill up the batch size
        
        self.global_step = 0
        
        self.mem_size = 1000000
        self.memory = ReplayBuffer(self.mem_size, self.observation_space,self.batch_size) #replay memory
        
        self.model = LinearDeepQNetwork(self.learning_rate,  self.action_size, self.observation_space) #action-value function Q
        self.target_model = LinearDeepQNetwork(self.learning_rate,  self.action_size, self.observation_space) # target action-value funtion Q^
        
        print(self.model)

        self.updateTargetModel()
        #Tien: TODO: Load trained model'
        if self.load_model:
            self.epsilon = 0.3
            self.global_step = self.load_episode# dummy for bypass the batch sizes
            print ('Loading model at episode: ', self.load_episode)
            self.model = T.load(self.dirPath + str(self.load_episode) + '.pt')
            self.model.eval()
            print ("Load model state dict: ", self.model.state_dict())
            # TODO: Load previos epsilon self.epsilon = 0.99 
        
        
    #set weights of neurons to target model
    def updateTargetModel(self): 
        self.target_model.load_state_dict(self.model.state_dict())
        print ('Updated Target Model')
    
    def choose_action (self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor(observation,dtype= T.float).to(self.model.device) # turn to pytorch tensor           
            actions = self.model(state) # pass to dqn, this is q value, same as self.model.forward(state)
            
            action =T.argmax(actions).item() # Tien: item will take the tensor back to env: interger 
            #print ("Action chose:",  action)
        return action

    
    #using ReplayBuffer    
    def store_trasition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if (self.memory.mem_cntr < self.train_start): #only learn when mem have something /
            print ("------Not training---------")
            return
        
        self.model.optimizer.zero_grad() #0 the gradient b4 back propragation(only in pytorch)
        
        if self.global_step <= agent.target_update: 
            target = False
        else:
            target = True #using target model to predict actions
       
       
        states, actions, rewards, states_ ,terminals, batch_index= self.memory.sample_buffer(self.memory.batch_size)
        
        states = T.tensor(states, dtype= T.float).to(self.model.device) # turn np.array to pytorch tensor
        states_ = T.tensor(states_, dtype= T.float).to(self.model.device)        
        rewards = T.tensor(rewards).to(self.model.device) # tensor([batchsize])
        terminals = T.tensor(terminals, dtype = T.float).to(self.model.device)   

        
        '''Perform feedforward to compare: estimate value of current state (state) toward the max value of next state(states_)'''             
        #we want the delta between action the agent actually took and max action
        # batch index loop trhough all state
        
    #    q_prediction = self.model(states)[batch_index, actions] 
        q_prediction = self.model(states)
        actions = T.tensor(actions, dtype= T.int64).to(self.model.device) # dont need to be a tensor   
        
   #     q_s_a = q_prediction.gather(0, (T.from_numpy(actions)).unsqueeze(1))     
     #   q_s_a = q_s_a.squeeze()
        q_s_a = q_prediction.gather(1, actions.unsqueeze(1)).squeeze()
        q_tp1_values = self.model(states_).detach()
        _, a_prime = q_tp1_values.max(1)
        
        # get Q values from frozen network for next state and chosen action
	# Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
        q_target_tp1_values = self.target_model(states_).detach()
        q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()
        
	# if current state is end of episode, then there is no next Q value        
        q_target_s_a_prime = (1 - terminals) * q_target_s_a_prime
        
        q_target = rewards + self.discount_factor * q_target_s_a_prime
        
      #  error = rewards + self.discount_factor * q_target_s_a_prime - q_s_a      
     #   print(error)
        # clip the error and flip 
        loss = self.model.loss(q_s_a, q_target)

	
	# backwards pass
#	optimizer.zero_grad()
#	q_s_a.backward(clipped_error.data.unsqueeze(1))

        
    #    if target:
    #        q_next = self.target_model(states_)
    #    else:
    #        q_next = self.model(states_)
            
    #    q_next [terminals] = 0.0 # if done. next Q value  = reward

    #    q_target = rewards + self.discount_factor * T.max(q_next,dim =1) [0] # TD target: maximum along 64 rows dim=1, [0] only value not index
        
        #print (q_target)
   #     loss = self.model.loss(q_target, q_prediction).to(self.model.device) # is the TD error
        loss.backward() # back propagate
        self.model.optimizer.step() # update model weights
        self.writer.add_scalar("Loss/train", loss, self.global_step)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon =  self.epsilon_min
        
        

if __name__ == '__main__':
 #   rospy.init_node('turtlebot3_dqn_stage_thesis')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 26
    action_size = 5 #must be odd number
 #   rospy.logdebug("------Init ENV")
#    env = Env(action_size)

    det = {0: -1.5, 1: -0.75, 2: 0, 3: 0.75, 4: 1.5}


    agent = ReinforceAgent(state_size, action_size, writer)
    scores, episodes = [], []
    agent.global_step = 0
    start_time = time.time()

    for e in tqdm(range(agent.load_episode + 1, EPISODES)):
        done = False
        state = env.reset()
        score = 0

        for t in range(agent.episode_step):

            action = agent.choose_action(state)
            # next_state, reward, done, info = env.step(np.array([det[action], 0.15], dtype=np.float64))
            next_state, reward, done, info = env.step(action)
            agent.store_trasition(state, action, reward, next_state, done)
            #learn from sars'tupe
            score += reward
            agent.learn()
            state = next_state
            
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            if t >= 500: #take so long to reach goal: timeout 500 default
                print("Time out!!")
                done = True

            if done:
              
                if e % 100 == 0: #save model after each 100 ep.
                    T.save(agent.model.state_dict(), 'ddqn_st1_model.pth')
                    print ('Saved model at episode', e)                    
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                
                writer.add_scalar("Reward/train", reward, e)

      #          rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
      #                        e, score, agent.memory.mem_cntr, agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break

            agent.global_step += 1
            if agent.global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")
                agent.updateTargetModel()

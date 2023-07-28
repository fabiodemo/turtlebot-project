# DQN and DDQN for mapless navigation of a turtlebot3

In this work we present two Deep Reinforcement Learning (Deep-RL) approaches to enhance the problem of mapless navigation for terrestrial mobile robot. Our methodology focus on comparing a Deep-RL technique based on the Deep Q-Network (DQN) algorithm with a second one based on the Double Deep Q-Network (DDQN) algorithm. We use 24 laser range findings samples and the relative position and angle of the agent to the target as information for our agents, which provide the actions as velocities for our robot.


It's a turtlebot3 project based on Robotis repositories.

I'm using ROS Noetic with Gym gazebo in a docker container.

We're training DQN and DDQN models to compare performance of both on simulation, and after testing the models on lab ("real").

*In "models" there are some saved models that I got after training for 3000 or 5000 episodes.
# turtlebot-project

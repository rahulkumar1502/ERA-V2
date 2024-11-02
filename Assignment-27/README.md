# Car Simulation using T3D (Twin Delayed Deep Detereministic Policy Gradient)

## Objective
Implementation of Car Simulation using T3D algorithm

## [Demo](https://youtu.be/yMlShKtYFZ0)


## Working

- Here our RL agent (car) learns to navigate from point to point using T3D (Twin Delayed Deep Detereministic Policy Gradient) - a reinforcement learning technique.
- T3D here uses actor-critic models built on fully connected deep learning network.
- There are 2 actors - an actor model & actor target
- There are 4 critics - 2 critic models & 2 critic targets
- Actor network predicts Q states based on which 3 actions are taken by the agent : move forward, turn left or turn right.
- Car here navigates from 3 points.
- The red point is the starting point.
- We move from red point to green point followed by blue point and finally coming back to our starting red point
- This loop will continue and car will keep hopping from one point to another.
- During its travel car is guided by living penalties and rewards.
- Penalties are punishments given to car for an undesired motion.
- Rewards are incentives given when it takes a right action.
- Penalties in this car are for straying out of road, roaming too close to the wall (boundaries of video frame).
- Rewards given are for travelling through road and travelling towards the target set.
  
## Files Structure
- ai.py -> This has the deep learning network (T3D, actor & critic)
- map.py -> Main file from which ai.py is called and rewards/penalties are awarded
- car.kv -> Kivy file for setting up car
- images -> This has the images
    - citymap.png -> This is the map which is shown in display
    - MASK1.png -> This is the black and white map with black areas as lanes
    - mask.png -> Opengl needs different coordinates. This is the same MASK1.png transformed by rotating to the right by 90 degrees
- T3D_implementation.ipynb -> This is a reference colab file that is helpful to understand how T3D works using "AntBulletEnv-v0" environment in gym==0.22 & pybullet

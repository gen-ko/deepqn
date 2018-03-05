

# To train / test a DQN without experience replay, use script_v1.py

# with experience replay, use script_v2.py


# To train / test on different envs, modify the string pass to the initializer of EnvWrapper

# To use a linear network, set 'v1' in the initializer of DeepQN,
           triple-layer DQN, set 'v3'
           Conv-DQN,       set  'v5'



An easy version:

1. Want experience replay?
   Yes       -->   use script_v1.py
   No        -->   use script_v2.py


2. Want which env to be played?
   CartPole  -->  env = EnvWrapper('CartPole-v0')        -->  continue to step 3
   Car       -->  env = EnvWrapper('MountainCar-v0')     -->  continue to step 3
   SpaceInv  -->  env = EnvWrapper('SpaceInvaders-v0')   -->  continue to step 4

3. Want linear or non-linear DQN?
   Linear    -->   qn = DeepQN(..., type='v1', ...)
   NonLinear -->   qn = DeepQN(..., type='v3', ...)
   Dual NonL -->   qn = DeepQN(..., type='v5', ...)

4. Use Conv DQN
             -->   qn = DeepQN(..., type='v4', ...)
             
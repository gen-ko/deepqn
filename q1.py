import tensorflow as tf
import numpy as np
import gym
from model import *

class LinearAgent(Agent):
    def __init__(self, env_name, seed, lr=0.0001, epsilon=0.05, gamma=0.9, iter_num=20000):
        self.env = gym.make(env_name)
        self.env_name = env_name
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        # Setting the session to allow growth, so it doesn't allocate all GPU memory.
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        self.sess = tf.Session(config=config)

        self.num_actions = self.env.action_space.n

        # the dimension of a state observation
        self.state_dim = self.env.observation_space.shape[0]

        self.learning_rate = lr

        self.gamma = gamma

        self.eps = epsilon

        self.iterations = int(iter_num)

        self.save_path = "./tmp/model_q1_{}.ckpt".format(env_name)

        # instantiate the DQN
        self.dqn = LinearDQN(self.state_dim, self.num_actions, self.learning_rate)

        # initialize tf graph
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    def train(self):
        episode_num: int = 0
        cumulative_reward = 0.0
        avg_reward = 0.0
        cumulative_episode: int = 0

        state = self.env.reset()

        print("{} starts training...".format(self.env_name))
        for iter_i in range(self.iterations):

            q = self.sess.run(fetches=self.dqn.q_tensor,
                              feed_dict={self.dqn.state_ph: state})

            action = self.epsilon_greedy_policy(q)

            state_next, reward, is_terminal, _ = self.env.step(action)

            cumulative_reward = cumulative_reward + reward

            q_next = self.sess.run(fetches=self.dqn.q_tensor,
                                   feed_dict={self.dqn.state_ph: state_next})

            # optimal action
            q_next_max = max(max(q_next))

            target = self.gamma * q_next_max + reward

            self.sess.run(fetches=self.dqn.train_op,
                          feed_dict={self.dqn.target_ph: target,
                                     self.dqn.state_ph: state,
                                     self.dqn.action_ph: action})

            # prepare the next loop
            state = state_next

            if is_terminal:
                episode_num += 1
                state = self.env.reset()
                cumulative_episode += 1

                avg_reward += cumulative_reward
                cumulative_reward = 0.0

                if cumulative_episode % 10 == 9:
                    print('recent 10 episode average award: ', avg_reward / 10)
                    avg_reward = 0.0

            # loop_counter += 1
            if iter_i % 10000 == 9999:
                print('iterations passed: ', iter_i + 1)

        print('Training done')

        self.save_model()
    
    def test(self, render=False):
        tf.reset_default_graph()
        env = gym.make(self.env_name)
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        self.sess = tf.Session(config=config)


        test_episodes = 100

        # instantiate the DQN
        dqn = DQN_v1(self.state_dim, self.num_actions, self.learning_rate)

        # load weights
        self.load_model()

        avg_episode_reward = 0.0
        for epi in range(test_episodes):
            # initialize the state

            state = env.reset()

            # WARNING: Assuming batch_size = 1
            cumulative_reward = 0.0

            while True:

                # find the q values

                # q shape (batch_size, num_actions)
                if render:
                    env.render()

                q = self.sess.run(fetches=dqn.q_tensor,
                             feed_dict={dqn.state_ph: state})

                # action shape (batch_size, num_actions)

                # find the corresponding action
                action = self.greedy_policy(q)


                state_next, reward, is_terminal, info = env.step(action)
                cumulative_reward = cumulative_reward * self.gamma + reward


                # prepare the next loop
                state = state_next

                if is_terminal:
                    state = env.reset()
                    avg_episode_reward += cumulative_reward / test_episodes
                    cumulative_reward = np.float32(0)
                    break


        print('Testing done')
        print('Average discounted episode cumulative reward: ', avg_episode_reward)


def main():
    cartpole_agent = LinearAgent('CartPole-v0', 2022)
    mountaincar_agent = LinearAgent('MountainCar-v0', 2022)
    is_train = True
    is_test = False
    if is_train:
        cartpole_agent.train()
        mountaincar_agent.train()
    if is_test:
        cartpole_agent.test(render=True)
        mountaincar_agent.test(render=True)

if __name__ == '__main__':
    main()
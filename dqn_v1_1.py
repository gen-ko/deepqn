#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

class Agent(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0')


        # Setting the session to allow growth, so it doesn't allocate all GPU memory.
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        self.sess = tf.Session(config=config)

        self.num_actions = self.env.action_space.n

        # the dimension of a state observation
        self.state_dim = self.env.observation_space.shape[0]

        self.learning_rate = 0.0001

        self.gamma = 0.99

        self.eps = 0.05

        self.iterations: int = 1000000

        self.save_path = "./tmp/model_dqn_v1_1.ckpt"

        # instantiate the DQN
        self.dqn = DQN_v1(self.state_dim, self.num_actions, self.learning_rate)

        # initialize tf graph
        init = tf.global_variables_initializer()
        self.sess.run(init)
        return

    def epsilon_greedy_policy(self, q):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform(low=0.0, high=1.0) < self.eps:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(q)
        return action

    def greedy_policy(self, q):
        action = np.argmax(q)
        return action

    def train(self):

        episode_num: int = 0
        cumulative_reward = 0.0
        avg_reward = 0.0
        cumulative_episode: int = 0

        state = self.env.reset()

        for iter_i in range(self.iterations):

            q = self.sess.run(fetches=self.dqn.q_tensor,
                              feed_dict={self.dqn.state_ph: state})

            action = self.epsilon_greedy_policy(q)

            ######

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

        return

    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.save_path)
        print("Model saved in path: %s" % save_path)




class DQN_v1(object):
    def __init__(self, state_dim, num_actions, learning_rate=0.0001):
        self.state_ph = tf.placeholder(dtype=tf.float32,
                                       shape=state_dim,
                                       name='state_ph')

        self.tensor1 = tf.reshape(self.state_ph, shape=(1, state_dim))

        self.tensor2 = tf.layers.dense(inputs=self.tensor1,
                                        units=8,
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        kernel_initializer=tf.random_normal_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='t2',
                                        trainable=True,
                                        reuse=None)

        self.q_tensor = tf.layers.dense(inputs=self.tensor2,
                                        units=num_actions,
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=tf.random_normal_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='q',
                                        trainable=True,
                                        reuse=None)

        self.target_ph = tf.placeholder(dtype=tf.float32,
                                        shape=None,
                                        name='target_ph')

        self.action_ph = tf.placeholder(dtype=tf.int32,
                                        shape=None,
                                        name='action_ph')

        self.q_selected_tensor = tf.gather(self.q_tensor, self.action_ph, axis=1, name='q_selected')

        self.loss = tf.reduce_mean(tf.squared_difference(self.target_ph, self.q_selected_tensor), name='loss')

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.hard_train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        return





def test(render=False):
    tf.reset_default_graph()
    env = gym.make('CartPole-v0')
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    num_actions = env.action_space.n

    # the dimension of a state observation
    state_dim = env.observation_space.shape[0]

    batch_size = 1

    learning_rate = 0.0001

    stop_criteria = 0.0001

    gamma = 1

    test_episodes = 100

    # instantiate the DQN
    dqn = DQN_v1(state_dim, num_actions, learning_rate)

    # load weights
    saver = tf.train.Saver()

    saver.restore(sess, "./tmp/model_dqn_v1.ckpt")

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

            q = sess.run(fetches=dqn.q_tensor,
                         feed_dict={dqn.state_ph: state})

            # action shape (batch_size, num_actions)

            # find the corresponding action
            action = greedy_policy(q, num_actions, batch_size)


            state_next, reward, is_terminal, info = env.step(action)
            cumulative_reward = cumulative_reward * gamma + reward


            # prepare the next loop
            state = state_next

            if is_terminal:
                state = env.reset()
                state = np.array(state)
                state = state.reshape((batch_size, state_dim))
                avg_episode_reward += cumulative_reward / test_episodes
                cumulative_reward = np.float32(0)
                break


    print('Testing done')
    print('Average discounted episode cumulative reward: ', avg_episode_reward)
    return


def main():
    print(tf.__version__)
    agent = Agent()
    is_train = True
    is_test = False
    if is_train:
        agent.train()
    if is_test:
        test(render=True)
    return


if __name__ == '__main__':
    main()
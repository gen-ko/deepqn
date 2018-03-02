#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse





def epsilon_greedy_policy(q, num_actions):
    # Creating epsilon greedy probabilities to sample from.

    eps = 0.05

    if np.random.uniform(low=0.0, high=1.0) < eps:
        actions = np.random.randint(0, num_actions)
    else:
        actions = np.argmax(q)

    return actions


def greedy_policy(q, num_actions, batch_size=1):
    if batch_size == 1:
        # actions shape (1, )
        actions = np.argmax(q)
    else:
        # actions shape (batch_size, 1)
        actions = np.zeros((batch_size, ), dtype=int)
        for i in range(batch_size):
            actions[i] = np.argmax(q[i])
    return actions


class DQN_v1(object):
    def __init__(self, state_dim, num_actions, learning_rate=0.0001):
        self.state_ph = tf.placeholder(dtype=tf.float32,
                                       shape=state_dim,
                                       name='state_ph')

        self.tensor1 = tf.reshape(self.state_ph, shape=(1, state_dim))

        self.tensor2 = tf.layers.dense(inputs=self.tensor1,
                                       units=30,
                                       activation=tf.nn.relu,
                                       use_bias=True,
                                       kernel_initializer=tf.random_normal_initializer(),
                                       bias_initializer=tf.zeros_initializer(),
                                       trainable=True,
                                       reuse=None)

        self.q_tensor = tf.layers.dense(inputs=self.tensor2,
                                        units=num_actions,
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=tf.random_normal_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        trainable=True,
                                        reuse=None)

        self.q_flat_tensor = tf.reshape(self.q_tensor, shape=[num_actions])

        self.target_ph = tf.placeholder(dtype=tf.float32,
                                        shape=None,
                                        name='target_ph')

        self.action_ph = tf.placeholder(dtype=tf.int32,
                                        shape=None,
                                        name='action_ph')

        self.q_selected_tensor = tf.gather(self.q_flat_tensor, self.action_ph, axis=0, name='q_selected')

        self.loss = tf.reduce_mean(tf.squared_difference(self.target_ph, self.q_selected_tensor), name='loss')

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.hard_train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        return


def train():
    env = gym.make('CartPole-v0')

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    num_actions = env.action_space.n

    # the dimension of a state observation
    state_dim = env.observation_space.shape[0]

    batch_size = 1

    update_steps: int = 10

    learning_rate = 0.0001

    gamma = 0.99

    iterations: int = 20000

    # instantiate the DQN
    dqn = DQN_v1(state_dim, num_actions, learning_rate)

    state = env.reset()

    # initialize tf graph
    init = tf.global_variables_initializer()
    sess.run(init)

    episode_num: int = 0

    cumulative_reward = 0.0
    avg_reward = 0.0
    cumulative_episode: int= 0
    minor_step: int = 0


    for iter_i in range(iterations):


        q = sess.run(fetches=dqn.q_tensor,
                     feed_dict={dqn.state_ph: state})

        action = epsilon_greedy_policy(q, num_actions)

        ######

        state_next, reward, is_terminal, _ = env.step(action)

        cumulative_reward = cumulative_reward + reward


        q_next = sess.run(fetches=dqn.q_tensor,
                          feed_dict={dqn.state_ph: state_next})

        # optimal action
        q_next_max = max(max(q_next))

        target = gamma * q_next_max + reward

        sess.run(fetches=dqn.train_op,
                 feed_dict={dqn.target_ph: target,
                            dqn.state_ph: state,
                            dqn.action_ph: action})

        # prepare the next loop
        state = state_next

        if is_terminal:
            episode_num += 1
            state = env.reset()
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
    #print('discrepency: ', discrepency)

    # save model
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.

    save_path = saver.save(sess, "./tmp/model_dqn_v1_1.ckpt")
    print("Model saved in path: %s" % save_path)

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

    saver.restore(sess, "./tmp/model_dqn_v1_1.ckpt")

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
    tf.set_random_seed(2022)
    is_train = True
    is_test = False
    if is_train:
        train()
    if is_test:
        test(render=True)
    return


if __name__ == '__main__':
    main()
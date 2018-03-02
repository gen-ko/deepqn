import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

class Agent:
    def __init__(self):
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        self.sess = tf.Session(config=config)
        self.save_path = "./tmp/model_dqn_v1.ckpt"
        self.eps = 0.05
        self.num_actions = 0
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
        pass
    
    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.save_path)
        print("Model saved in path: %s" % save_path)

    def load_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.save_path)


    def test(self, render=False):
        pass

class LinearDQN:
    def __init__(self, state_dim, num_actions, learning_rate=0.0001):
        self.state_ph = tf.placeholder(dtype=tf.float32,
                                       shape=state_dim,
                                       name='state_ph')

        self.q_tensor = tf.layers.dense(inputs=tf.reshape(self.state_ph, shape=(1, state_dim)),
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
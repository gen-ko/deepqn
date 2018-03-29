import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

from dqn import DQN
from memory_replay import MemoryReplayerTF as MemoryReplayer


class Agent:
    def __init__(self, state_shape, num_actions, batch_size=64, gamma=1.0, net_type='linear', eps=0.5):

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.eps = eps
        self.gamma = gamma
        self.net_type = net_type

        self.state_ndim = len(self.state_shape)
        self.state_batch_shape = [self.batch_size]
        for i in range(self.state_ndim):
            self.state_batch_shape.append(self.state_shape[i])
        self.state_single_shape = self.state_batch_shape
        self.state_single_shape[0] = 1

        self.dqn = DQN(state_shape, num_actions, gamma=self.gamma, net_type=self.net_type)

        self.s0 = tf.placeholder(dtype=tf.float32,
                                 shape=self.state_shape,
                                 name='s0')

        self.s0_reshaped = tf.reshape(self.s0, shape=self.state_single_shape)

        self.q0 = self.dqn.core_graph(self.s0_reshaped)

        # train

        self.mr = MemoryReplayer(state_shape, capacity=50000, batch_size=batch_size)

        self.s, self.s_, self.r, self.a, self.done = self.mr.replay_register()

        self.loss = self.loss_graph(self.s, self.s_, self.r, self.a, self.done, self.gamma)

        self.train_op = self.set_train_op(self.loss)

        batch = tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32)

        random_action = tf.random_uniform(shape=[1], minval=0, maxval=self.num_actions, dtype=tf.int32)

        self.a_greedy = tf.argmax(self.q0, axis=1, output_type=tf.int32)

        self.a_epsilon_greedy = tf.where(batch < self.eps, random_action, self.a_greedy)

        return

    def loss_graph(self, s, s_, r, a, done, gamma=1.0):
        q = self.dqn.core_graph(s)
        q_ = self.dqn.core_graph(s_)
        q_ = tf.reduce_max(q_, axis=1)
        tmp1 = tf.shape(a)
        tmp2 = tmp1[0]
        tmp3 = tf.range(tmp2, dtype=tf.int32)
        tmp4 = [tmp3, a]
        tmp5 = tf.stack(tmp4, axis=1)
        estimate = tf.gather_nd(params=q, indices=tmp5)
        q_ = tf.where(done, tf.zeros(self.batch_size), q_)
        target = gamma * q_ + r
        loss = tf.reduce_mean(tf.squared_difference(target, estimate))
        return loss

    def set_train_op(self, loss, lr=0.0001, beta1=0.9, beta2=0.999):
        return tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2).minimize(loss)

    def epsilon_greedy_policy_run(self, sess, s0):
        return sess.run(self.a_epsilon_greedy, {self.s0: s0})

    def greedy_policy_run(self, sess, s0):
        return sess.run(self.a_greedy, {self.s0: s0})

    def train_run(self, sess):
        sess.run(self.train_op)
        pass

    def remember(self, sess, s, s_, r, a, done):
        return self.mr.remember(sess, s, s_, r, a, done)
    
    def save_model(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.save_path)
        print("Model saved in path: %s" % save_path)

    def load_model(self, sess):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.save_path)

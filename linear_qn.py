#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse


class LinearQN(object):
    def __init__(self, state_dim, num_actions, gamma=1.0):
        self.s = tf.placeholder(dtype=tf.float32,
                                 shape=[None, state_dim],
                                 name='s0')

        self.s_ = tf.placeholder(dtype=tf.float32,
                                 shape=[None, state_dim],
                                 name='s1')

        self.a = tf.placeholder(dtype=tf.int32,
                                shape=None,
                                name='a')

        self.r = tf.placeholder(dtype=tf.float32,
                                shape=[None],
                                name='r')

        self.q = tf.layers.dense(inputs=self.s,
                                 units=num_actions,
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='q',
                                 trainable=True,
                                 reuse=None)

        self.q_ = tf.layers.dense(inputs=self.s_,
                                 units=num_actions,
                                 name='q',
                                 trainable=False,
                                 reuse=True)

        # self.estimate = tf.gather(params=self.q, indices=self.a, axis=1)

        a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)

        self.estimate = tf.gather_nd(params=self.q, indices=a_indices)  # shape=(None, )

        # self.t1 = self.q[self.a]

        target = gamma * tf.reduce_max(self.q_, axis=1) + self.r

        self.target = tf.stop_gradient(target)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.estimate))
        return



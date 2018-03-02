#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse


class LinearQN(object):
    def __init__(self, state_dim, num_actions, gamma=1.0):
        self.s0 = tf.placeholder(dtype=tf.float32,
                                 shape=[None, state_dim],
                                 name='s0')

        self.s1 = tf.placeholder(dtype=tf.float32,
                                 shape=[None, state_dim],
                                 name='s1')

        self.a = tf.placeholder(dtype=tf.int32,
                                shape=[None, 1],
                                name='a')

        self.r = tf.placeholder(dtype=tf.float32,
                                shape=[None, 1],
                                name='r')

        self.q0 = tf.layers.dense(inputs=self.s0,
                                 units=num_actions,
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='q',
                                 trainable=True,
                                 reuse=None)

        self.q1 = tf.layers.dense(inputs=self.s1,
                                 units=num_actions,
                                 activation=None,
                                 use_bias=True,
                                 kernel_initializer=tf.random_normal_initializer(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='q',
                                 trainable=False,
                                 reuse=True)

        self.q0a = tf.gather(self.q0, self.a)

        self.q1a = tf.reduce_max(self.q1)

        self.target = tf.add(tf.scalar_mul(gamma, self.q1a), self.r)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.q0a))
        return



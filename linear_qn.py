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
                                shape=[None],
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

        self.estimate = tf.gather(self.q, self.a, axis=1)

        self.target = tf.add(tf.scalar_mul(tf.constant(gamma), tf.reduce_max(self.q_)), self.r)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.estimate))
        return



#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse


class DQN(object):
    def __init__(self, state_shape, num_actions, gamma=1.0, net_type='linear'):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.net_type = net_type
        return

    def core_graph(self, s):
        if self.net_type is None:
            raise ValueError('DQN must be specified with a type')
        h_last = None

        if self.net_type == 'linear':
            h_last = s

        if self.net_type == '3-layer':
            h1 = tf.layers.dense(inputs=s,
                                 units=24,
                                 activation=tf.nn.tanh,
                                 use_bias=True,
                                 kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='core_graph_h1',
                                 trainable=True,
                                 reuse=tf.AUTO_REUSE)

            h_last = tf.layers.dense(inputs=h1,
                                     units=48,
                                     activation=tf.nn.tanh,
                                     use_bias=True,
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                     bias_initializer=tf.zeros_initializer(),
                                     name='core_graph_h2',
                                     trainable=True,
                                     reuse=tf.AUTO_REUSE)

        if self.net_type == 'conv':
            s_trans = tf.transpose(s, [0, 2, 3, 1])
            h1 = tf.layers.conv2d(
                inputs=s_trans,
                filters=16,
                kernel_size=[8, 8],
                strides=(4, 4),
                padding="same",
                activation=tf.nn.relu,
                data_format='core_graph_channels_last',
                reuse=tf.AUTO_REUSE,
                name='h1')
            h2 = tf.layers.conv2d(
                inputs=h1,
                filters=32,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                reuse=tf.AUTO_REUSE,
                name='core_graph_h2')

            h3 = tf.contrib.layers.flatten(
                inputs=h2,
                outputs_collections=None
            )

            # dense layer automatically make the inputs flattened
            h_last = tf.layers.dense(
                inputs=h3,
                units=256,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                reuse=tf.AUTO_REUSE,
                name='core_graph_h_last'
            )

        # dualing network
        if self.net_type == 'v5':
            h1 = tf.layers.dense(inputs=s,
                                 units=24,
                                 activation=tf.nn.tanh,
                                 use_bias=True,
                                 kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='core_graph_h1',
                                 reuse=tf.AUTO_REUSE,
                                 trainable=True)

            h2_v = tf.layers.dense(inputs=h1,
                                   units=48,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='core_graph_h2_v',
                                   reuse=tf.AUTO_REUSE,
                                   trainable=True)

            h2_a = tf.layers.dense(inputs=h1,
                                   units=48,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='core_graph_h2_a',
                                   reuse=tf.AUTO_REUSE,
                                   trainable=True)

            h3_v = tf.layers.dense(inputs=h2_v,
                                   units=1,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='core_graph_h3_v',
                                   reuse=tf.AUTO_REUSE,
                                   trainable=True)

            h3_a = tf.layers.dense(inputs=h2_a,
                                   units=self.num_actions,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='core_graph_h3_a',
                                   reuse=tf.AUTO_REUSE,
                                   trainable=True)

            return h3_v + h3_a

        return tf.layers.dense(inputs=h_last,
                               units=self.num_actions,
                               activation=None,
                               use_bias=True,
                               kernel_initializer=tf.keras.initializers.glorot_uniform(),
                               bias_initializer=tf.zeros_initializer(),
                               trainable=True,
                               name='core_graph_q',
                               reuse=tf.AUTO_REUSE)


#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse


class DeepQN(object):
    def __init__(self, state_shape, num_actions, gamma=1.0, type='v3', scope='A'):
        self.sess = None
        self.state_shape = state_shape
        self.state_ndim = len(state_shape)
        self.num_actions = num_actions
        self.state_batch_shape = [None]

        for i in range(self.state_ndim):
            self.state_batch_shape.append(self.state_shape[i])

        with tf.name_scope(scope):
            self.s = tf.placeholder(dtype=tf.float32,
                                     shape=self.state_batch_shape,
                                     name='s')

            self.a = tf.placeholder(dtype=tf.int32,
                                    shape=None,
                                    name='a')

            self.r = tf.placeholder(dtype=tf.float32,
                                    shape=[None],
                                    name='r')

            if type == 'v1':
                self.h_last = self.s

            if type == 'v3':
                self.h1 = tf.layers.dense(inputs=self.s,
                                         units=24,
                                         activation=tf.nn.tanh,
                                         use_bias=True,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                         bias_initializer=tf.zeros_initializer(),
                                         name='h1',
                                         trainable=True,
                                         reuse=None)

                self.h_last = tf.layers.dense(inputs=self.h1,
                                         units=48,
                                         activation=tf.nn.tanh,
                                         use_bias=True,
                                         kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                         bias_initializer=tf.zeros_initializer(),
                                         name='h2',
                                         trainable=True,
                                         reuse=None)

            if type == 'v4':
                self.s_trans = tf.transpose(self.s, [0, 2, 3, 1])
                self.h1 = tf.layers.conv2d(
                    inputs=self.s_trans,
                    filters=16,
                    kernel_size=[8, 8],
                    strides=(4, 4),
                    padding="same",
                    activation=tf.nn.relu,
                    data_format='channels_last',
                    name='h1')
                self.h2 = tf.layers.conv2d(
                    inputs=self.h1,
                    filters=32,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    data_format='channels_last',
                    name='h2')

                self.h3 = tf.contrib.layers.flatten(
                    inputs=self.h2,
                    outputs_collections=None,
                )

                # dense layer automatically make the inputs flattened
                self.h_last = tf.layers.dense(
                    inputs=self.h3,
                    units=256,
                    activation=tf.nn.relu,
                    use_bias=True,
                    kernel_initializer=tf.keras.initializers.glorot_uniform(),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    trainable=True,
                    name='h3',
                    reuse=None
                )

            self.q = tf.layers.dense(inputs=self.h_last,
                                     units=num_actions,
                                     activation=None,
                                     use_bias=True,
                                     kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                     bias_initializer=tf.zeros_initializer(),
                                     name='q',
                                     trainable=True,
                                     reuse=None)

            # q_ is the selected q wrt a
            self.q_ = tf.placeholder(dtype=tf.float32,
                                     shape=[None],
                                     name='q_')

            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)

            self.estimate = tf.gather_nd(params=self.q, indices=a_indices)  # shape=(None, )

            target = gamma * self.q_ + self.r

            self.target = tf.stop_gradient(target)

            self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.estimate))

        return

    def reset_sess(self, sess):
        self.sess = sess

    def set_train(self, lr, beta1=0.9, beta2=0.999):
        self.train_op = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2).minimize(self.loss)

    def predict(self, state):
        if state.ndim == self.state_ndim:
            tmp_shape = self.state_batch_shape
            tmp_shape[0] = 1
            state = state.reshape(tmp_shape)
        return self.sess.run(self.q, {self.s: state})

    def predict_tf_register(self, state):
        # state is a tensor
        tf.assign(self.s, state)
        return

    def select_action_greedy(self, state):
        q = self.predict(state)
        return np.argmax(q, axis=1)

    def select_action_eps_greedy(self, eps, state):
        if np.random.uniform(low=0.0, high=1.0) < eps:
            return [np.random.randint(0, self.num_actions)]
        else:
            return self.select_action_greedy(state)

    def train(self, s, s_, r, a, done):
        q_ = self.predict(s_)
        qa_ = np.max(q_, axis=1)
        qa_ = np.where(done, 0, qa_)
        self.sess.run(self.train_op, {self.s: s,
                                      self.q_: qa_,
                                      self.r: r,
                                      self.a: a})
        return

    def train_tf_register(self, s, s_, r, a, done):
        # TODO

    def save(self, path='./tmp/dqn_v2.ckpt'):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("Model saved in path: %s" % save_path)
        return

    def load(self, path='./tmp/dqn_v2.ckpt'):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("Model loaded in path: %s" % path)
        return




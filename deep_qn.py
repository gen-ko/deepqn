#!/usr/bin/env python

import tensorflow as tf
import numpy as np


class DeepQN(object):
    def __init__(self, state_shape, num_actions, gamma=1.0, type='v3', scope='DQN'):
        self.sess = None
        self.type = type
        self.scope = scope
        self.state_shape = state_shape
        self.state_ndim = len(state_shape)
        self.num_actions = num_actions
        self.state_batch_shape = [None]
        self.state_batch_shape_valid = [1]

        for i in range(self.state_ndim):
            self.state_batch_shape.append(self.state_shape[i])

        for i in range(self.state_ndim):
            self.state_batch_shape_valid.append(self.state_shape[i])


        self.s = tf.placeholder(dtype=tf.float32,
                                shape=self.state_batch_shape,
                                name='s')

        self.a = tf.placeholder(dtype=tf.int32,
                                shape=None,
                                name='a')

        self.r = tf.placeholder(dtype=tf.float32,
                                shape=[None],
                                name='r')

        self.q = self.core_graph(self.s, type=type)
        self.q_ = tf.placeholder(dtype=tf.float32,
                                 shape=[None],
                                 name='q_')

        a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)

        self.estimate = tf.gather_nd(params=self.q, indices=a_indices)  # shape=(None, )

        target = gamma * self.q_ + self.r

        self.target = tf.stop_gradient(target)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, self.estimate))
        return

    def core_graph(self, s, type='v3'):
        h_last = None

        if type == 'v1':
            h_last = s

        if type == 'v3':
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

        if type == 'v4':
            s_trans = tf.transpose(s, [0, 2, 3, 1])
            h1 = tf.layers.conv2d(
                inputs=s_trans,
                filters=16,
                kernel_size=[8, 8],
                strides=(4, 4),
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
                name='h1')
            h2 = tf.layers.conv2d(
                inputs=h1,
                filters=32,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                data_format='channels_last',
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
                name='core_graph_h_last'
            )

        # dualing network
        if type == 'v5':
            h1 = tf.layers.dense(inputs=s,
                                 units=24,
                                 activation=tf.nn.tanh,
                                 use_bias=True,
                                 kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='core_graph_h1',
                                 trainable=True)

            h2_v = tf.layers.dense(inputs=h1,
                                   units=48,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='core_graph_h2_v',
                                   trainable=True)

            h2_a = tf.layers.dense(inputs=h1,
                                   units=48,
                                   activation=tf.nn.tanh,
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='core_graph_h2_a',
                                   trainable=True)

            h3_v = tf.layers.dense(inputs=h2_v,
                                   units=1,
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='core_graph_h3_v',
                                   trainable=True)

            h3_a = tf.layers.dense(inputs=h2_a,
                                   units=self.num_actions,
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='core_graph_h3_a',
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




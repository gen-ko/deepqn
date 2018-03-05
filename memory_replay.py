#!/usr/bin/env python

import numpy as np
import tensorflow as tf

TF_FLOAT_TYPE = tf.float32
TF_INT_TYPE = tf.int32

class MemoryReplayer(object):
    def __init__(self, state_shape, capacity: int=100000):
        self.capacity = capacity
        self.state_shape = state_shape
        self.state_ndim = len(self.state_shape)

        self.tensor_shape = [capacity]
        for i in range(self.state_ndim):
            self.tensor_shape.append(self.state_shape[i])

        self.s = np.zeros(shape=self.tensor_shape, dtype=np.float32)
        self.s_ = np.zeros(shape=self.tensor_shape, dtype=np.float32)
        self.r = np.zeros(shape=(capacity, ), dtype=np.float32)
        self.a = np.zeros(shape=(capacity, ), dtype=np.int32)
        self.done = np.zeros(shape=(capacity, ), dtype=np.bool)

        self.used_counter = 64
        self.mem_counter = 0
        return

    def remember(self, s, s_, r, a, done):

        self.s[self.mem_counter] = s
        self.s_[self.mem_counter] = s_
        self.r[self.mem_counter] = r
        self.a[self.mem_counter] = a
        self.done[self.mem_counter] = done
        self.mem_counter = (self.mem_counter + 1) % self.capacity
        if self.used_counter < self.capacity:
            self.used_counter += 1

    def replay(self, batch_size):
        batch_idx = np.random.randint(low=0, high=self.used_counter, size=min(batch_size, self.used_counter), dtype=np.int32)
        s = self.s[batch_idx]
        s_ = self.s_[batch_idx]
        r = self.r[batch_idx]
        a = self.a[batch_idx]
        done = self.done[batch_idx]
        return s, s_, r, a, done


class MemoryReplayerTF(object):
    def __init__(self, state_shape, capacity: int = 100000, batch_size=128):

        self.capacity = tf.constant(value=capacity, dtype=TF_INT_TYPE)
        self.batch_size = tf.constant(value=batch_size, dtype=TF_INT_TYPE)
        self.batch_size_scalar = batch_size
        self.state_shape = state_shape
        self.state_ndim = len(self.state_shape)

        self.tensor_shape = [capacity]
        for i in range(self.state_ndim):
            self.tensor_shape.append(self.state_shape[i])
        self.batch_shape = [batch_size]
        for i in range(self.state_ndim):
            self.batch_shape.append(self.state_shape[i])

        self.s = tf.Variable(initial_value=tf.zeros(shape=self.tensor_shape, dtype=TF_FLOAT_TYPE), trainable=False)
        self.s_ = tf.Variable(initial_value=tf.zeros(shape=self.tensor_shape, dtype=TF_FLOAT_TYPE), trainable=False)
        self.r = tf.Variable(initial_value=tf.zeros(shape=[self.capacity], dtype=TF_FLOAT_TYPE))
        self.a = tf.Variable(initial_value=tf.zeros(shape=[self.capacity], dtype=TF_INT_TYPE))
        self.done = tf.Variable(initial_value=tf.zeros(shape=[self.capacity], dtype=tf.bool))

        self.used_counter = tf.Variable(initial_value=0, dtype=TF_INT_TYPE)
        self.mem_counter = tf.Variable(initial_value=0, dtype=TF_INT_TYPE)

        self.s_old_ph = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=self.state_shape)
        self.s_new_ph = tf.placeholder(dtype=TF_FLOAT_TYPE, shape=self.state_shape)
        self.r_ph = tf.placeholder(dtype=TF_FLOAT_TYPE)
        self.a_ph = tf.placeholder(dtype=TF_INT_TYPE)
        self.done_ph = tf.placeholder(dtype=tf.bool)

        self.update_op1 = tf.assign(self.s[self.mem_counter], self.s_old_ph)
        self.update_op2 = tf.assign(self.s_[self.mem_counter], self.s_new_ph)
        self.update_op3 = tf.assign(self.r[self.mem_counter], self.r_ph)
        self.update_op4 = tf.assign(self.a[self.mem_counter], self.a_ph)
        self.update_op5 = tf.assign(self.done[self.mem_counter], self.done_ph)

        self.update_op6 = tf.assign(self.mem_counter, tf.mod(tf.assign_add(self.mem_counter, 1), self.capacity))
        self.update_op7 = tf.where(tf.less(self.used_counter, self.capacity), tf.assign_add(self.mem_counter, 1),
                                  tf.assign_add(self.mem_counter, 0))

        self.update_op = [self.update_op1, self.update_op2, self.update_op3,
                          self.update_op4, self.update_op5, self.update_op6, self.update_op7]
        return

    def remember(self, sess, s, s_, r, a, done):
        self.used_counter += 1
        return sess.run(self.update_op, {self.s_old_ph: s,
                                         self.s_new_ph: s_,
                                         self.r_ph: r,
                                         self.a_ph: a,
                                         self.done_ph: done})

    def replay_register(self):
        batch_idx = tf.random_uniform(shape=[self.batch_size], minval=0, maxval=self.capacity, dtype=TF_INT_TYPE)
        #s = self.s[batch_idx[0]]
        s = tf.gather(self.s, batch_idx)
        s_ = tf.gather(self.s_, batch_idx)
        r = tf.gather(self.r, batch_idx)
        a = tf.gather(self.a, batch_idx)
        done = tf.gather(self.done, batch_idx)
        self.replay_op = [s, s_, r, a, done]
        return s, s_, r, a, done


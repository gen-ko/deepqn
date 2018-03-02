#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse
from enum import Enum

class MemoryReplayer(object):
    def __init__(self, env_name: str ="CartPole-v0", cache_size: int=10000, eps=0.2, policy='random'):

        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.cache_size = cache_size
        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.eps = eps

        self.s0 = np.zeros(shape=(cache_size, self.state_dim), dtype=np.float32)
        self.s1 = np.zeros(shape=(cache_size, self.state_dim), dtype=np.float32)
        self.r  = np.zeros(shape=(cache_size, ), dtype=np.float32)
        self.a  = np.zeros(shape=(cache_size, ), dtype=np.int32)

        self.q_ = np.zeros(shape=(cache_size, ), dtype=np.float32)
        self.init_run()

        self.load_counter = 0
        return

    def init_run(self):
        self.env.reset()
        self.a = np.random.randint(self.num_actions, size=self.cache_size)
        self.q_ = np.random.standard_normal(self.cache_size)
        for i in range(self.cache_size):
            if self.env.env.state is None:
                self.env.reset()
            if self.env.env.steps_beyond_done is not None:
                self.env.reset()
            self.s0[i] = self.env.env.state
            self.s1[i], self.r[i], is_terminal, _ = self.env.step(self.a[i])
            self.r[i] -= 1.0
            if is_terminal:
                self.r[i] -= 1.0


        index = np.random.permutation(np.arange(0, self.cache_size))
        self.s0 = self.s0[index]
        self.s1 = self.s1[index]
        self.a = self.a[index]
        self.r = self.r[index]
        self.q_ = self.q_[index]
        self.load_counter = 0
        return

    def run(self, qn, sess):
        self.env.reset()
        self.a = np.random.randint(self.num_actions, size=self.cache_size)
        for i in range(self.cache_size):
            if self.env.env.state is None:
                self.env.reset()
            if self.env.env.steps_beyond_done is not None:
                self.env.reset()
            self.s0[i] = self.env.env.state
            q = sess.run(qn.q, {qn.s:[self.s0[i]]})

            if np.random.uniform(low=0.0, high=1.0) > self.eps:
                self.a[i] = np.argmax(q, axis=1)

            self.s1[i], self.r[i], is_terminal, _ = self.env.step(self.a[i])
            self.q_[i] = np.amax(sess.run(qn.q, {qn.s:[self.s1[i]]}), axis=1)
            self.r[i] -= 1.0
            if is_terminal:
                self.r[i] -= 1.0


        index = np.random.permutation(np.arange(0, self.cache_size))
        self.s0 = self.s0[index]
        self.s1 = self.s1[index]
        self.a = self.a[index]
        self.r = self.r[index]
        self.q_ = self.q_[index]
        self.load_counter = 0
        return

    def get_batch(self, qn, sess, size=32):
        a = self.load_counter
        b = self.load_counter + size
        self.load_counter += size
        if b > self.cache_size:
            self.run(qn, sess)
            print('load counter reset')
            a = self.load_counter
            b = self.load_counter + size
        return self.s0[a:b,:], self.s1[a:b,:], self.r[a:b], self.a[a:b], self.q_[a:b]






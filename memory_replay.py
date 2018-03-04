#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse
from enum import Enum


class MemoryReplayer(object):
    def __init__(self, env, cache_size: int=100000):
        self.cache_size = cache_size
        self.num_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        self.state_shape = env.observation_space.shape
        self.state_ndim = len(self.state_shape)

        tmp_shape = [cache_size]
        for i in range(self.state_ndim):
            tmp_shape.append(self.state_shape[i])

        self.s = np.zeros(shape=tmp_shape, dtype=np.float32)
        self.s_ = np.zeros(shape=tmp_shape, dtype=np.float32)
        self.r  = np.zeros(shape=(cache_size, ), dtype=np.float32)
        self.a  = np.zeros(shape=(cache_size, ), dtype=np.int32)
        self.done = np.zeros(shape=(cache_size, ), dtype=np.bool)

        self.used_counter = 0
        self.mem_counter = 0
        return

    def remember(self, s, s_, r, a, done):

        self.s[self.mem_counter] = s
        self.s_[self.mem_counter] = s_
        self.r[self.mem_counter] = r
        self.a[self.mem_counter] = a
        self.done[self.mem_counter] = done
        self.mem_counter = (self.mem_counter + 1) % self.cache_size
        if self.used_counter < self.cache_size:
            self.used_counter += 1

    def replay(self, batch_size):
        batch_idx = np.random.randint(low=0, high=self.used_counter, size=min(batch_size, self.used_counter), dtype=np.int32)
        avg = np.mean(self.r)
        # self.r = self.r * 0.995 + avg * 0.005
        s = self.s[batch_idx]
        s_ = self.s_[batch_idx]
        r = self.r[batch_idx]
        a = self.a[batch_idx]
        done = self.done[batch_idx]
        return s, s_, r, a, done



#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse
from enum import Enum

class MemoryReplayer(object):
    def __init__(self, env_name: str ="CartPole-v0", cache_size: int=10000, eps=0.2, gamma=0.99):

        self.env_name = env_name
        self.env = gym.make(self.env_name)

        self.cache_size = cache_size
        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.eps = eps
        self.gamma = gamma
        self.env.close()

        self.s = np.zeros(shape=(cache_size, self.state_dim), dtype=np.float32)
        self.s_ = np.zeros(shape=(cache_size, self.state_dim), dtype=np.float32)
        self.r  = np.zeros(shape=(cache_size, ), dtype=np.float32)
        self.a  = np.zeros(shape=(cache_size, ), dtype=np.int32)
        self.done = np.zeros(shape=(cache_size, ), dtype=np.bool)

        self.used_counter = 0
        self.mem_counter = 0
        return

    def remember(self, s, s_, r, a, done):

        self.s[self.mem_counter] = s
        self.s[self.mem_counter] = s_
        self.r[self.mem_counter] = r
        self.a[self.mem_counter] = a
        self.done[self.mem_counter] = done
        self.mem_counter = (self.mem_counter + 1) % self.cache_size
        if self.used_counter < self.cache_size:
            self.used_counter += 1

    def replay(self, batch_size):
        batch_idx = np.random.randint(low=0, high=self.used_counter, size=batch_size, dtype=np.int32)
        s = self.s[batch_idx]
        s_ = self.s_[batch_idx]
        r = self.r[batch_idx]
        a = self.a[batch_idx]
        done = self.done[batch_idx]
        return s, s_, r, a, done



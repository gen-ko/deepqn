#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse
from enum import Enum

class MemoryReplayer(object):
    def __init__(self, env_name: str ="CartPole-v0", cache_size: int=10000, policy='random'):

        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.cache_size = cache_size
        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]

        self.s0 = np.zeros(shape=(cache_size, self.state_dim), dtype=np.float32)
        self.s1 = np.zeros(shape=(cache_size, self.state_dim), dtype=np.float32)
        self.r  = np.zeros(shape=(cache_size, ), dtype=np.float32)
        self.a  = np.zeros(shape=(cache_size, ), dtype=np.int32)
        self.run()

        self.load_counter = 0
        return

    def run(self):
        self.env.reset()
        self.a = np.random.randint(self.num_actions, size=self.cache_size)
        for i in range(self.cache_size):
            if self.env.env.state is None:
                self.env.reset()
            if self.env.env.steps_beyond_done is not None:
                self.env.reset()
            self.s0[i] = self.env.env.state
            self.s1[i], self.r[i], _, _ = self.env.step(self.a[i])

        index = np.random.permutation(np.arange(0, self.cache_size))
        self.s0 = self.s0[index]
        self.s1 = self.s1[index]
        self.a = self.a[index]
        self.r = self.r[index]
        self.load_counter = 0
        return


    def get_batch(self, size=32):
        a = self.load_counter
        b = self.load_counter + size
        if b > self.cache_size:
            self.run()
            print('load counter reset')
            a = self.load_counter
            b = self.load_counter + size
        return self.s0[a:b,:], self.s1[a:b,:], self.r[a:b], self.a[a:b]

    def load_memory(self):
        return self.s0, self.s1, self.r, self.a





#!/usr/bin/env python

import numpy as np

class MemoryReplayer(object):
    def __init__(self, state_shape, capacity: int=100000):
        self.capacity = capacity
        self.state_shape = state_shape
        self.state_ndim = len(self.state_shape)

        tmp_shape = [capacity]
        for i in range(self.state_ndim):
            tmp_shape.append(self.state_shape[i])

        self.s = np.zeros(shape=tmp_shape, dtype=np.float32)
        self.s_ = np.zeros(shape=tmp_shape, dtype=np.float32)
        self.r = np.zeros(shape=(capacity, ), dtype=np.float32)
        self.a = np.zeros(shape=(capacity, ), dtype=np.int32)
        self.done = np.zeros(shape=(capacity, ), dtype=np.bool)

        self.used_counter = 0
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



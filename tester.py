#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

class Tester(object):
    def __init__(self, qn, env_name: str="CartPole-v0", episodes: int=100, report_interval: int=10):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.qn = qn
        self.report_interval = report_interval

    def run(self, sess):
        report_counter: int = 0
        r_sum_avg = 0.0
        for epi in range(100):
            s = self.env.reset()
            is_terminal = False
            r_sum = 0.0

            while not is_terminal:
                q = sess.run(self.qn.q, feed_dict={self.qn.s: [s]})
                a = np.argmax(q)
                s_, r, is_terminal, _ = self.env.step(a)
                r_sum += r
                s = s_

                if is_terminal:
                    r_sum_avg += r_sum / self.report_interval
                    report_counter += 1
                    if report_counter % self.report_interval == 0:
                        print('Total reward avg: ', r_sum_avg)
                        r_sum_avg = 0.0
                        report_counter = 0
        return





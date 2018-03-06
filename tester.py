#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

class Tester(object):
    def __init__(self, qn, env, episodes: int=100, report_interval: int=10):
        self.env = env
        self.qn = qn
        self.report_interval = report_interval
        self.episodes = episodes
        return

    def run(self, qn, sess, render=False):
        report_counter: int = 0
        r_sum_avg = 0.0
        return_value = 0.0
        r_per_epi = []
        for epi in range(self.episodes):
            s = self.env.reset()
            if render:
                self.env.render()
            is_terminal = False
            r_sum = 0.0

            while not is_terminal:
                q = sess.run(qn.q, feed_dict={qn.s: [s]})
                a = np.argmax(q)
                s_, r, is_terminal, _ = self.env.step(a)
                if render:
                    self.env.render()
                r_sum += r
                s = s_

                if is_terminal:
                    r_per_epi.append(r)
                    r_sum_avg += r_sum / self.report_interval
                    report_counter += 1
                    if report_counter % self.report_interval == 0:
                        print('Test reward avg: ', r_sum_avg)
                        return_value = r_sum_avg
                        r_sum_avg = 0.0
                        report_counter = 0
        return return_value, r_per_epi


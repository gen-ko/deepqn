#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

from memory_replay import MemoryReplayer
from linear_qn import LinearQN
from deep_qn import DeepQN
from tester import Tester



class EnvWrapper(object):
    def __init__(self, env, frame_stack=1):
        self.env = env
        self.action_space = self.ActionSpace(env.action_space.n)
        self.observation_space = self.ObservationSpace((env.observation_space.shape[0]*frame_stack,))
        self.state_dim = self.observation_space.shape[0]
        self.si = []
        self.frame_stack = frame_stack

        self.min_location = 0.0
        self.max_location = 0.0

        self.si = []
        self.si.append(self.env.reset())

        self.min_location = self.si[-1][0] - abs(self.si[-1][0])
        self.max_location = self.si[-1][0] + abs(self.si[-1][0])


        return

    def step(self, a):
        si, r, done, info = self.env.step(a)
        self.si.pop(0)
        self.si.append(si)

        s = np.array(self.si).reshape(self.state_dim)

        new_min_location = self.si[-1][0]
        new_max_location = self.si[-1][0]

        x = si[0]

        v = si[1]

        h = np.sin(3 * x) * 0.45 + 0.55

        v_update = np.cos(3 * x) * (-0.025)

        r_update = (v - self.v - v_update) * np.sign(v) * 1000

        if abs(v) >= 0.07:
            r_update = 1

        if h > 0.55:
            r += r_update * 2

        else:
            r += r_update

        self.v = v

        if x >= 0.5:
            r += 1000


        return s, r, done, info

    def reset(self):
        self.si = []

        si = self.env.reset()
        self.si.append(si)

        self.v = si[1]


        for fi in range(1, self.frame_stack):
            si, _, _, _ = self.env.step(0)
            self.si.append(si)
        s = np.array(self.si).reshape(self.state_dim)
        return s

    class ActionSpace(object):
        def __init__(self, n):
            self._n = n
            return

        @property
        def n(self):
            return self._n

    class ObservationSpace(object):
        def __init__(self, shape):
            self._shape = shape
            return

        @property
        def shape(self):
            return self._shape



def main():
    print(tf.__version__)


    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    envi = gym.make('MountainCar-v0')

    env = EnvWrapper(envi, frame_stack=1)

    env2 = gym.make('MountainCar-v0')

    mr = MemoryReplayer(env, cache_size=100000)

    qn = DeepQN(state_dim=mr.state_dim, num_actions=mr.num_actions, gamma=0.99)

    qn.reset_sess(sess)

    qn.set_train(0.001)

    init = tf.global_variables_initializer()
    sess.run(init)

    testor = Tester(qn, env2, report_interval=100, episodes=100)

    #print('Pretrain test:')
    #testor.run(qn, sess)

    score = []
    score2 = []

    for epi in range(1000000):

        s = env.reset()

        done = False

        rc = 0

        while not done:
            a = qn.select_action_eps_greedy(get_eps(epi), s)

            a_ = a[0]

            s_, r, done, _ = env.step(a_)

            mr.remember(s, s_, r, a_, done)

            s = s_

            rc += r

        score.append(rc)

        # replay

        s, s_, r, a, done = mr.replay(batch_size=64)

        qn.train(s, s_, r, a, done)

        if (epi + 1) % 200 == 0:
            avg_score = np.mean(score)
            print('avg score last 200 episodes ', avg_score)
            score = []

            if testor.run(qn, sess, render=False) > -110.0:
                qn.save('./tmp/dqn_v3.ckpt')
            
    return


def get_eps(t):
    return max(0.01, 1.0 - np.log10(t + 1) * 0.995)

if __name__ == '__main__':
    main()
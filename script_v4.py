#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse


from memory_replay import MemoryReplayer
from deep_qn import DeepQN
from tester import Tester

from env_wrapper import EnvWrapper

def train():
    print(tf.__version__)


    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    env = EnvWrapper('SpaceInvaders-v0')


    mr = MemoryReplayer(state_shape=env.state_shape, capacity=1000)

    qn = DeepQN(state_shape=env.state_shape, num_actions=env.num_actions, gamma=0.99, type='v4')

    qn.reset_sess(sess)

    qn.set_train(0.001)

    init = tf.global_variables_initializer()
    sess.run(init)

    testor = Tester(qn, env, report_interval=1, episodes=1)

    score = []

    for epi in range(1000):

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

        s, s_, r, a, done = mr.replay(batch_size=32)

        qn.train(s, s_, r, a, done)

        if (epi + 1) % 20 == 0:
            avg_score = np.mean(score)
            print('avg score last 20 episodes ', avg_score)
            score = []

            if testor.run(qn, sess, render=False) > 5000:
                qn.save('./tmp/dqn_v4.ckpt')
                break

    return

def test(render=False, path='./tmp/dqn_v3.ckpt', episodes=100):
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    env = EnvWrapper('SpaceInvaders-v0')

    qn = DeepQN(state_shape=env.state_shape, num_actions=env.num_actions, gamma=0.99)

    qn.reset_sess(sess)

    qn.load(path)

    env = gym.make('MountainCar-v0')

    testor = Tester(qn, env, report_interval=100, episodes=100)

    testor.run(qn, sess, render=render)

    return


def main():
    is_train = True
    is_test = False

    if is_train:
        train()

    if is_test:
        test(render=True)



def get_eps(t):
    return max(0.01, 1.0 - np.log10(t + 1) * 0.995)

if __name__ == '__main__':
    main()
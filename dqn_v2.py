#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

from memory_replay import MemoryReplayer
from linear_qn import LinearQN
from deep_qn import DeepQN
from tester import Tester







def main():
    print(tf.__version__)
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)


    mr = MemoryReplayer(cache_size=50000, eps=0.5, gamma=0.95)

    qn = DeepQN(state_dim=mr.state_dim, num_actions=mr.num_actions, gamma=0.99)

    qn.reset_sess(sess)

    qn.set_train(0.001)

    init = tf.global_variables_initializer()
    sess.run(init)

    testor = Tester(qn, report_interval=100)

    print('Pretrain test:')
    testor.run(qn, sess)


    env = gym.make('CartPole-v0')

    eps = 0.2

    score = []

    for epi in range(1000000):
        s = env.reset()

        done = False

        rc = 0

        while not done:
            a = qn.select_action_eps_greedy(eps, s)

            a_ = a[0]

            s_, r, done, _ = env.step(a_)

            mr.remember(s, s_, r, a_, done)

            s = s_

            rc += r

        score.append(rc)

        # replay

        s, s_, r, a, done = mr.replay(batch_size=32)

        qn.train(s, s_, r, a, done)

        if (epi + 1) % 1000 == 0:
            print('avg score last 1000 episodes ', np.mean(score))
            score = []

    return





if __name__ == '__main__':
    main()
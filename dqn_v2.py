#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

from memory_replay import MemoryReplayer
from linear_qn import LinearQN
from tester import Tester







def main():
    print(tf.__version__)

    mr = MemoryReplayer(cache_size=30000)
    qn = LinearQN(state_dim=mr.state_dim, num_actions=mr.num_actions, gamma=0.99)

    learning_rate = 0.0001

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(qn.loss)


    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)


    init = tf.global_variables_initializer()
    sess.run(init)

    testor = Tester(qn, report_interval=50)

    print('Pretrain test:')
    testor.run(qn, sess)

    for i in range(100):
        mr.run_env()

        sess.run(train_op, feed_dict={qn.s: mr.s0, qn.s_: mr.s1, qn.r: mr.r, qn.a: mr.a})

        print('update round: ', i + 1)
        testor.run(qn, sess)
    return





if __name__ == '__main__':
    main()
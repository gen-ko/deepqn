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

    mr = MemoryReplayer(cache_size=50000)
    qn = DeepQN(state_dim=mr.state_dim, num_actions=mr.num_actions, gamma=0.99, hidden_units=20)

    learning_rate = 0.0001

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(qn.loss)


    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)


    init = tf.global_variables_initializer()
    sess.run(init)

    testor = Tester(qn, report_interval=100)

    print('Pretrain test:')
    testor.run(qn, sess)




    for i in range(1000):
        s, s_, r, a = mr.get_batch(size=64)

        sess.run(train_op, feed_dict={qn.s: s, qn.s_: s_, qn.r: r, qn.a: a})

        t1 = sess.run(qn.q, {qn.s: mr.s0})
        t2 = sess.run(qn.q_, {qn.s_: mr.s0})


        with tf.variable_scope('q', reuse=True):
            w = tf.get_variable('kernel')

        w_value = sess.run(w)

        print('update round: ', i + 1)
        testor.run(qn, sess)
    return





if __name__ == '__main__':
    main()
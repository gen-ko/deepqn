#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

from memory_replay import MemoryReplayer
from deep_qn import DeepQN
from tester import Tester
from plotter import Plotter

from env_wrapper import EnvWrapper

def record(qn, sess, env):
    test = Tester(qn, env, 20, 20)
    return test.run(qn, sess)


def main():
    print(tf.__version__)
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops, log_device_placement=False)
    sess = tf.Session(config=config)

    env = EnvWrapper('CartPole-v0')
    env_test = EnvWrapper('CartPole-v0')

    mr = MemoryReplayer(env.state_shape, capacity=100000, enabled=False)

    # set type='v1' for linear model, 'v3' for three layer model (two tanh activations)

    # type='v5' use dual

    qn = DeepQN(state_shape=env.state_shape, num_actions=env.num_actions, gamma=0.99, type='v1')

    qn.reset_sess(sess)

    qn.set_train(0.001)

    init = tf.global_variables_initializer()
    sess.run(init)

    plotter = Plotter()

    testor = Tester(qn, env, report_interval=100)

    print('Pretrain test:')
    testor.run(qn, sess)

    score = []
    reward_record = []
    cnt_iter = 0

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
            cnt_iter += 1
            if (cnt_iter + 1) % 10000 == 0:
                reward_record.append(record(qn, sess, env_test))

        score.append(rc)

        # replay

        s, s_, r, a, done = mr.replay(batch_size=64)

        qn.train(s, s_, r, a, done)

        if cnt_iter > 1000000:
            break

        # if (epi + 1) % 200 == 0:
        #     avg_score = np.mean(score)
        #     plotter.plot(avg_score)
        #     print('avg score last 200 episodes ', avg_score)
        #     score = []
        #     if avg_score > 195:
        #         break

    f = open('CartPole-v0_q1_data.log', 'w')
    f.write(str(reward_record))
    f.close()
    return


def get_eps(t):
    return max(0.01, 1.0 - np.log10(t + 1) * 0.995)


if __name__ == '__main__':
    main()
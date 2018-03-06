#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse


from agent import Agent

from env_wrapper import EnvWrapper

from plotter import Plotter

def main():




    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops, log_device_placement=False)
    sess = tf.Session(config=config)

    env = EnvWrapper('CartPole-v0')
    agent = Agent(state_shape=env.state_shape, num_actions=env.num_actions, batch_size=64,
                  gamma=0.99, net_type='linear')

    init = tf.global_variables_initializer()
    sess.run(init)

    plotter = Plotter()
    score = []
    for epi in range(1000000):
        s = env.reset()

        done = False

        rc = 0

        while not done:
            a = agent.epsilon_greedy_policy_run(sess, s)

            a_ = a[0]

            s_, r, done, _ = env.step(a_)

            agent.remember(sess, s, s_, r, a_, done)

            s = s_

            rc += r

        score.append(rc)

        agent.train_run(sess)

        if (epi + 1) % 200 == 0:
            avg_score = np.mean(score)
            plotter.plot(avg_score)
            print('avg score last 200 episodes ', avg_score)
            score = []
            if avg_score > 195:
                break

if __name__ == '__main__':
    main(sys.argv)
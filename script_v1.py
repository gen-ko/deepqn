#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

from memory_replay import MemoryReplayer
from deep_qn import DeepQN
from tester import Tester
from plotter import Plotter

from collections import deque

from env_wrapper import EnvWrapper


def train(args=None):
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops, log_device_placement=False)
    sess = tf.Session(config=config)

    env = EnvWrapper(args, mod_r=True)
    env_test = EnvWrapper(args, mod_r=False)

    if args.use_mr:
        print('Set experience replay ON')
    else:
        print('Set experience replay OFF')

    if args.quick_save:
        print('Set quick save        ON')
    else:
        print('Set quick save        OFF')

    mr = MemoryReplayer(env.state_shape, capacity=args.mr_capacity, enabled=args.use_mr)

    # set type='v1' for linear model, 'v3' for three layer model (two tanh activations)

    # type='v5' use dual

    print('Set Q-network version: ', args.qn_version)
    qn = DeepQN(state_shape=env.state_shape, num_actions=env.num_actions, gamma=args.gamma, type=args.qn_version)

    qn.reset_sess(sess)

    qn.set_train(lr=args.lr, beta1=args.beta1, beta2=args.beta2)


    if not args.reuse_model:
        print('Set reuse model      OFF')
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        print('Set reuse model      ON')
        qn.load(args.model_path)

    plotter = Plotter(save_path=args.performance_plot_path, interval=args.performance_plot_interval,
                      episodes=args.performance_plot_episodes)

    pretrain_test = Tester(qn, env, report_interval=100)
    print('Pretrain test:')
    pretrain_test.run(qn, sess)
    print('Pretrain test done.')

    test = Tester(qn, env_test, episodes=args.tester_episodes, report_interval=args.tester_report_interval)

    score = deque([], maxlen=args.performance_plot_episodes)
    reward_record = []
    cnt_iter = 0


    for epi in range(args.max_episodes):
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
                r_avg, _ = test.run(qn, sess)
                reward_record.append(r_avg)

        score.append(rc)

        # replay

        s, s_, r, a, done = mr.replay(batch_size=args.batch_size)

        qn.train(s, s_, r, a, done)

        if (epi + 1) % args.quick_save_interval == 0 and args.quick_save:
            qn.save('./tmp/quick_save.ckpt')
            
        if (epi + 1) % args.performance_plot_interval == 0:
            plotter.plot(np.mean(score))

        if cnt_iter > args.max_iter:
            break

    qn.save(args.model_path)
    f = open(args.log_name, 'w')
    f.write(str(reward_record))
    f.close()
    return

def test(args):
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)
    env = EnvWrapper(args.env)
    qn = DeepQN(state_shape=env.state_shape, num_actions=env.num_actions, gamma=args.gamma, type=args.qn_version)
    qn.reset_sess(sess)
    qn.load(args.model_path)
    testor = Tester(qn, env, report_interval=args.tester_report_interval, episodes=args.tester_episodes)
    _, rs = testor.run(qn, sess, render=args.render)
    f = open(args.model_path+'_test.log', 'w')
    f.write(str(rs))
    f.close()
    return

def get_eps(t):
    return max(0.01, 1.0 - np.log10(t + 1) * 0.995)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model_path',dest='model_path',type=str, default='./tmp/blabla.ckpt')
    parser.add_argument('--use_mr', dest='use_mr', type=int, default=1)
    parser.add_argument('--mr_capacity', dest='mr_capacity', type=int, default=100000)
    parser.add_argument('--gamma', dest='gamma', type=float, default=1.0)
    parser.add_argument('--qn_version', dest='qn_version', type=str, default='v1')
    parser.add_argument('--learning_rate', dest='lr', type=float, default=0.0001)
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--max_iter', dest='max_iter', type=int, default=1000000)
    parser.add_argument('--max_episodes', dest='max_episodes', type=int, default=100000)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--tester_report_interval', dest='tester_report_interval', type=int, default=20)
    parser.add_argument('--tester_episodes', dest='tester_episodes', type=int, default=20)
    parser.add_argument('--quick_save', dest='quick_save', type=int, default=1)
    parser.add_argument('--quick_save_interval', dest='quick_save_interval', type=int, default=200)
    parser.add_argument('--performance_plot_path', dest='performance_plot_path', type=str, default='./figure/perfplot.png')
    parser.add_argument('--performance_plot_interval', dest='performance_plot_interval', type=int, default=20)
    parser.add_argument('--performance_plot_episodes', dest='performance_plot_episodes', type=int, default=100)
    parser.add_argument('--reuse_model', dest='reuse_model', type=int, default=0)
    parser.add_argument('--use_monitor', dest='use_monitor', type=int, default=0)
    return parser.parse_args()

def main(argv):
    # parse arguments
    args = parse_arguments()

    if args.use_mr == 0 and args.qn_version == 'v1':
        qnum = "q1"
    elif args.use_mr == 1 and args.qn_version == 'v1':
        qnum = "q2"
    elif args.use_mr == 1 and args.qn_version == 'v3':
        qnum = "q3"
    elif args.use_mr == 1 and args.qn_version == 'v5':
        qnum = "q4"
    elif args.use_mr == 1 and args.qn_version == 'v4' and args.env == 'SpaceInvaders-v0':
        qnum = "q5"
    else:
        print("Wrong settings!")
        return
    args.log_name = "{}_{}_data.log".format(args.env, qnum)
    args.qnum = qnum
    args.model_path = "{}_{}_model".format(args.env, qnum)
    if args.train == 1:
        train(args)
    else:
        test(args)

if __name__ == '__main__':
    main(sys.argv)
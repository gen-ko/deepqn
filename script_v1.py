#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

from memory_replay import MemoryReplayer
from deep_qn import DeepQN
from tester import Tester
from plotter import Plotter

from env_wrapper import EnvWrapper


def train(args=None):
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops, log_device_placement=False)
    sess = tf.Session(config=config)

    env = EnvWrapper(env_name=args.env, mod_r=True)
    env_test = EnvWrapper(args.env, mod_r=False)

    mr = MemoryReplayer(env.state_shape, capacity=args.mr_capacity, enabled=args.use_mr)

    # set type='v1' for linear model, 'v3' for three layer model (two tanh activations)

    # type='v5' use dual

    qn = DeepQN(state_shape=env.state_shape, num_actions=env.num_actions, gamma=args.gamma, type=args.qn_version)

    qn.reset_sess(sess)

    qn.set_train(lr=args.lr, beta1=args.beta1, beta2=args.beta2)

    init = tf.global_variables_initializer()
    sess.run(init)

    plotter = Plotter()

    pretrain_test = Tester(qn, env, report_interval=100)
    print('Pretrain test:')
    pretrain_test.run(qn, sess)
    print('Pretrain test done.')

    test = Tester(qn, env_test, episodes=args.tester_episodes, report_interval=args.tester_report_interval)

    score = []
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
                qn.save(args.model_path)
                reward_record.append(test.run(qn, sess))

        score.append(rc)

        # replay

        s, s_, r, a, done = mr.replay(batch_size=args.batch_size)

        qn.train(s, s_, r, a, done)

        if cnt_iter > args.max_iter:
            break

        # if (epi + 1) % 200 == 0:
        #     avg_score = np.mean(score)
        #     plotter.plot(avg_score)
        #     print('avg score last 200 episodes ', avg_score)
        #     score = []
        #     if avg_score > 195:
        #         break
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
    testor.run(qn, sess, render=args.render)
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
    return parser.parse_args()

def main(argv):
    # parse arguments
    args = parse_arguments()

    if args.use_mr == 0 and args.qn_version == 'v1':
        log_name = "{}-v0_q1_data.log".format(args.env)
        model_path = "tmp/{}-v0_q1_model".format(args.env)
    elif args.use_mr == 1 and args.qn_version == 'v1':
        log_name = "{}-v0_q2_data.log".format(args.env)
        model_path = "tmp/{}-v0_q2_model".format(args.env)
    elif args.use_mr == 1 and args.qn_version == 'v3':
        log_name = "{}-v0_q3_data.log".format(args.env)
        model_path = "tmp/{}-v0_q3_model".format(args.env)
    elif args.use_mr == 1 and args.qn_version == 'v5':
        log_name = "{}-v0_q4_data.log".format(args.env)
        model_path = "tmp/{}-v0_q4_model".format(args.env)
    elif args.use_mr == 1 and args.qn_version == 'v4' and args.env == 'SpaceInvaders-v0':
        log_name = "{}-v0_q5_data.log".format(args.env)
        model_path = "tmp/{}-v0_q5_model".format(args.env)
    else:
        print("Wrong settings!")
        return
    args.log_name = log_name
    args.model_path = model_path
    if args.train == 1:
        train(args)
    else:
        test(args)
    #test(env_name, model_path, is_render)

if __name__ == '__main__':
    main(sys.argv)
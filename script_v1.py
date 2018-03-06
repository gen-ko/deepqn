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

    test = Tester(qn, env_test, 20, 20)

    score = []
    reward_record = []
    cnt_iter = 0

    max_iter = args.max_iter
    max_episodes = args.max_episodes
    batch_size = args.batch_size

    for epi in range(max_episodes):
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
                reward_record.append(test.run(qn, sess))

        score.append(rc)

        # replay

        s, s_, r, a, done = mr.replay(batch_size=batch_size)

        qn.train(s, s_, r, a, done)

        if cnt_iter > max_iter:
            break

        # if (epi + 1) % 200 == 0:
        #     avg_score = np.mean(score)
        #     plotter.plot(avg_score)
        #     print('avg score last 200 episodes ', avg_score)
        #     score = []
        #     if avg_score > 195:
        #         break
    qn.save(model_path)
    f = open(log_name, 'w')
    f.write(str(reward_record))
    f.close()
    return

def test(env_name, model_path, render=False, episodes=100):
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    qn = DeepQN(state_shape=(2,), num_actions=3, gamma=0.99)

    qn.reset_sess(sess)

    qn.load(model_path)

    env = gym.make(env_name)

    testor = Tester(qn, env, report_interval=100, episodes=episodes)

    testor.run(qn, sess, render=render)

    return

def get_eps(t):
    return max(0.01, 1.0 - np.log10(t + 1) * 0.995)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
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
    return parser.parse_args()

def main(argv):
    # parse arguments
    args = parse_arguments()

    env_name = args.env
    has_memrory = args.use_mr == 1

    qn_ver = 'v1'
    if not has_memrory and qn_ver == 'v1':
        log_name = "{}-v0_q1_data.log".format(env_name)
        model_path = "tmp/{}-v0_q1_model".format(env_name)
    elif has_memrory and qn_ver == 'v1':
        log_name = "{}-v0_q2_data.log".format(env_name)
        model_path = "tmp/{}-v0_q2_model".format(env_name)
    elif has_memrory and qn_ver == 'v3':
        log_name = "{}-v0_q3_data.log".format(env_name)
        model_path = "tmp/{}-v0_q3_model".format(env_name)
    elif has_memrory and qn_ver == 'v5':
        log_name = "{}-v0_q4_data.log".format(env_name)
        model_path = "tmp/{}-v0_q4_model".format(env_name)
    elif has_memrory and qn_ver == 'v4' and env_name == 'SpaceInvaders-v0':
        log_name = "{}-v0_q5_data.log".format(env_name)
        model_path = "tmp/{}-v0_q5_model".format(env_name)
    else:
        print("Wrong settings!")
        return
    train(args)
    #test(env_name, model_path, is_render)

if __name__ == '__main__':
    main(sys.argv)
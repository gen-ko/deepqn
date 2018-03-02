import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse

class Agent:
    def __init__(self):
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        self.sess = tf.Session(config=config)
        self.save_path = "./tmp/model_dqn_v1.ckpt"
        self.eps = 0.9
        self.num_actions = 0
        return

    def epsilon_greedy_policy(self, q):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform(low=0.0, high=1.0) < self.eps:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(q)
        return action

    def greedy_policy(self, q):
        action = np.argmax(q)
        return action

    def train(self):
        pass
    
    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.save_path)
        print("Model saved in path: %s" % save_path)

    def load_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.save_path)


    def test(self, render=False):
        pass
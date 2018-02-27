#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse

SIMPLE_ENVS = set(['CartPole-v0', 'MountainCar-v0'])

class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        if env.spec.id in SIMPLE_ENVS:
            # Initialize a linear function
            self.input = tf.placeholder(dtype=tf.float32, shape=env.observation_space.shape, name='input')
            self.output = tf.layers.dense(
                inputs=self.h2,
                units=env.action_space.n,
                activation=None,
                use_bias=True,
                kernel_initializer=None,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name='output',
                reuse=None)
        else:


        # using "NCHW" format
        self.input = tf.placeholder(dtype=tf.float32, shape=(32, 4, 84, 84), name="input")
        self.h1 = tf.layers.conv2d(
            inputs=self.input,
            filters=16,
            kernel_size=[8, 8],
            strides=(4, 4),
            padding="same",
            activation=tf.nn.relu,
            data_format='channels_first',
            name='h1')
        self.h2 = tf.layers.conv2d(
            inputs=self.h1,
            filters=32,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu,
            data_format='channels_first',
            name='h2')

        # dense layer automatically make the inputs flattened
        self.h3 = tf.layers.dense(
            inputs=self.h2,
            units=256,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name='h3',
            reuse=None
        )

        # output layer
        self.output = tf.layers.dense(
            inputs=self.h3,
            units=env.action_space.n,
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            trainable=True,
            name='output',
            reuse=None
        )
        return

    def save_model_weights(self, sess):
        # Helper function to save your model / weights.

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
        return


    def load_model(self, sess):
        # Helper function to load an existing model.

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model restored.")
        return

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        pass


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        pass

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        pass

    def append(self, transition):
        # Appends transition to the memory.
        pass


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env, epsilon=0.05, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.model = QNetwork(env=env)
        self.epsilon = epsilon

        return

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() > self.epsilon:
            # choose the action with maximum q
            return np.argmax(q_values)
        else:
            return np.random.randint(low=0, high=len(q_values))


    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)
        pass


    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        pass

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        pass

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    env = gym.make(args.env)

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    agent = DQN_Agent('DAMN')
    agent.train()
    agent.test(model_file='model.mdf')

    # Finish
    print('script concluded.')

if __name__ == '__main__':
    main(sys.argv)


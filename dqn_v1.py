#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse




def epsilon_greedy_policy(q, num_actions, batch_size=1):
    # Creating epsilon greedy probabilities to sample from.

    eps = 0.1

    # return this operator
    if batch_size == 1:
        # actions shape (1, )
        if np.random.uniform(low=0.0, high=1.0) < eps:
            actions = np.random.randint(0, num_actions)
        else:
            actions = np.argmax(q)
    else:
        # actions shape (batch_size, 1)
        actions = np.zeros((batch_size, ), dtype=int)
        for i in range(batch_size):
            if np.random.uniform(low=0.0, high=1.0) < eps:
                actions[i] = np.random.randint(0, num_actions)
            else:
                actions[i] = np.argmax(q[i])
    return actions


def greedy_policy(q, num_actions, batch_size=1):
    if batch_size == 1:
        # actions shape (1, )
        actions = np.argmax(q)
    else:
        # actions shape (batch_size, 1)
        actions = np.zeros((batch_size, ), dtype=int)
        for i in range(batch_size):
            actions[i] = np.argmax(q[i])
    return actions


class DQN_v1(object):
    def __init__(self, batch_size, state_dim, num_actions, learning_rate=0.0001):
        # build the network
        self.state_ph = tf.placeholder(dtype=tf.float32,
                                  shape=(batch_size, state_dim),
                                  name='state_current_ph')

        self.state_next_ph = tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, state_dim),
                                       name='state_next_ph')


        # a linear dense layer
        self.q_tf = tf.layers.dense(inputs=self.state_ph,
                               units=num_actions,
                               use_bias=True,
                               kernel_initializer=tf.random_normal_initializer(),
                               bias_initializer=tf.zeros_initializer(),
                               name='q',
                               trainable=True,
                               reuse=None)

        self.target_ph = tf.placeholder(dtype=tf.float32,
                                   shape=(batch_size, num_actions),
                                   name='target_ph')

        self.loss_tf = tf.reduce_mean(tf.squared_difference(self.target_ph, self.q_tf))

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_tf)
        return


def train():
    env = gym.make('CartPole-v0')

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    num_actions = env.action_space.n

    # the dimension of a state observation
    state_dim = env.observation_space.shape[0]

    batch_size = 1


    learning_rate = 0.0001

    stop_criteria = 0.0001

    gamma = 0.99

    iterations: int = 1000000

    # instantiate the DQN
    dqn = DQN_v1(batch_size, state_dim, num_actions, learning_rate)


    # operate on the env
    if batch_size == 1:
        state = env.reset()
        state = np.array(state)
        state = state.reshape((batch_size, state_dim))
    else:
        env = [env]
        for i in range(1, batch_size):
            env.append(gym.make('CartPole-v0'))

        state = np.zeros(shape=(batch_size, state_dim), dtype=np.float32)
        for i in range(batch_size):
            state[i, :] = env[i].reset()


    # initialize tf graph
    init = tf.global_variables_initializer()
    sess.run(init)

    if batch_size == 1:
        episode_num: int = 0
        loop_counter: int = 0

    else:
        episode_num = np.zeros(shape=(batch_size, ), dtype=int)
        loop_counter = np.zeros(shape=(batch_size, ), dtype=int)

    cumulative_reward = 0.0
    avg_reward = 0.0
    cumulative_episode: int= 0


    for iter_i in range(iterations):
        # find the q values

        # q shape (batch_size, num_actions)

        q = sess.run(fetches=dqn.q_tf,
                     feed_dict={dqn.state_ph: state})


        # action shape (batch_size, num_actions)

        # find the corresponding action
        action = epsilon_greedy_policy(q, num_actions, batch_size)

        if batch_size == 1:
            state_next, reward, is_terminal, info = env.step(action)
            state_next = np.array(state_next)
            state_next = state_next.reshape((batch_size, state_dim))
        else:
            state_next = np.zeros(shape=(batch_size, state_dim), dtype=np.float32)
            reward = np.zeros(shape=(batch_size, ), dtype=np.float32)
            is_terminal = np.zeros(shape=(batch_size, ), dtype=np.bool)
            for i in range(batch_size):
                state_next[i], reward[i], is_terminal[i], _ = env[i].step(action[i])

        q_next = sess.run(fetches=dqn.q_tf,
                          feed_dict={dqn.state_ph: state_next})

        # optimal action
        if batch_size == 1:
            action_max = np.argmax(q_next, axis=1)
            q_next_max = max(max(q_next))
        else:
            action_max = np.argmax(q_next, axis=1)
            q_next_max = np.max(q_next, axis=1)



        target = np.array(q)

        if batch_size == 1:
            target[:, action] = gamma * q_next_max + reward
        else:
            for i in range(batch_size):
                target[i, action[i]] = gamma * q_next_max[i] + reward[i]


        sess.run(fetches=dqn.train_op,
                 feed_dict={dqn.target_ph: target,
                            dqn.state_ph: state})

        # prepare the next loop
        state = state_next

        if batch_size == 1:
            discrepency = abs((target[:, action] - q[:, action]))
        else:
            discrepency = sum(sum(abs((target[:, action] - q[:, action]))))

        cumulative_reward += reward

        if batch_size == 1:
            if is_terminal:
                episode_num += 1
                state = env.reset()
                state = np.array(state)
                state = state.reshape((batch_size, state_dim))
                # print('episode: ', episode_num, 'terminated')
                # print('loop count: ', loop_counter)
                # print('discrepency: ', discrepency)

                cumulative_episode += 1

                avg_reward += cumulative_reward
                cumulative_reward = 0.0

                if cumulative_episode % 100 == 99:
                    print('recent 100 episoe average award: ', avg_reward / 100)
                    avg_reward = 0

                loop_counter = 0
        else:
            for i in range(batch_size):
                if is_terminal[i]:
                    episode_num[i] += 1
                    state[i] = env[i].reset()
                    # print("batch id: ", i)
                    # print('episode: ', episode_num[i], 'terminated')
                    # print('loop count: ', loop_counter[i])
                    # print('discrepency: ', discrepency)
                    loop_counter[i] = 0

        # loop_counter += 1
        iter_i += 1
        if iter_i % 10000 == 9999:
            print('iterations passed: ', iter_i + 1)


    print('Training done')
    #print('discrepency: ', discrepency)

    # save model
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.

    save_path = saver.save(sess, "./tmp/model_dqn_v1.ckpt")
    print("Model saved in path: %s" % save_path)

    return


def test(render=False):
    tf.reset_default_graph()
    env = gym.make('CartPole-v0')
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    num_actions = env.action_space.n

    # the dimension of a state observation
    state_dim = env.observation_space.shape[0]

    batch_size = 1

    learning_rate = 0.0001

    stop_criteria = 0.0001

    gamma = 1

    test_episodes = 100

    # instantiate the DQN
    dqn = DQN_v1(batch_size, state_dim, num_actions, learning_rate)

    # load weights
    saver = tf.train.Saver()

    saver.restore(sess, "./tmp/model_dqn_v1.ckpt")

    avg_episode_reward = 0.0
    for epi in range(test_episodes):
        # initialize the state
        if batch_size == 1:
            state = env.reset()
            state = np.array(state)
            state = state.reshape((batch_size, state_dim))
        else:
            env = [env]
            for i in range(1, batch_size):
                env.append(gym.make('CartPole-v0'))

            state = np.zeros(shape=(batch_size, state_dim), dtype=np.float32)
            for i in range(batch_size):
                state[i, :] = env[i].reset()

        # WARNING: Assuming batch_size = 1
        cumulative_reward = 0.0

        while True:

            # find the q values

            # q shape (batch_size, num_actions)
            if render:
                env.render()

            q = sess.run(fetches=dqn.q_tf,
                         feed_dict={dqn.state_ph: state})

            # action shape (batch_size, num_actions)

            # find the corresponding action
            action = greedy_policy(q, num_actions, batch_size)

            if batch_size == 1:
                state_next, reward, is_terminal, info = env.step(action)
                state_next = np.array(state_next)
                state_next = state_next.reshape((batch_size, state_dim))
                cumulative_reward = cumulative_reward * gamma + reward
            else:
                state_next = np.zeros(shape=(batch_size, state_dim), dtype=np.float32)
                reward = np.zeros(shape=(batch_size,), dtype=np.float32)
                is_terminal = np.zeros(shape=(batch_size,), dtype=np.bool)
                for i in range(batch_size):
                    state_next[i], reward[i], is_terminal[i], _ = env[i].step(action[i])



            # prepare the next loop
            state = state_next

            if batch_size == 1:
                if is_terminal:
                    state = env.reset()
                    state = np.array(state)
                    state = state.reshape((batch_size, state_dim))
                    avg_episode_reward += cumulative_reward / test_episodes
                    cumulative_reward = np.float32(0)
                    break
            else:
                for i in range(batch_size):
                    if is_terminal[i]:
                        state[i] = env[i].reset()

    print('Testing done')
    print('Average discounted episode cumulative reward: ', avg_episode_reward)
    return


def main():
    tf.set_random_seed(9999)
    is_train = True
    is_test = True
    if is_train:
        train()
    if is_test:
        test()
    return


if __name__ == '__main__':
    main()
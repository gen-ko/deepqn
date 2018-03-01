#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse


def epsilon_greedy_policy(q, num_actions):
    # Creating epsilon greedy probabilities to sample from.

    eps = 0.01

    # return this operator

    if np.random.uniform(low=0.0, high=1.0) < eps:
        actions = np.random.randint(0, num_actions)
    else:
        actions = np.argmax(q)

    return actions


def main(args):

    env = gym.make('CartPole-v0')



    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    num_actions = env.action_space.n

    # the dimension of a state observation
    state_dim = env.observation_space.shape[0]

    batch_size = 1

    learning_rate = 0.001

    stop_criteria = 0.001



    # build the network
    state_ph = tf.placeholder(dtype=tf.float32,
                              shape=(batch_size, state_dim),
                              name='state_current_ph')

    state_next_ph = tf.placeholder(dtype=tf.float32,
                                   shape=(batch_size, state_dim),
                                   name='state_next_ph')

    reward_ph = tf.placeholder(dtype=tf.float32,
                               shape=(batch_size, num_actions),
                               name='reward_ph')

    # a linear dense layer
    q_tf = tf.layers.dense(inputs=state_ph,
                           units=num_actions,
                           use_bias=True,
                           kernel_initializer=tf.random_uniform_initializer(),
                           bias_initializer=tf.zeros_initializer(),
                           name='q',
                           trainable=True,
                           reuse=None)

    target_ph = tf.placeholder(dtype=tf.float32,
                               shape=(batch_size, num_actions),
                               name='reward_ph')


    loss_tf = tf.reduce_mean(tf.squared_difference(target_ph, q_tf))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_tf)


    # operate on the env
    state = env.reset()
    state = np.array(state)
    state = state.reshape((batch_size, state_dim))
    # initialize tf graph
    init = tf.global_variables_initializer()
    sess.run(init)

    episode_num: int = 0
    loop_counter: int = 0
    while True:
        # find the q values
        q = sess.run(fetches=q_tf,
                     feed_dict={state_ph: state})

        # find the corresponding action
        action = epsilon_greedy_policy(q, num_actions)

        state_next, reward, is_terminal, info = env.step(action)

        state_next = np.array(state_next)
        state_next = state_next.reshape((batch_size, state_dim))

        q_next = sess.run(fetches=q_tf,
                          feed_dict={state_ph: state_next})

        # optimal action
        action_max = np.argmax(q_next, axis=1)
        q_next_max = max(max(q_next))

        target = np.array(q)

        target[:, action] = q_next_max + reward

        sess.run(fetches=train_op,
                 feed_dict={target_ph: target,
                            state_ph: state})

        # prepare the next loop
        state = state_next

        discrepency = abs((target[:, action] - q[:, action])[0])
        if is_terminal:
            episode_num += 1
            state = env.reset()
            state = np.array(state)
            state = state.reshape((batch_size, state_dim))
            print('episode: ', episode_num, 'terminated')
            print('loop count: ', loop_counter)
            print('discrepency: ', discrepency)
            loop_counter = 0

        if discrepency < stop_criteria:
            break

        loop_counter += 1

    print('Training done')
    print('discrepency: ', discrepency)
    return




if __name__ == '__main__':
    main(sys.argv)
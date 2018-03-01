#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse




def epsilon_greedy_policy(q, num_actions, batch_size=1 ):
    # Creating epsilon greedy probabilities to sample from.

    eps = 0.05

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
                           kernel_initializer=tf.random_normal_initializer(),
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

    while True:
        # find the q values

        # q shape (batch_size, num_actions)

        q = sess.run(fetches=q_tf,
                     feed_dict={state_ph: state})


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

        q_next = sess.run(fetches=q_tf,
                          feed_dict={state_ph: state_next})

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


        sess.run(fetches=train_op,
                 feed_dict={target_ph: target,
                            state_ph: state})

        # prepare the next loop
        state = state_next

        if batch_size == 1:
            discrepency = abs((target[:, action] - q[:, action]))
        else:
            discrepency = sum(sum(abs((target[:, action] - q[:, action]))))

        if batch_size == 1:
            if is_terminal:
                episode_num += 1
                state = env.reset()
                state = np.array(state)
                state = state.reshape((batch_size, state_dim))
                print('episode: ', episode_num, 'terminated')
                print('loop count: ', loop_counter)
                print('discrepency: ', discrepency)
                cumulative_reward = np.float32(0)
                loop_counter = 0
        else:
            for i in range(batch_size):
                if is_terminal[i]:
                    episode_num[i] += 1
                    state[i] = env[i].reset()
                    print("batch id: ", i)
                    print('episode: ', episode_num[i], 'terminated')
                    print('loop count: ', loop_counter[i])
                    print('discrepency: ', discrepency)
                    loop_counter[i] = 0

        if discrepency < stop_criteria:
            break

        loop_counter += 1

    print('Training done')
    print('discrepency: ', discrepency)

    # save model
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.

    save_path = saver.save(sess, "./tmp/model_dqn_v1.ckpt")
    print("Model saved in path: %s" % save_path)

    return


def main():
    train()
    return


if __name__ == '__main__':
    main()
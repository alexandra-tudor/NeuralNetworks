import tensorflow as tf
import gym
import numpy as np


# _____________F u n c t i o n s_____________
def crop_grayscale(rgb):
    # r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # #
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.array(rgb[54:195, 9:150, :])


def out(images, net):
    l1 = tf.nn.relu(tf.nn.conv2d(images, net[1]['W'], strides=[1, 4, 4, 1], padding='SAME') + net[1]['b'])
    l2 = tf.nn.relu(tf.nn.conv2d(l1, net[2]['W'], strides=[1, 2, 2, 1], padding='SAME') + net[2]['b'])
    l2_flat = tf.reshape(l2, [-1, 10368])
    fc = tf.nn.relu(tf.nn.xw_plus_b(l2_flat, net['FC']['W'], net['FC']['b']))
    output = tf.nn.xw_plus_b(fc, net['output']['W'], net['output']['b'])
    return output


def sampler(env, sample_size):
    sample_counter = 0
    memory = []
    while sample_counter < sample_size:

        done = False

        obs = env.reset()

        while not done:

            env.render()
            action = env.action_space.sample()
            new_obs, reward, done, info = env.step(action)

            if not done and reward > 0:
                memory.append([crop_grayscale(obs), action, crop_grayscale(new_obs), reward])
                sample_counter += 1

            obs = new_obs

    return memory


def get_action(session, net, observation):
    return session.run([tf.argmax(out(observation, net), axis=1)])


def get_Q(session, net, observation):
    return session.run([tf.argmax(out(observation, net), axis=1)])


def memory_sampler(memory, sampling_scale):
    sample_memory = []

    for i in range(sampling_scale):
        sample_memory.append(memory.pop(np.random.randint(0, len(memory) - 1)))

    return sample_memory


def train(sess, memory, agent, input_shape, op):
    obs = np.array([np.array(memory[i][0], dtype=np.float32) for i in range(len(memory))])

    new_obs = np.array([np.array(memory[i][2], dtype=np.float32) for i in range(len(memory))])

    reward = np.array([memory[i][3] for i in range(len(memory))], dtype=np.float32)

    obs_place = tf.placeholder('float', shape=[None, 141,141,3], name='Observations')
    obs_place = tf.reshape(obs_place, [-1, 141, 141, 3])
    new_obs_place = tf.placeholder('float', shape=[None, 141,141,3], name='NewObservations')
    new_obs_place = tf.reshape(new_obs_place, [-1, 141, 141, 3])
    Q = out(obs_place, agent)

    Q_ = out(new_obs_place, agent)

    reward_place = tf.placeholder('float', shape=[None], name='Rewards')
    cost = tf.reduce_mean(

        tf.pow(reward_place + 0.6 * tf.reduce_max(Q_, axis=1) - tf.reduce_max(Q, axis=1), 2)

    )

    optimizer = op.minimize(cost)
    total = 0
    for i in range(10):
        _, c = sess.run([optimizer, cost], feed_dict={
            obs_place: obs,
            new_obs_place: new_obs,
            reward_place: reward
        })

        total += c
    print('loss:', total / 10)


# ==========================
# Game & Agent Configuration
# ==========================

episode = 1

frame_per_episode = 10

training_time = 10

enviroment = gym.make('Breakout-v0')

sampling_scale = 20

random_action_proba = 0.6

game_input_shape = 141 * 141 * 3

game_output_shape = 4

agent = {
    1: {
        'W': tf.Variable(tf.truncated_normal([8, 8, 3, 16]), name='W1'),
        'b': tf.Variable(tf.truncated_normal([16]), name='b1')
    },
    2: {
        'W': tf.Variable(tf.truncated_normal([4, 4, 16, 32]), name='W2'),
        'b': tf.Variable(tf.truncated_normal([32]), name='b2')
    },
    'FC': {
        'W': tf.Variable(tf.truncated_normal([10368, 256]), name='W_FC'),
        'b': tf.Variable(tf.truncated_normal([256]), name='b_FC')
    },
    'output': {
        'W': tf.Variable(tf.truncated_normal([256, game_output_shape]), name='W_out'),
        'b': tf.Variable(tf.truncated_normal([game_output_shape]), name='b_out')
    }
}

is_train_time = True

remaining_time_to_train = 0

model_save_path = 'model.ckpt'

memory = sampler(enviroment, 10)

# ===============================
# Playing Game and Training Agent
# ===============================

with tf.Session() as sess:
    op = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    saver = tf.train.Saver()
    saver.restore(sess, model_save_path)

    # sess.run(tf.global_variables_initializer())

    for e in range(episode):

        obs = enviroment.reset()
        print (e)
        score = 0
        done = False

        while not done:
            enviroment.render()

            do_random_action = np.random.randint(1, 100) / 100

            obs = np.array(obs, dtype=np.float32)

            if do_random_action > random_action_proba:

                action = enviroment.action_space.sample()

            else:

                action = get_action(sess, agent, np.array(crop_grayscale(obs)).reshape(-1,141,141,3))[0][0]

            new_obs, reward, done, info = enviroment.step(action)
            new_obs = np.array(new_obs, np.float32)

            # =============================================
            # Adding New game information to 'Agent memory'
            # =============================================

            if not done:
                memory.append([crop_grayscale(obs), action, crop_grayscale(new_obs), reward, done])

            remaining_time_to_train += 1
            print(remaining_time_to_train)
            score += reward
            print("Score {}".format(score))
            obs = new_obs

            if remaining_time_to_train % 5 == 0:
                is_train_time = True

            if is_train_time:
                is_train_time = False

                print('Training Process Started...')
                print('Episode:', e)
                print('======================================')

                train(sess, memory, agent, game_input_shape, op)

        print('Score:', score)
    saver.save(sess, model_save_path)

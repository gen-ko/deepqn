import numpy as np
import cv2
import os.path
import pickle


def burn_in(env, mr):
    path = './tmp/burn_in_' + env.env_name + '-' + str(mr.capacity) + '.pickle'
    if os.path.exists(path):
        print('Found existing burn_in memory replayer, load...')
        with open(path, 'rb') as f:
            mr = pickle.load(file=f)
            return mr
    print('No exist burn_in memory replayer found')
    print('start burn-in')
    for i in range(int(mr.capacity)):
        s = env.reset()
        done = False
        while not done:
            a = np.random.randint(env.num_actions)
            s_, r, done, _ = env.step(a)
            mr.remember(s, s_, r=r, a=a, done=done)
            s = s_

    with open(path, 'wb+') as f:
        pickle.dump(mr, f)
    print('end burn-in')
    return mr

def image_prep(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (84, 110))
    img_cropped = img_resized[18:102, :]
    return img_cropped
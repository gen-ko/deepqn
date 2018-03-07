import numpy as np
import cv2


def burn_in(env, mr):
    for i in range(mr.capacity):
        s = env.reset()
        done = False
        while not done:
            a = np.random.randint(env.num_actions)
            s_, r, done, _ = env.step(a)
            mr.remember(s, s_, r=r, a=a, done=done)
            s = s_

def image_prep(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (84, 110))
    img_cropped = img_resized[18:102, :]
    return img_cropped
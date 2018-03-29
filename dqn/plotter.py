import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class Plotter(object):
    def __init__(self, fig_num=0, capacity=20000, save_path='./figure/v1',
                 show_figure=False, interval=100, episodes=100):
        self.fig_num = fig_num
        self.x = deque(iterable=[], maxlen=capacity)
        self.t = deque(iterable=[], maxlen=capacity)
        self.show_figure = show_figure
        self.t_now = 0
        self.interval = interval
        self.episodes = episodes
        self.title_text = 'Average Reward on {interval} Steps'.format(interval=self.episodes)
        self.save_path = save_path
        return

    def plot(self, x):
        plt.figure(self.fig_num)
        self.x.append(x)
        self.t.append(self.t_now)
        self.t_now += self.interval
        line_1, = plt.plot(self.t, self.x, label='line_0')
        plt.legend(handles=[line_1])
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title(self.title_text)
        plt.savefig(self.save_path)
        plt.close(self.fig_num)
        return




import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class Plotter(object):
    def __init__(self, fig_num=0, capacity=20000, titletext='reward figure', savepath='./figure/v1',
                 show_figure=False):
        self.fig_num = fig_num
        self.x = deque(iterable=[], maxlen=capacity)
        self.t = deque(iterable=[], maxlen=capacity)
        self.show_figure = show_figure

        self.t_now = 0
        self.titletext = titletext
        self.savepath = savepath

        return

    def plot(self, x):
        plt.figure(self.fig_num)
        self.x.append(x)
        self.t.append(self.t_now)
        self.t_now += 1
        line_1, = plt.plot(self.t, self.x, label='reward')
        plt.legend(handles=[line_1])
        plt.xlabel('time step')
        plt.ylabel('reward')
        plt.title(self.titletext)
        plt.savefig(self.savepath)
        plt.close(self.fig_num)
        return




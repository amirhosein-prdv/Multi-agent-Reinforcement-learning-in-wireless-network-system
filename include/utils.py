import numpy as np
import copy
import matplotlib.pyplot as plt

class DotDic(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))


def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = list(range(max(i - window + 1, 0),
                                min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)

def plot_learning_curve(scores):
    x = [i+1 for i in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Rewards versus episodes')
    plt.xlabel('episode'); plt.ylabel('total rewards')
    #plt.savefig(figure_file)
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

file_species = 'data'
test_species = 'test'


def load_config():  # Load configure
    with open(file_species, 'r') as f:
        config = f.readline()
        config = config.split(' ')
    return config


def load_data():  # load configure
    class_set = []
    with open(file_species, 'r') as f:
        config = f.readline()
        config = config.split(' ')
        lines = [line.strip() for line in f.readlines()]

    lines = [line.split(",") for line in lines if line]  # Remove ','
    num_line = 0
    for i in range(int(config[0])):  # classification
        temp = []
        for j in range(int(config[1])):
            temp += [lines[num_line + j]]
        temp = np.asarray(temp, dtype=np.float)
        temp = np.transpose(temp)
        class_set += [temp]
        num_line += int(config[1])
    return class_set


def load_test_data():  # Load test data
    with open(test_species, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    lines = [line.split(",") for line in lines if line]
    return lines


def get_w():  # Calculate w
    config = load_config()
    num_species = int(config[0])
    num_feature = int(config[1])
    data_ = load_data()
    m = np.zeros(num_feature)
    num_total = 0
    for i in range(len(data_)):
        m += sum(data_[i])
        num_total += len(data_[i])
    m /= num_total
    # Mean matrix
    mean_vector = [np.mean(data_[i], axis=0) for i in range(num_species)]
    # Within class scatter matrix
    sw = sum([np.dot((data_[i] - mean_vector[i]).T, (data_[i] - mean_vector[i])) for i in range(num_species)])
    # Between class scatter matrix
    sb = sum([np.outer((mean_vector[i] - m), (mean_vector[i] - m).T) * len(data_[i]) for i in range(num_species)])
    a = np.dot(np.linalg.inv(sw), sb)
    value, vector = np.linalg.eig(a)
    w_ = vector[:, :2]
    draw_plot(data_, w_)
    w_ = np.asarray(w_, dtype=np.float)
    return w_


def draw_plot(data_, w):
    class1 = np.dot(data_[0], w)
    class2 = np.dot(data_[1], w)
    class3 = np.dot(data_[2], w)
    plt.plot(class1[:, 0], class1[:, 1], "bs", label="w1")
    plt.plot(class2[:, 0], class2[:, 1], "go", label="w2")
    plt.plot(class3[:, 0], class3[:, 1], "rp", label="w3")

    plt.legend()
    plt.show()


def get_threshold(w_):
    w = np.asarray(w_)
    w = w.T
    data_ = load_data()
    y = []
    for i in range(len(data_)):
        temp = []
        for j in range(len(data_[i])):
            x = np.asarray(data_[i][j])
            x = x.reshape(-1, 1)
            res = np.dot(w, x)
            res = np.array([res[0][0], res[1][0]])
            temp.append(res)
        y.append(temp)
    mean = [np.mean(y[i], axis=0) for i in range(len(y))]
    dic = {}
    for i in range(len(mean)):
        s = 'w' + str(i + 1)
        dic[s] = mean[i]
    return dic


def discriminate(w, data_, threshold):
    for i in range(len(data_)):
        w_ = w.reshape(2, -1)
        x = np.asarray(data_[i], dtype=float)
        x = x.reshape(-1, 1)
        res = np.dot(w_, x)
        res = np.array([res[0][0], res[1][0]])
        s = None
        mini = sys.maxsize
        for key, value in threshold.items():
            te = value - res
            distance = math.hypot(te[0], te[1])
            if distance < mini:
                mini = distance
                s = key
        st = '样本属于' + str(s) + "类。"
        print(st)


if __name__ == '__main__':
    _w = get_w()
    data = load_test_data()
    t = get_threshold(_w)
    discriminate(_w, data, t)

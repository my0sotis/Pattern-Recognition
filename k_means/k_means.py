import numpy as np
from sklearn import datasets


def load_data():
    iris = datasets.load_iris()
    iris_data = iris['data']
    iris_data = [np.asarray(iris_data[i]) for i in range(len(iris_data))]
    return iris_data


def k_means():
    k = 3
    data = load_data()
    last = [np.asarray(data[i]) for i in range(k)]
    while True:
        g = [[] for i in range(k)]
        for i in range(len(data)):
            t = [np.linalg.norm(data[i] - last[j]) for j in range(k)]
            g[t.index(min(t))].append(data[i])
        z = [np.mean(g[i], axis=0) for i in range(k)]
        is_eq = True
        for i in range(k):
            if not (z[i] == last[i]).all():
                is_eq = False
                break
        if is_eq:
            break
        else:
            last = z
    for i in range(k):
        print('w' + str(i + 1) + ':')
        for j in range(len(g[i])):
            print(g[i][j])


if __name__ == '__main__':
    k_means()

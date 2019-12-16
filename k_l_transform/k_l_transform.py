import numpy as np


def load_w1():
    with open('w1', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [np.asarray(line.split(','), dtype=np.float) for line in lines if line]
        return lines


def load_w2():
    with open('w2', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        lines = [np.asarray(line.split(','), dtype=np.float) for line in lines if line]
        return lines


def k_l(dataset, target):
    dataset_ = dataset
    num_species = len(dataset_[0])
    mean_v = np.mean(dataset_, axis=0)
    # 坐标系平移
    dataset_ = [(d - mean_v).reshape(1, num_species) for d in dataset_]
    r_x = np.mean([np.dot(d.T, d) for d in dataset_], axis=0)
    value, vector = np.linalg.eig(r_x)
    sorted_indices = np.argsort(-value)
    top_n_vec = vector[:, sorted_indices[0:num_species-1]]
    y = [np.dot(top_n_vec.T, d.reshape(num_species, 1)) for d in dataset_]
    if len(y[0]) != target:
        y = k_l(y, target)
    return y


if __name__ == '__main__':
    print("w1样本集降至2维:")
    print(k_l(load_w1(), 2))
    print("w2样本集降至1维:")
    print(k_l(load_w2(), 1))

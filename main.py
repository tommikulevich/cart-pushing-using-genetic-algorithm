import numpy as np


def state_model(x, u, N):
    model_equations = np.array([
        x[1],
        2 * x[1] - x[0] + (1 / (N ** 2)) * u
    ])
    return model_equations

def criterion(x, u, N):
    return x[0] - (1 / (2 * N)) * np.sum(np.square(u))    # Need to check this

def genetic_algorithm():
    # TODO: choose method and implement it
    pass


if __name__ == '__main__':
    print('[AG Project]')

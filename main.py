#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

SPACE_SIZE = 100
TIME_LENGTH = 100
VIRUS_DECAY = 3
N = 10**2
TIME_STD = 0.2
SPATIAL_STD = 0.3
QUERY_WINDOW_SIZE = 5
MAX_DENSITY = 10


def test_algorithm(x, normal_mechanism, noisy_mechanism):
    random_queries = np.array([])
    errors = []
    for query in random_queries:
        total_error.append(abs(normal_mechanism(query) - noisy_mechanism(query)))
    accuracy = sum(errors) / len(random_queries)
    return random_queries, errors, accuracy


def create_normal_release_mechanism(x):
    return lambda query: release(query, x)


def create_release_mechanism(x, epsilon):
    """
    Creates private query function

    Inputs
        x: unmodified dataset
        epsilon: privacy parameter
    """
    # calculate laplace noise for each column
    #   global sensitivity is 100/n
    temporal_noise = np.random.laplace(scale=TIME_LENGTH / (N * epsilon), size=N)
    spatial_noise_x = np.random.laplace(scale=SPACE_SIZE / (N * epsilon), size=N)
    spatial_noise_y = np.random.laplace(scale=SPACE_SIZE / (N * epsilon), size=N)

    noisy_data = x.copy()
    noisy_data[:, 0] += temporal_noise
    noisy_data[:, 1] += spatial_noise_x
    noisy_data[:, 2] += spatial_noise_y

    return lambda query: release(query, noisy_data)


def release(query, x):
    """releases windowed statistic"""
    timestamp = query[0]
    xcoord = query[1]
    ycord = query[2]
    subset = x[(np.abs(timestamp - x[:, 0]) < VIRUS_DECAY)
                & (np.abs(xcoord - x[:, 1]) < QUERY_WINDOW_SIZE)
                & (np.abs(xcoord - x[:, 1]) < QUERY_WINDOW_SIZE)]
    return len(subset) / MAX_DENSITY


def generate_data(size):
    """
    Inputs
        size: number of entries
    Outputs
        nx3 array with tuples -- (timestamp, x coord, y coord)
    """
    timestamps = np.clip(TIME_LENGTH * (1 - np.random.exponential(scale=TIME_STD, size=size)),
                         a_min=0, a_max=TIME_LENGTH)
    x_coords = np.clip(SPACE_SIZE * np.random.normal(scale=SPATIAL_STD, size=size),
                       a_min=-SPACE_SIZE, a_max=SPACE_SIZE)
    y_coords = np.clip(SPACE_SIZE * np.random.normal(scale=SPATIAL_STD, size=size),
                       a_min=-SPACE_SIZE, a_max=SPACE_SIZE)
    return np.column_stack((timestamps, x_coords, y_coords))


def visualize_data_in_time(x, time_interval=[0, TIME_LENGTH]):
    for i in range(*time_interval):
        points = x[(x[:, 0] > i - VIRUS_DECAY) & (x[:, 0] < i + VIRUS_DECAY)]
        plt.scatter(points[:, 1], points[:, 2])
        plt.xlim(-SPACE_SIZE, SPACE_SIZE)
        plt.ylim(-SPACE_SIZE, SPACE_SIZE)
        plt.title("Reported Infections\n Timestep = {}".format(i))
        plt.draw()
        plt.pause(.01)
        plt.clf()


def main():
    X = generate_data(size=N)
    visualize_data_in_time(X)
    # query_mechanism = create_release_mechanism(X)
    # queries, accuracies, accuracy_tot = test_algorithm(query_mechanism)
    # create_motion_plot(queries, X, accuracies)


if __name__ == '__main__':
    main()

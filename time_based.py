#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

SPACE_SIZE = 100
TIME_LENGTH = 100
VIRUS_DECAY = 3
N = 10**3
TIME_STD = 0.1
SPATIAL_STD = 0.3
QUERY_WINDOW_SIZE = 5
MAX_DENSITY = 100
MIN_SUBSET_SIZE = 5


def test_algorithm(x, normal_mechanism, noisy_mechanism, test_size=N):
    timestamps = np.arange(test_size)
    x_coords = np.clip(SPACE_SIZE * np.random.normal(scale=SPATIAL_STD, size=test_size),
                       a_min=-SPACE_SIZE, a_max=SPACE_SIZE)
    y_coords = np.clip(SPACE_SIZE * np.random.normal(scale=SPATIAL_STD, size=test_size),
                       a_min=-SPACE_SIZE, a_max=SPACE_SIZE)
    random_queries = np.column_stack((timestamps, x_coords, y_coords))
    errors = []
    for query in random_queries:
        baseline = normal_mechanism(query)
        estimate = noisy_mechanism(query)
        if estimate is not None:
            total_error.append(abs(baseline - estimate))
        else:
            total_error.append(0)
    accuracy = sum(errors) / len(errors)
    return random_queries, errors, accuracy


def create_normal_release_mechanism(x):
    return lambda query: release(query, x)


def create_noisy_release_mechanism(x, epsilon=0.01):
    """
    Creates private query function

    Inputs
        x: unmodified dataset
        epsilon: privacy parameter
    """
    # calculate laplace noise for each column
    #   global sensitivity is 100/n (i.e. when timelength 100)
    temporal_noise = np.random.laplace(scale=TIME_LENGTH / (N * epsilon), size=N)
    spatial_noise_x = np.random.laplace(scale=SPACE_SIZE / (N * epsilon), size=N)
    spatial_noise_y = np.random.laplace(scale=SPACE_SIZE / (N * epsilon), size=N)

    noisy_data = x.copy()
    noisy_data[:, 0] += temporal_noise
    noisy_data[:, 1] += spatial_noise_x
    noisy_data[:, 2] += spatial_noise_y

    return lambda query: release(query, noisy_data, protect_low_bound=True)


def release(query, x, protect_low_bound=False):
    """
    Releases windowed statistic based on density of cases within certain distance
       and time contstraints

    Inputs
        query: (timestamp, xcoord, ycoord)
        x: reported infections dataset
    Output
        float: percentage of infection density (proxy to risk of infection)
           None if small set protection enabled and subset is too small
    """
    timestamp = query[0]
    xcoord = query[1]
    ycord = query[2]
    subset = x[(np.abs(timestamp - x[:, 0]) < VIRUS_DECAY)
                & (np.abs(xcoord - x[:, 1]) < QUERY_WINDOW_SIZE)
                & (np.abs(xcoord - x[:, 1]) < QUERY_WINDOW_SIZE)]
    if protect_low_bound and (len(subset) < MIN_SUBSET_SIZE):
        return None
    else:
        return len(subset) / MAX_DENSITY


def generate_data():
    """
    Inputs
        size: number of entries
    Outputs
        nx3 array with tuples -- (timestamp, x coord, y coord)
    """
    timestamps = np.clip(TIME_LENGTH * (1 - np.random.exponential(scale=TIME_STD, size=N)),
                         a_min=0, a_max=TIME_LENGTH)
    x_coords = np.clip(SPACE_SIZE * np.random.normal(scale=SPATIAL_STD, size=N),
                       a_min=-SPACE_SIZE, a_max=SPACE_SIZE)
    y_coords = np.clip(SPACE_SIZE * np.random.normal(scale=SPATIAL_STD, size=N),
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
        plt.pause(.001)
        plt.clf()


def main():
    X = generate_data()
    # visualize_data_in_time(X)
    normal_release_mechanism = create_normal_release_mechanism(X)
    noisy_release_mechanism = create_noisy_release_mechanism(X)
    print("Actual", normal_release_mechanism([95, 0, 0]))
    print("Noisy", noisy_release_mechanism([95, 0, 0]))
    # queries, accuracies, accuracy_tot = test_algorithm(query_mechanism)


if __name__ == '__main__':
    main()

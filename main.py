#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

SPACE_SIZE = 128
N = 2**10
SPATIAL_STD = 0.3
QUERY_WINDOW_SIZE = 4
MIN_RELEASE_SIZE = 3
HIST_BINS = 32


def create_multigranular_mechanism(x, epsilon=0.01):
    # TODO:
    levels = []
    for i in range(np.log2(HIST_BINS)):
        bins = 2**(i + 1)
        values = noisy_hist(x, epsilon, bins)
        levels.append(values)

    def special_release(query):
        ans = 0
        for i in range(len(levels)):
            histogram = levels[i - 1]
            count = release_bin(query, histogram)
            if count >= MIN_RELEASE_SIZE:
                pass
            else:
                break


def create_simple_hist_mechanism(x, epsilon=0.01):
    values = noisy_hist(x, epsilon, bins=HIST_BINS)
    values = np.around(np.clip(values, a_min=0, a_max=None))
    mask = (values >= MIN_RELEASE_SIZE).astype(int)
    values = np.multiply(values, mask)
    return lambda query: release_bin(query, values), values


def noisy_hist(x, epsilon, bins):
    data = x.copy()
    data = data + SPACE_SIZE
    values = np.zeros([bins, bins])
    for coord in data:
        xbin = int(np.floor(coord[0] / (2 * SPACE_SIZE / bins)))
        ybin = int(np.floor(coord[1] / (2 * SPACE_SIZE / bins)))
        values[xbin, ybin] += 1
    return values + np.random.laplace(scale=1 / (epsilon * N), size=[bins, bins])


def create_windowed_release_mechanism(x):
    return lambda query: release_window(query, x)


def create_noisy_windowed_release_mechanism(x, epsilon=0.01):
    """
    Creates private query function

    Inputs
        x: unmodified dataset
        epsilon: privacy parameter
    """
    # calculate laplace noise for each column
    #   global sensitivity is 100/n (i.e. when timelength 100)
    spatial_noise_x = np.random.laplace(scale=SPACE_SIZE / (N * epsilon), size=N)
    spatial_noise_y = np.random.laplace(scale=SPACE_SIZE / (N * epsilon), size=N)

    noisy_data = x.copy()
    noisy_data[:, 0] += spatial_noise_x
    noisy_data[:, 1] += spatial_noise_y

    return lambda query: release_window(query, noisy_data, protect_low_bound=True), noisy_data


def release_window(query, x, protect_low_bound=False):
    """
    Releases windowed statistic based on density of cases within certain distance
       and time contstraints

    Inputs
        query: (xcoord, ycoord)
        x: reported infections dataset
    Output
        float: percentage of infection density (proxy to risk of infection)
           None if small set protection enabled and subset is too small
    """
    xcoord = query[0]
    ycoord = query[1]
    subset = x[(np.abs(xcoord - x[:, 0]) < QUERY_WINDOW_SIZE)
               & (np.abs(ycoord - x[:, 1]) < QUERY_WINDOW_SIZE)]
    if protect_low_bound and (len(subset) < MIN_RELEASE_SIZE):
        return 0
    else:
        return len(subset)


def release_bin(query, histogram):
    bin_loc_x = int(np.floor(query[0] / (2 * SPACE_SIZE / HIST_BINS)))
    bin_loc_y = int(np.floor(query[1] / (2 * SPACE_SIZE / HIST_BINS)))
    count = histogram[bin_loc_x, bin_loc_y]
    return count


def generate_data1():
    """
    Inputs
        size: number of entries
    Outputs
        nx2 array with tuples -- (x coord, y coord)
    """
    x_coords = np.clip(SPACE_SIZE * np.random.normal(scale=SPATIAL_STD, size=N),
                       a_min=-SPACE_SIZE + 0.0001, a_max=SPACE_SIZE - 0.0001)
    y_coords = np.clip(SPACE_SIZE * np.random.normal(scale=SPATIAL_STD, size=N),
                       a_min=-SPACE_SIZE + 0.0001, a_max=SPACE_SIZE - 0.0001)
    return np.column_stack((x_coords, y_coords))


def generate_data2():
    """
    Inputs
        size: number of entries
    Outputs
        nx2 array with tuples -- (x coord, y coord)
    """
    x_coords1 = np.clip(SPACE_SIZE * np.random.normal(loc=0.5, scale=SPATIAL_STD / 2, size=int(N / 2)) - 15,
                        a_min=-SPACE_SIZE + 0.0001, a_max=SPACE_SIZE - 0.0001)
    y_coords1 = np.clip(SPACE_SIZE * np.random.normal(loc=0.5, scale=SPATIAL_STD / 2, size=int(N / 2)) - 15,
                        a_min=-SPACE_SIZE + 0.0001, a_max=SPACE_SIZE - 0.0001)
    x_coords2 = np.clip(SPACE_SIZE * np.random.normal(loc=-0.5, scale=SPATIAL_STD / 2, size=int(N / 2)) + 15,
                        a_min=-SPACE_SIZE + 0.0001, a_max=SPACE_SIZE - 0.0001)
    y_coords2 = np.clip(SPACE_SIZE * np.random.normal(loc=-0.5, scale=SPATIAL_STD / 2, size=int(N / 2)) + 50,
                        a_min=-SPACE_SIZE + 0.0001, a_max=SPACE_SIZE - 0.0001)
    pop1 = np.column_stack((x_coords1, y_coords1))
    pop2 = np.column_stack((x_coords2, y_coords2))
    out = np.concatenate((pop1, pop2), axis=0)
    return out


def calc_dataset_distance(a, b):
    # since N is large, this measure works okay
    x1 = a.copy() + SPACE_SIZE
    x2 = b.copy() + SPACE_SIZE

    diff = len(x2) - len(x1)
    zeros = np.zeros([int(np.abs(diff)), 2])
    if diff < 0:
        x2 = np.concatenate((x2, zeros), axis=0)
    elif diff > 0:
        x1 = np.concatenate((x1, zeros), axis=0)

    xsim = (x1[:, 0] @ x2[:, 0].T) / (np.linalg.norm(x1[:, 0]) * np.linalg.norm(x2[:, 0]))
    ysim = (x1[:, 1] @ x2[:, 1].T) / (np.linalg.norm(x1[:, 1]) * np.linalg.norm(x2[:, 1]))
    return (xsim + ysim) / 2


def hist_2_fake_data(x):
    out = np.zeros([int(np.sum(x)), 2])
    scale_factor = 2 * SPACE_SIZE / HIST_BINS
    loc = 0
    for i in range(HIST_BINS):
        for j in range(HIST_BINS):
            count = int(x[i][j])
            if count > 0:
                random_x_coords = np.random.uniform(low=i * scale_factor, high=(i + 1) * scale_factor, size=count)
                random_y_coords = np.random.uniform(low=j * scale_factor, high=(j + 1) * scale_factor, size=count)
                fake_data = np.column_stack((random_x_coords, random_y_coords))
                out[loc:loc + len(fake_data), :] = fake_data
                loc = loc + len(fake_data)
    out = out - SPACE_SIZE
    return out


def visualize_hist(x, title=""):
    plt.figure()
    plt.imshow(np.rot90(x, 1), extent=[-SPACE_SIZE, SPACE_SIZE, -SPACE_SIZE, SPACE_SIZE], aspect=1)
    plt.title(title)
    plt.colorbar(pad=0.06)
    plt.axis('off')


def visualize_datapoints(x, title=""):
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], s=5)
    plt.xlim(-SPACE_SIZE, SPACE_SIZE)
    plt.ylim(-SPACE_SIZE, SPACE_SIZE)
    plt.title(title)
    plt.axis('off')


def visualize_data_density_map(x, title=""):
    plt.figure()
    (h, _, _, _) = plt.hist2d(x[:, 0], x[:, 1], bins=HIST_BINS,
                              range=[[-SPACE_SIZE, SPACE_SIZE], [-SPACE_SIZE, SPACE_SIZE]])
    plt.title(title)
    plt.colorbar(pad=0)
    plt.axis('off')
    plt.xlim(-SPACE_SIZE, SPACE_SIZE)
    plt.ylim(-SPACE_SIZE, SPACE_SIZE)
    plt.axis('equal')
    return h


def test_accuracy(mechanism, true_hist, num_queries=200):
    x_coords = np.clip(SPACE_SIZE * np.random.uniform(size=num_queries),
                       a_min=-SPACE_SIZE + 0.1, a_max=SPACE_SIZE - 0.1)
    y_coords = np.clip(SPACE_SIZE * np.random.uniform(size=num_queries),
                       a_min=-SPACE_SIZE + 0.1, a_max=SPACE_SIZE - 0.1)
    queries = np.column_stack((x_coords, y_coords))
    error = 0
    count = 0
    for query in queries:
        model_output = mechanism(query)
        if model_output > 0:
            true_value = release_bin(query, true_hist)
            error += np.abs(true_value - model_output)
            count += 1
    return error / count


def main():
    X = generate_data2()

    visualize_datapoints(X, "Baseline Data")
    baseline_bins = visualize_data_density_map(X, "Baseline Data")

    normal_release_mechanism = create_windowed_release_mechanism(X)

    noisy_release_mechanism, X_noisy = create_noisy_windowed_release_mechanism(X)
    visualize_datapoints(X_noisy, "Noisy Data")
    noisy_data_bins = visualize_data_density_map(X_noisy, "Noisy Data")

    print("Noisy dataset distance:", calc_dataset_distance(X, X_noisy))

    simple_hist_release_mechanism, simple_hist_bins = create_simple_hist_mechanism(X)
    visualize_hist(simple_hist_bins, "Simple Noisy Histogram")
    fake_simple = hist_2_fake_data(simple_hist_bins)
    visualize_datapoints(fake_simple, "Simple Noisy Histogram")

    print("Noisy simple histogram distance:", calc_dataset_distance(X, fake_simple))

    # multigranular_hist_release_mechanism, multbins = create_multigranular_mechanism(X)
    # visualize_hist(multibins, "Multilevel Histogram")
    # fake_multilevel = hist_2_fake_data(multibins)
    # visualize_datapoints(fake_multilevel, title="Multilevel Histogram")

    noisy_data_avg_err = test_accuracy(noisy_release_mechanism, baseline_bins)
    print("Noisy data average error:", noisy_data_avg_err)

    simple_hist_avg_err = test_accuracy(simple_hist_release_mechanism, baseline_bins)
    print("Noisy simple histogram average error:", simple_hist_avg_err)

    plt.show()


if __name__ == '__main__':
    main()

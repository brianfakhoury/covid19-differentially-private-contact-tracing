#!/usr/bin/env python3
import numpy as np
import seaborn as sns


def main():
    X = generate_data(size=10**3)
    query_mechanism = create_release_mechanism()
    queries, accuracies, accuracy_tot = test_algorithm(query_mechanism, X)
    create_motion_plot(queries, X, accuracies)


if __name__ == '__main__':
    main()

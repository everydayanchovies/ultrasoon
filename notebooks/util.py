import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import hilbert

DF_PART_A = "A"
DF_PART_B = "B"


def x_y_pairs_for_csv(path):
    df = pd.read_csv(path, delimiter='\t')
    df.columns = ["x", "y"]
    return np.array(df["x"]), np.array(df["y"])


def x_y_pairs_couple_for_params(salinity, v=False):
    def avg_x_y_pairs(signal_part):
        w_x = []
        w_avg_y = []
        for i in range(1, 4):
            filepath = f"../data/{str(salinity).strip('0').strip('.') or '0'} promille zout {signal_part} ({i})"
            x, y = x_y_pairs_for_csv(filepath)
            if len(w_x) == 0:
                w_x = np.array(x)
                w_avg_y = np.array(y)
            else:
                w_avg_y = np.average([w_avg_y, np.array(y)], axis=0)
        return w_x, w_avg_y

    def envelope_for_x_y_pairs(x, y):
        y = y - np.mean(y)
        y_env = np.abs(hilbert(y))

        if v:
            plt.plot(x, y)
            plt.plot(x, y_env)
        return x, y_env

    a = avg_x_y_pairs(DF_PART_A)
    a = envelope_for_x_y_pairs(*a)

    b = avg_x_y_pairs(DF_PART_B)
    b = envelope_for_x_y_pairs(*b)

    if v:
        plt.plot(*a)
        plt.plot(*b)
        plt.show()

    return *a, *b


def index_of_next_peak(y):
    rms = np.sqrt(np.mean(y ** 2))

    window_len = 1

    last_max = y[0]
    count_since_max = 0
    for i in range(len(y)):
        if i < window_len:
            continue

        if y[i] > last_max:
            last_max = y[i]
            count_since_max = 0

        if count_since_max > 50 and last_max > rms:
            return i - count_since_max

        count_since_max += 1

    raise ValueError


def index_of_next_min(y):
    window_len = 1

    last_min = y[0]
    count_since_min = 0
    for i in range(len(y)):
        if i < window_len:
            continue

        if y[i] < last_min:
            last_min = y[i]
            count_since_min = 0

        if count_since_min > 50:
            return i - count_since_min

        count_since_min += 1

    raise ValueError


def eat_peak(x, y, n=1, v=False):
    """
    Will remove everything right of the highest peak.
    """

    next_min_i = index_of_next_min(y)

    max_i = index_of_next_peak(y[next_min_i:]) + next_min_i
    leftovers = x[max_i:], y[max_i:]

    if v:
        plt.plot(x, y, alpha=1/n)
        plt.plot(*leftovers)

    if n > 1:
        return eat_peak(*leftovers, n - 1, v=v)

    if v:
        plt.show()

    return x[max_i], y[max_i], leftovers


def delay_for_x_y_pairs_couple(x_a, y_a, x_b, y_b, v=False):
    peak_a_x, peak_a_y, _ = eat_peak(x_a, y_a, n=2, v=v)
    peak_b_x, peak_b_y, _ = eat_peak(x_b, y_b, n=1, v=v)

    if v:
        plt.plot(x_a, y_a)
        plt.plot(x_b, y_b)
        plt.hlines(min(peak_a_y, peak_b_y), peak_a_x, peak_b_x, colors='r')
        plt.xlabel(r"Time [ms]")
        plt.ylabel(r"Amplitude [V]")
        plt.show()

    return peak_b_x - peak_a_x

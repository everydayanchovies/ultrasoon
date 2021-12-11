import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import hilbert

DF_PART_A = "A"
DF_PART_B = "B"


def x_y_pairs_for_csv(path):
    """
    Leest een csv bestand op en zet het om in een dataframe.

    :param path: het pad van de csv bestand om te lezen
    :return: een dataframe
    """
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
        y = np.abs(hilbert(y))
        return x, y

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
    :param n:
    :param x:
    :param y:
    :param v:
    :return:
    """

    next_min_i = index_of_next_min(y)

    max_i = index_of_next_peak(y[next_min_i:]) + next_min_i
    max_x = x[max_i]
    right_x, right_y = x[max_i:], y[max_i:]

    if v:
        plt.plot(x, y, linestyle="dashed", alpha=0.4)
        plt.plot(right_x, right_y, alpha=1)

    if n <= 1:
        if v:
            plt.show()
        return max_x, (right_x, right_y)

    return eat_peak(right_x, right_y, n - 1, v=v)


def delay_for_x_y_pairs_couple(x_a, y_a, x_b, y_b, v=False):
    peak_a, _ = eat_peak(x_a, y_a, v=v)

    peak_b, _ = eat_peak(x_b, y_b, n=2, v=v)

    if v:
        plt.plot(x_a, y_a)
        plt.plot(x_b, y_b)
        plt.hlines(50, peak_a, peak_b, colors='r')
        plt.show()

    return peak_b - peak_a

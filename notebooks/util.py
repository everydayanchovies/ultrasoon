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
                # todo this is fishy maybe?
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


def index_of_next_peak(y, v=False):
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

        if count_since_max > 100 and last_max > rms:
            return i - count_since_max

        count_since_max += 1

    print("ERROR")


def index_of_next_min(y, v=False):
    near_min_y = np.argmin(y) * 4

    window_len = 1

    last_min = y[0]
    count_since_min = 0
    for i in range(len(y)):
        if i < window_len:
            continue

        if y[i] < last_min:
            last_min = y[i]
            count_since_min = 0

        if count_since_min > 100:
            return i - 100

        count_since_min += 1


def eat_peak(x, y, v=False):
    """
    Will remove everything right of the highest peak.
    :param x:
    :param y:
    :param v:
    :return:
    """

    next_min_i = index_of_next_min(y, v=v)

    max_i = index_of_next_peak(y[next_min_i:]) + next_min_i
    max_x = x[max_i]
    right_x, right_y = x[max_i:], y[max_i:]

    if v:
        plt.plot(x, y, 'r-', alpha=0.4)
        plt.plot(right_x, right_y, alpha=1)
        plt.show()

    return max_x, (right_x, right_y)


def delay_for_x_y_pairs_couple(x_a, y_a, x_b, y_b, v=False):
    peak_a, _ = eat_peak(x_a, y_a, v=v)

    _, right = eat_peak(x_b, y_b, v=v)
    peak_b, _ = eat_peak(*right, v=v)

    return peak_b - peak_a

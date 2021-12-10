import numpy as np
import pandas as pd
import matplotlib
from scipy import signal
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
    # x_a, y_a = a

    b = avg_x_y_pairs(DF_PART_B)
    b = envelope_for_x_y_pairs(*b)
    # x_b, y_b = b

    if v:
        plt.plot(*a)
        plt.plot(*b)
        plt.show()

    return *a, *b


def delay_for_x_y_pairs_couple(x_a, y_a, x_b, y_b, v=False):
    pass



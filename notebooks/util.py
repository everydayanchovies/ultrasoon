import numpy as np
import pandas as pd
import matplotlib
from scipy import signal
from matplotlib import pyplot as plt
from scipy.signal import hilbert

DF_PART_A = "A"
DF_PART_B = "B"

def df_for_csv(path):
    """
    Leest een csv bestand op en zet het om in een dataframe.

    :param path: het pad van de csv bestand om te lezen
    :return: een dataframe
    """
    df = pd.read_csv(path, delimiter='\t')
    df.columns = ["x", "y"]
    return df


def df_for_params(salinity, v=False):
    def avg_df(signal_part):
        df = None
        w_x = []
        w_avg_y = []
        for i in range(1, 4):
            filepath = f"../data/{salinity} promille zout {signal_part} ({i})"
            df = df_for_csv(filepath)
            if len(w_x) == 0:
                w_x = np.array(df["x"])
                w_avg_y = np.array(df["y"])
            else:
                # todo this is fishy
                w_avg_y = np.average([w_avg_y, np.array(df["x"])], axis=0)
        df["y"] = w_avg_y
        return df

    def envelope_for_df(df):
        y = df["y"]
        y = y - np.mean(y)
        df["y"] = np.abs(hilbert(y))
        return df

    df_a = avg_df(DF_PART_A)
    df_a = envelope_for_df(df_a)
    df_b = avg_df(DF_PART_B)
    df_b = envelope_for_df(df_b)

    df = pd.concat([df_a, df_b], axis=1)
    df.columns = ["Ax", "Ay", "Bx", "By"]

    if v:
        plt.plot(df["Ax"], df["Ay"])
        plt.plot(df["Bx"], df["By"])
        plt.show()
    return df


def delay_for_df(df, v=False):
    Ax = np.array(df["Ax"])
    Ay = np.array(df["Ay"])
    Bx = np.array(df["Bx"])
    By = np.array(df["By"])

    peaks_a = signal.find_peaks_cwt(Ay, 20)
    peaks_b = signal.find_peaks_cwt(By, 0.1)

    if v:
        print(peaks_a)
        print(peaks_b)

        plt.plot(Ax, Ay)
        plt.vlines([Ax[i] for i in peaks_a], 0, 200)
        plt.show()



    return np.abs(Ax[peaks_a[1]] - Bx[peaks_b[0]])



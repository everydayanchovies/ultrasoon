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
    df = pd.read_csv(path, delimiter='\t', skiprows=70)
    df.columns = ["x", "y"]
    return df


def df_for_params(salinity, v=False):
    def avg_df(signal_part):
        df = None
        w_x = []
        w_avg_y = []
        for i in range(1, 4):
            filepath = f"../data/{str(salinity).strip('0').strip('.') or '0'} promille zout {signal_part} ({i})"
            df = df_for_csv(filepath)
            if len(w_x) == 0:
                w_x = np.array(df["x"])
                w_avg_y = np.array(df["y"])
            else:
                # todo this is fishy
                w_avg_y = np.average([w_avg_y, np.array(df["y"])], axis=0)
        df["y"] = w_avg_y
        return df

    def noise_reduction_for_df(df):
        y = df["y"]
        n = 3  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        y = signal.lfilter(b, a, y)

        df["y"] = y
        return df

    def envelope_for_df(df):
        y = df["y"]
        y = y - np.mean(y)
        df["y"] = np.abs(hilbert(y))
        return df

    def derivative_for_df(df, v=False):
        dx = 1
        y = np.array(df["y"])
        dy = np.array(np.diff(y) / dx)
        dy = np.append(dy, 0)

        for i in range(len(dy)):
            if np.abs(dy[i]) > 5:
                dy[i] = 0
        if v:
            plt.plot(df["x"], dy)
            plt.show()

        df["dy"] = dy ** 3
        return df

    df_a = avg_df(DF_PART_A)
    df_a = noise_reduction_for_df(df_a)
    df_a = envelope_for_df(df_a)
    df_a = derivative_for_df(df_a, v=v)

    df_b = avg_df(DF_PART_B)
    df_b = noise_reduction_for_df(df_b)
    df_b = envelope_for_df(df_b)
    df_b = derivative_for_df(df_b, v=v)

    df = pd.concat([df_a, df_b], axis=1)
    df.columns = ["Ax", "Ay", "Ady", "Bx", "By", "Bdy"]

    if v:
        plt.plot(df["Ax"], df["Ay"])
        plt.plot(df["Bx"], df["By"])
        plt.show()
    return df


def delay_for_df(df, v=False):
    Ax = np.array(df["Ax"])
    Ay = np.array(df["Ady"])
    Bx = np.array(df["Bx"])
    By = np.array(df["Bdy"])

    s = 40
    noise_p = 20
    min_snr = 3
    peaks_a = signal.find_peaks_cwt(Ay, s, noise_perc=noise_p, min_snr=min_snr)
    peaks_b = signal.find_peaks_cwt(By, s, noise_perc=noise_p, min_snr=min_snr)

    if v:
        print(peaks_a)
        print(peaks_b)

        plt.plot(Ax, Ay)
        plt.vlines([Ax[i] for i in peaks_a], 0, 200)
        plt.plot(Bx, By)
        plt.vlines([Bx[i] for i in peaks_b], 0, 200)
        plt.ylim(-7, 7)
        plt.show()



    return np.abs(Ax[peaks_a[2]] - Bx[peaks_b[1]])



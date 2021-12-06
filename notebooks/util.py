import numpy as np
import pandas as pd
import matplotlib
import scipy as scipy
from matplotlib import pyplot as plt


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
    #todo take avg
    filepath_a = f"../data/{salinity} promille zout A (1)"
    filepath_b = f"../data/{salinity} promille zout B (1)"

    df_a = df_for_csv(filepath_a)
    df_b = df_for_csv(filepath_b)

    df = pd.concat([df_a, df_b], axis=1)
    df.columns = ["Ax", "Ay", "Bx", "By"]

    if v:
        plt.plot(df["Ax"], df["Ay"])
        plt.plot(df["Bx"], df["By"])
        plt.show()
    return df


def delay_for_df(df):
    Ax = np.array(df["Ax"])
    Ay = np.array(df["Ay"])
    Bx = np.array(df["Bx"])
    By = np.array(df["By"])

    peaks_a = scipy.

    x_at_max_amp_a = Ax[np.argmax(Ay)]
    x_at_max_amp_b = Bx[np.argmax(By)]

    return np.abs(x_at_max_amp_a - x_at_max_amp_b)



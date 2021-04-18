from scipy.io import loadmat
import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt


if __name__ == "__main__":
    channel_model = "VehA"
    SNR = 12
    number_of_pilot = 48
    val_split = 1 / 9
    perfect = loadmat("Perfect_" + 'channel_model.mat')['My_perfect_H']
    noisy = loadmat("Noisy_" + channel_model + "_" + "SNR_" + str(SNR) + ".mat")['My_noisy_H']
    noisy_image = np.zeros((40000, 72, 14, 2))

    noisy_image[:, :, :, 0] = np.real(noisy)
    noisy_image[:, :, :, 1] = np.imag(noisy)

    if number_of_pilot == 48:
        idx = [14 * i for i in range(1, 72, 6)] + [4 + 14 * i for i in range(4, 72, 6)] + [7 + 14 * i for i in
                                                                                           range(1, 72, 6)] + [
                  11 + 14 * i for i in range(4, 72, 6)]
    elif number_of_pilot == 16:
        idx = [4 + 14 * i for i in range(1, 72, 9)] + [9 + 14 * i for i in range(4, 72, 9)]
    elif number_of_pilot == 24:
        idx = [14 * i for i in range(1, 72, 9)] + [6 + 14 * i for i in range(4, 72, 9)] + [11 + 14 * i for i in
                                                                                           range(1, 72, 9)]
    elif number_of_pilot == 8:
        idx = [4 + 14 * i for i in range(5, 72, 18)] + [9 + 14 * i for i in range(8, 72, 18)]
    elif number_of_pilot == 36:
        idx = [14 * i for i in range(1, 72, 6)] + [6 + 14 * i for i in range(4, 72, 6)] + [11 + 14 * i for i in
                                                                                           range(1, 72, 6)]

    r = [x // 14 for x in idx]
    c = [x % 14 for x in idx]

    print((r, c))

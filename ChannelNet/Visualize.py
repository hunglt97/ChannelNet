import numpy
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import correlate
from models import interpolation, psnr, pilot_comb
import math

if __name__ == "__main__":

    #data1 = numpy.load('SNR_22.npz')
    data2 = numpy.load('SNR_12.npz')
    # data3 = numpy.load('LS_ce.npz')
    # data4 = numpy.load('MMSE_ce.npz')

    # plot the test
    # plt.plot(data1['x1'], data1['y1'], 'b.-', label='High SNR model')
    plt.plot(data2['x2'], data2['y2'], 'r.-', label='Low SNR model')
    # plt.plot(data3['x3'], data3['y3'], 'g.-', label='LS estimator')
    # plt.plot(data4['x4'], data4['y4'], 'y.-', label='MMSE estimator')
    plt.legend()
    plt.xlabel('SNR')
    plt.ylabel('Peak-SNR')
    plt.title("Channel Estimation PSNR in term of SNR for VehA channel model.")
    plt.grid(True)
    plt.show()



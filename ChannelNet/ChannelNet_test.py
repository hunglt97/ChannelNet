import numpy
import math
from models import interpolation, final_test, mse, pilot_comb, no_int_test
from scipy.io import loadmat, savemat
from scipy.signal import correlate
import matplotlib.pyplot as plt

if __name__ == "__main__":
    channel_model = "VehA"
    Number_of_pilots = 48
    perfect = loadmat("Perfect_" + 'channel.mat')['My_perfect_H']
    perfect_image = numpy.zeros((len(perfect), 72, 14, 2))
    perfect_image[:, :, :, 0] = numpy.real(perfect)
    perfect_image[:, :, :, 1] = numpy.imag(perfect)
    perfect_image = numpy.concatenate((perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]), axis=0).reshape(
        2 * len(perfect), 72, 14, 1)

    # test with different SNRs
    # SNR = 22
    # x1 = numpy.zeros(6, dtype=int)
    # y1 = numpy.zeros(6, dtype=int)
    # for SNR_test in range(6):
    #     x1[SNR_test] = (1+SNR_test)*5
    #     y1[SNR_test] = final_test(x1[SNR_test], channel_model, Number_of_pilots, SNR, perfect_image)
    # numpy.savez('SNR_22.npz', x1=x1, y1=y1)

    SNR = 12
    x2 = numpy.zeros(6, dtype=int)
    y2 = numpy.zeros(6, dtype=int)
    for SNR_test in range(6):
        x2[SNR_test] = (3+SNR_test)*2
        y2[SNR_test] = no_int_test(x2[SNR_test], channel_model, Number_of_pilots, SNR, perfect_image)
        savemat("SNR_" + str(SNR) + ".mat", y2[SNR_test])
    numpy.savez('SNR_12.npz', x2=x2, y2=y2)


    # # LS estimation
    # x3 = numpy.zeros(6, dtype=int)
    # y3 = numpy.zeros(6, dtype=int)
    # for SNR_test in range(6):
    #     x3[SNR_test] = (1 + SNR_test) * 5
    #     test_input = loadmat("Noisy_" + channel_model + "_" + "SNR_" + str(x3[SNR_test]) + ".mat")['Y']
    #     y3_val = interpolation(test_input, SNR_test, Number_of_pilots, 'spline')
    #     y3[SNR_test] = psnr(y3_val, perfect_image)
    # numpy.savez('LS_ce.npz', x3=x3, y3=y3)
    #
    # # MMSE estimation
    # perfect_image_flat = perfect_image.reshape(2 * len(perfect), 1008)
    # x4 = numpy.zeros(6, dtype=int)
    # y4 = numpy.zeros(6, dtype=int)
    # for SNR_test in range(6):
    #     x4[SNR_test] = (1 + SNR_test) * 5
    #     snr = 10 ** (x4[SNR_test] * 0.1)
    #     test_input = loadmat("Noisy_" + channel_model + "_" + "SNR_" + str(x4[SNR_test]) + ".mat")['Y']
    #     test_flat = test_input.reshape(40000, 1008)
    #     y4_val = pilot_comb(test_flat, x4[SNR_test], Number_of_pilots)
    #     Nfft = 72 * 14
    #     Nps = Nfft / Number_of_pilots
    #     H_MMSE = numpy.zeros((40000, 1008), dtype=complex)
    #     for i in range(len(test_input)):
    #         K1 = test_flat[i, :].reshape(Nfft, 1)
    #         K2 = y4_val[i, :].reshape(Number_of_pilots, 1).conj().T
    #         rf = correlate(K1, K2, "full")
    #         K3 = y4_val[i, :].reshape(Number_of_pilots, 1)
    #         K4 = y4_val[i, :].reshape(Number_of_pilots, 1).conj().T
    #         rf2 = correlate(K3, K4, "full")
    #         Rhp = rf
    #         Rpp = rf2 + numpy.eye(len(y4_val[i, :]), len(y4_val[i, :])) / snr
    #         H_MMSE[i, :] = (Rhp.dot(numpy.linalg.inv(Rpp)).dot(y4_val[i, :].T)).T
    #     H_MMSE_r = numpy.zeros((40000, 1008, 2))
    #     H_MMSE_r[:, :, 0] = numpy.real(H_MMSE)
    #     H_MMSE_r[:, :, 1] = numpy.imag(H_MMSE)
    #     H_MMSE_r = numpy.concatenate((H_MMSE_r[:, :, 0], H_MMSE_r[:, :, 1]), axis=0).reshape(80000, 1008)
    #     y4[SNR_test] = psnr(H_MMSE_r, perfect_image_flat)
    # numpy.savez('MMSE_ce.npz', x4=x4, y4=y4)
    #
    #
    # # # plot the test
    # # plt.plot(x1, y1, 'b.-', label='High SNR model')
    # # plt.plot(x2, y2, 'r.-', label='Low SNR model')
    # # plt.legend()
    # # plt.xlabel('SNR')
    # # plt.ylabel('Peak-SNR')
    # # plt.title("Channel Estimation PSNR in term of SNR for VehA channel model.")
    # # plt.grid(True)
    # # plt.show()
    #
    #

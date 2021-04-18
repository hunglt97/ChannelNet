import numpy
import math
from models import mse, interpolation, SRCNN_train, SRCNN_model, SRCNN_predict, DNCNN_train, DNCNN_model, DNCNN_predict, final_test
# from scipy.misc import imresize
from scipy.io import loadmat
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load datasets 
    channel_model = "VehA"
    SNR = 12
    Number_of_pilots = 48
    val_split = 1/9
    perfect = loadmat("Perfect_" + 'channel.mat')['My_perfect_H']
    noisy_input = loadmat("Noisy_" + "SNR_" + str(SNR) + ".mat")['My_noisy_H']

    interp_noisy = interpolation(noisy_input, SNR, Number_of_pilots, 'rbf')

    perfect_image = numpy.zeros((len(perfect), 72, 14, 2))
    perfect_image[:, :, :, 0] = numpy.real(perfect)
    perfect_image[:, :, :, 1] = numpy.imag(perfect)
    perfect_image = numpy.concatenate((perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]), axis=0).reshape(
        2 * len(perfect), 72, 14, 1)

    ####### ------ training SRCNN ------ #######
    idx_random = numpy.random.rand(len(perfect_image)) < (
            9 / 10)  # uses 32000 from 36000 as training and the rest as validation
    train_data, train_label = interp_noisy[idx_random, :, :, :], perfect_image[idx_random, :, :, :]
    test_data, test_label = interp_noisy[~idx_random, :, :, :], perfect_image[~idx_random, :, :, :]
    SRCNN_train(train_data, train_label, val_split, channel_model, Number_of_pilots, SNR)

    ####### ------ prediction using SRCNN ------ #######
    srcnn_pred_train = SRCNN_predict(train_data, channel_model, Number_of_pilots, SNR)
    srcnn_pred_test = SRCNN_predict(test_data, channel_model, Number_of_pilots, SNR)

    ####### ------ training DNCNN ------ #######
    DNCNN_train(train_data, train_label, val_split, channel_model, Number_of_pilots, SNR)

    ####### ------ prediction using DNCNN ------ #######
    dncnn_pred_train = DNCNN_predict(srcnn_pred_train, channel_model, Number_of_pilots, SNR)
    dncnn_pred_test = DNCNN_predict(srcnn_pred_test, channel_model, Number_of_pilots, SNR)

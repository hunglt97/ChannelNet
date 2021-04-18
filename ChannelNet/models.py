from keras.models import Sequential, Model
from keras.layers import Convolution2D, Input, BatchNormalization, Conv2D, Activation, Lambda, Subtract, \
    Conv2DTranspose, PReLU
from keras.regularizers import l2
from keras.layers import Reshape, Dense, Flatten
from scipy.io import loadmat
import keras.backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt


# from scipy.misc import imresize


def mse(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    mse = np.mean(diff ** 2.)

    return mse


def pilot_comb(noisy_flat, SNR, number_of_pilot):
    pilot = np.zeros((40000, number_of_pilot), dtype=complex)
    for i in range(0, number_of_pilot-1):
        for j in range(0, 1007, number_of_pilot):
            pilot[:, i] = noisy_flat[:, j]
    return pilot


def interpolation(noisy, SNR, number_of_pilot, interp):
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

    interp_noisy = np.zeros((40000, 72, 14, 2))

    for i in range(len(noisy)):
        z = [noisy_image[i, j, k, 0] for j, k in zip(r, c)]
        if (interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z, function='gaussian')
            X, Y = np.meshgrid(range(72), range(14))
            z_intp = f(X, Y)
            interp_noisy[i, :, :, 0] = z_intp.T
        elif (interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72), range(14), tck)
            interp_noisy[i, :, :, 0] = z_intp
        z = [noisy_image[i, j, k, 1] for j, k in zip(r, c)]
        if (interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z, function='gaussian')
            X, Y = np.meshgrid(range(72), range(14))
            z_intp = f(X, Y)
            interp_noisy[i, :, :, 1] = z_intp.T
        elif (interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72), range(14), tck)
            interp_noisy[i, :, :, 1] = z_intp

    interp_noisy = np.concatenate((interp_noisy[:, :, :, 0], interp_noisy[:, :, :, 1]), axis=0).reshape(80000, 72, 14,
                                                                                                        1)

    return interp_noisy


def no_interpol(noisy):
    noisy_image = np.zeros((len(noisy), 72, 14, 2))

    noisy_image[:, :, :, 0] = np.real(noisy)
    noisy_image[:, :, :, 1] = np.imag(noisy)

    no_interp_noisy = np.concatenate((noisy_image[:, :, :, 0], noisy_image[:, :, :, 1]), axis=0).reshape(2*len(noisy), 72, 14,
                                                                                                        1)
    return no_interp_noisy


def SRCNN_model():
    input_shape = (72, 14, 1)
    x = Input(shape=input_shape)
    c1 = Convolution2D(64, 9, 9, activation='relu', init='he_normal', border_mode='same')(x)
    c2 = Convolution2D(32, 1, 1, activation='relu', init='he_normal', border_mode='same')(c1)
    c3 = Convolution2D(1, 5, 5, init='he_normal', border_mode='same')(c2)
    # c4 = Input(shape = input_shape)(c3)
    model = Model(input=x, output=c3)
    ### Compile
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def SRCNN_train(train_data, train_label, val_split, channel_model, num_pilots, SNR):
    srcnn_model = SRCNN_model()
    print(srcnn_model.summary())

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    history = srcnn_model.fit(train_data, train_label, batch_size=128, validation_split=val_split,
                    callbacks=callbacks_list, shuffle=True, epochs=200, verbose=0)

    # # Display curves of loss every epoch
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.ylabel('Validation loss')
    # plt.xlabel('No. epoch')
    # plt.legend()
    # plt.show()

    # srcnn_model.save_weights("drive/codes/my_srcnn/SRCNN_SUI5_weights/SRCNN_48_12.h5")
    srcnn_model.save_weights("SRCNN_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".h5")
    srcnn_model.save("SRCNN_" + "model_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".h5")


def SRCNN_predict(input_data, channel_model, num_pilots, SNR):
    srcnn_model = SRCNN_model()
    srcnn_model.load_weights("SRCNN_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".h5")
    predicted = srcnn_model.predict(input_data)
    return predicted


def DNCNN_model():
    inpt = Input(shape=(None, None, 1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(8):
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Subtract()([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def DNCNN_train(train_data, train_label, val_split, channel_model, num_pilots, SNR):
    dncnn_model = DNCNN_model()
    print(dncnn_model.summary())

    checkpoint = ModelCheckpoint("DNCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    history = dncnn_model.fit(train_data, train_label, batch_size=128, validation_split=val_split,
                    callbacks=callbacks_list, shuffle=True, epochs=10, verbose=0)

    # # Display curves of loss every epoch
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.ylabel('Validation loss')
    # plt.xlabel('No. epoch')
    # plt.legend()
    # plt.show()

    dncnn_model.save_weights("DNCNN_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".h5")
    dncnn_model.save("DNCNN_" + "model_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".h5")


def DNCNN_predict(input_data, channel_model, num_pilots, SNR):
    dncnn_model = DNCNN_model()
    dncnn_model.load_weights("DNCNN_" + channel_model + "_" + str(num_pilots) + "_" + str(SNR) + ".h5")
    predicted = dncnn_model.predict(input_data)
    return predicted


def final_test(SNR_test, channel_model, Number_of_pilots, SNR, perfect_image):
    test_input = loadmat("Noisy_" + "SNR_" + str(SNR_test) + ".mat")['My_noisy_H']
    interp_test = interpolation(test_input, SNR_test, Number_of_pilots, 'rbf')
    srcnn_pred_test_snr = SRCNN_predict(interp_test, channel_model, Number_of_pilots, SNR)
    dncnn_pred_test_snr = DNCNN_predict(srcnn_pred_test_snr, channel_model, Number_of_pilots, SNR)
    return mse(dncnn_pred_test_snr, perfect_image)


def no_int_test(SNR_test, channel_model, Number_of_pilots, SNR, perfect_image):
    test_input = loadmat("Noisy_" + "SNR_" + str(SNR_test) + ".mat")['My_noisy_H']
    interp_test = no_int_test(test_input)
    srcnn_pred_test_snr = SRCNN_predict(interp_test, channel_model, Number_of_pilots, SNR)
    dncnn_pred_test_snr = DNCNN_predict(srcnn_pred_test_snr, channel_model, Number_of_pilots, SNR)
    return mse(dncnn_pred_test_snr, perfect_image)
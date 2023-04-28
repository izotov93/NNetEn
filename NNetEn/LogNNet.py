#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:11:35 2023

@author: Izotov Yuriy
@user: izotov93
"""

import numpy as np
from numba import njit, types
import math

@njit(cache=True, fastmath=True)
def Fout(input : types.float64[:]) -> types.float64[:]:
    """
    Logistic activation function (1 / (1 + exp(-1 * value))
        :param input: The meaning of the neuron
        :return: Neuron value after activation
    """
    return np.divide(1, np.add(1, np.exp(-input)))

@njit(cache=True, fastmath=True)
def back_prop_calc_first_layer(Y, W, Sh_max,
                               Sh_min, Sh_mean):
    """
    Calculation of neurons of the first hidden layer
        :param Y: Input vector
        :param W: Weight Matrix
        :param Sh_max, Sh_min and Sh_mean: Normalization coefficient
        :return: The value of the neurons of one hidden layer
    """
    Sh = np.sum(np.multiply(Y, W), axis=1)
    Sh = np.divide(Sh - Sh_min, Sh_max - Sh_min) \
                    - 0.5 - Sh_mean
    return np.hstack((np.ones(1), Sh)) # bias

@njit(cache=True, fastmath=True)
def back_prop_calc_value(S, W):
    """
    Calculation of neurons in hidden layers (second and further)
        :param S: The value of the neurons of the past hidden layer
        :param W: Weight Matrix
        :return: The value of the neurons of hidden layer
    """
    Sh_next = np.sum(np.multiply(S, W), axis=1)
    return Fout(Sh_next)

@njit(cache=True, fastmath=True)
def weight_training(W, err, Sh, NORMA):
    """
    Weight training function
        :param W: Weight Matrix
        :param err: Error vector computed backpropagation method
        :param Sh: The value of the neurons of hidden layer
        :param NORMA: -
        :return: Weight matrix after backpropagation
    """
    dW = np.empty_like(W)
    dW[:] = Sh * NORMA
    dW2 = dW * err.reshape((-1, 1))
    return np.add(W, dW2)

@njit(cache=True, fastmath=True)
def err_calc(Sh: types.float64[:]) -> types.float64[:]:
    """
    Calculate errors
        :param Sh: The value of the neurons of hidden layer
        :return: Error vector
    """
    return Sh * (1 - Sh)

@njit(cache=True, fastmath=True)
def err_calc_hid_layers(err: types.float64[:], W : types.float64[:]) \
        -> types.float64[:]:
    """
    Error Calculation on Hidden Layers (intermediate function)
        :param err: Error vector
        :param W: Weight Matrix
        :return: Error vector
    """
    return np.sum(np.multiply(err, W.T), axis=1)

@njit(cache=True, fastmath=True)
def err_calc_last_layer(techer, Sh):
    """
    Error Calculation on Last Layer
        :param techer: reference vector
        :param Sh: The value of the neurons of hidden layer
        :return: Error vector
    """
    return (techer - Sh) * err_calc(Sh)

@njit(cache=True, fastmath=True)
def err_calc_rest_layers(Sh, err_prev, W_prev):
    """
    Error Calculation on Hidden Layers
        :param Sh: The value of the neurons of hidden layer
        :param err_prev: The error vector of the previous layer
        :param W_prev: Weight matrix of the previous layer
        :return: Error vector
    """
    return err_calc(Sh) * err_calc_hid_layers(err_prev, W_prev)

# Step 3
@njit(cache=True, fastmath=True)
def formation_W1_coef_time_series(xn_params, W1, method=5):
    """
    Formation of the weight matrix from the time series
        :param xn_params: Time series
        :param W1: Weight matrix
        :param method: One of 6 methods for forming a reservoir matrix
        :return: Weight matrix W1
    """
    Y = W1.shape[1]#conf_layers[0] + 1
    P = W1.shape[0]#conf_layers[1]
    k = 0
    if (method == 1): # line by line doubling
        for i in range(P):
            j = 0
            while (j < Y):
                if (k < xn_params.size):
                    W1[i, j] = xn_params[k]
                else:
                    k = -1
                    j -= 1
                k += 1
                j += 1
    elif (method == 2): # Line by line with zeros
        for i in range(P):
            j = 0
            while (j < Y):
                if (k < xn_params.size):
                    W1[i, j] = xn_params[k]
                else:
                    k = -1
                    j = Y
                k += 1
                j += 1
            if (k >= xn_params.size):
                k = 0
    elif (method == 3): # Line Stretch
        XXk = np.empty_like(xn_params)
        for index in range(XXk.size):
            XXk[index] = ((index/(XXk.size-1)) *
                          ((Y*P)-1))+1
        for i in range(P):
            for j in range(Y):
                z = 0
                k += 1
                while ((z < xn_params.size-1) and
                       (int(XXk[z]) <= k)):
                    z += 1
                if z == 0:
                    z = 1
                x1 = XXk[z - 1]
                x2 = XXk[z]
                y1 = xn_params[z - 1]
                y2 = xn_params[z]
                if x1 == x2:
                    koef = 0
                else:
                    koef = (y1-y2)/(x1-x2)
                b_kof = y1 - koef * x1
                W1[i, j] = koef * k + b_kof
    elif (method == 4):
        for i in range(Y):
            j = 0
            while (j < P):
                if (k < xn_params.size):
                    W1[j, i] = xn_params[k]
                else:
                    k = -1
                    j -= 1
                j += 1
                k += 1
    elif (method == 5):
        for i in range(Y):
            j = 0
            while (j < P):
                if (k < xn_params.size):
                    W1[j, i] = xn_params[k]
                else:
                    j = P
                j += 1
                k += 1
            if (k >= xn_params.size):
                k = 0
    elif (method == 6):
        XXk = np.empty_like(xn_params)
        for index in range(XXk.size):
            XXk[index] = ((index / (XXk.size - 1)) *
                          ((Y * P) - 1)) + 1
        for i in range(Y):
            for j in range(P):
                k += 1
                z = 0
                while ((z < xn_params.size-1) and
                       (int(XXk[z]) <= k)):
                    z += 1
                if z == 0:
                    z = 1
                x1 = XXk[z - 1]
                x2 = XXk[z]
                y1 = xn_params[z - 1]
                y2 = xn_params[z]
                if x1 == x2:
                    koef = 0
                else:
                    koef = (y1-y2)/(x1-x2)
                b_kof = y1 - koef * x1
                W1[j, i] = koef * k + b_kof

    return W1

# Step 4
@njit(cache=True, fastmath=True)
def calculation_min_max_params(database, W1):
    """
    Calculation of minimum and maximum normalization vectors
        :param database: database
        :param W1: Weight matrix W1
        :return: minimum and maximum normalization vectors
    """
    Sh_max = np.zeros(W1.shape[0], dtype=np.float64)  # P
    Sh_min = np.empty(W1.shape[0], dtype=np.float64)  # P
    Sh_min[:] = 100
    for data in database:
        # We multiply element by element Y by the transposed matrix W
        # We get the matrix and find the sum along axis 1
        Sh = np.sum(np.multiply(data, W1), axis=1)

        Sh_max = np.maximum(Sh, Sh_max)
        Sh_min = np.minimum(Sh, Sh_min)
    return Sh_max, Sh_min

#Step 5
#@njit(cache=True, fastmath=True)
def calculation_normalization_params(database, W1):
    """
    Calculation normalization params
        :param database: Normalized database
        :param W1: Weight matrix W1
        :return: minimum, maximum and mean normalization vectors
    """
    Sh_max, Sh_min = calculation_min_max_params(database, W1)
    # If the base is less than 1000, then the entire base
    meanS_size = min([1000, database.shape[0]])
    meanS = np.zeros((meanS_size, W1.shape[0]), dtype=np.float64)

    for index in range(meanS.shape[0]):
        meanS[index] = np.sum(np.multiply(database[index], W1), axis=1)
        div = np.divide((meanS[index] - Sh_min), (Sh_max - Sh_min),
                        where=(Sh_max-Sh_min)!=0)
        meanS[index] = np.subtract(div, 0.5, where=div!=0)

    return Sh_max, Sh_min, np.mean(meanS, axis=0)

# Step 6
def formation_matrix_Wn(num, conf_layers, xr_coef=0):
    a = conf_layers[num - 1] + 1
    b = conf_layers[num]
    if (conf_layers.__len__() - 1 - num) != 0:
        b += 1

    W_list = np.empty(a * b, dtype=np.float64)
    if (xr_coef != 0):
        for i in range(W_list.__len__()):
            xr_coef = 1 - 1.95 * np.square(xr_coef)
            W_list[i] = xr_coef
    else:
        W_list[:] = 0.5
    W_list = np.reshape(W_list, (b, a))

    return W_list

def formation_reservoir_coef(conf_layers, LCG):
    """
    Formation of the matrix of weights using the linear congruential generator
        :param conf_layers: Neural network layer configuration
        :param LCG: List to coefficients the linear congruential generator
        :return: Weight matrix W1
    """
    W1 = np.empty((conf_layers[0] + 1, conf_layers[1]))  # (Y+1, P)
    # LCG = ("K", "D", "L", "C")
    zn = LCG[3]
    for j in range(conf_layers[1]): # P
        for i in range(conf_layers[0] + 1): # Y
            # W = (D - K * W) % L;
            zn = int(math.fmod((LCG[1] - LCG[0] * zn), LCG[2]))
            W1[i, j] = zn / LCG[2]
    return W1.T

# Step 8
@njit(cache=True, fastmath=True)
def calculation_neuron_first_layer(norm_data, W1,
                                   Sh_max, Sh_min, Sh_mean):
    """
    Calculation of neurons of the first hidden layer over the entire database
        :param norm_data: Normalized database
        :param W1: Weight matrix W1
        :param Sh_max, Sh_min and Sh_mean: Normalization coefficient
        :return: Matrix Sh containing the values of the neurons of the first hidden layer
    """
    Sh = np.zeros((norm_data.shape[0], Sh_max.shape[0]+1),
                  dtype=np.float64)
    for index in range(norm_data.shape[0]):
        Sh[index] = back_prop_calc_first_layer(norm_data[index], W1,
                                            Sh_max, Sh_min, Sh_mean)
    return Sh

@njit(cache=True, fastmath=True)
def calc_metric(model : np.ndarray, Sout: np.ndarray,
                metric : str) -> np.float64:
    """
    Calculating the entropy value from a metric
        :param model: Reference Vector
        :param Sout: The result of the neural network
        :param metric, 'Acc' - accuracy metric,
                    'R2E' - R2 Efficiency metric,
                    'PE' - Pearson Efficiency metric.
        :return: NNetEn – the entropy value
    """

    if metric == 'Acc':
        if (np.argmax(model) == np.argmax(Sout)):
            return 1
        else:
            return 0
    elif metric == 'R2E':
        R2 = 1 - (np.sum(np.square(model - Sout)) / \
                np.sum(np.square(model-np.mean(model))))
        return R2
    elif metric == 'PE':
        ro = np.sum((model - np.mean(model))*(Sout- np.mean(Sout))) / \
                (np.sqrt(np.sum(np.square(model-np.mean(model))))*
                    np.sqrt(np.sum(np.square(Sout - np.mean(Sout)))))
        return ro


def main():
    pass

if __name__ == '__main__':
    main()
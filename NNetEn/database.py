#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:11:35 2023

@author: Izotov Yuriy
@user: izotov93
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]:%(levelname)s: %(message)s')

import numpy as np
from numba import njit
from datetime import datetime
import os

def read_mnist_images(file_name : str, limit=1):
    """
    Reading their MNIST images
        :param file_name: Image file name mnist. Format idx3-ubyte
        :param limit: Usage fraction of the selected database
        :return: Linear array containing image (0 to 255)
    """
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize
    images = np.fromfile(file_name, dtype='ubyte')
    magicBytes, nImages, width, height = \
        np.frombuffer(images[:nMetaDataBytes].tobytes(), intType)
    images = images[nMetaDataBytes:].astype(
        dtype='uint8').reshape([nImages, width, height])
    # Restriction on reading data
    limit = round(limit * images.shape[0])
    images = images[:limit]

    # Rotating and Mirroring an Image
    images = [np.flipud(np.rot90(i[:], 3, axes=(1, 0)))
                .reshape(width * height) for i in images]

    return np.array(images)

def read_mnist_labels(file_name : str, limit=1):
    """
    Reading their MNIST labels
        :param file_name: Labels file name mnist. Format idx1-ubyte
        :param limit: Usage fraction of the selected database
        :return: Linear array containing labels
    """
    intType = np.dtype('int32').newbyteorder('>')
    labels = np.fromfile(file_name, dtype='ubyte')[2 * intType.itemsize:]
    # Restriction on reading data
    limit = round(limit * labels.shape[0])
    labels = labels[:limit]

    return labels

def T_pattern_load(pattern_file : str):
    """
    Loading a Pattern
        :param pattern_file: Pattern file name mnist. Format .VM
        :return: A linear array containing a pattern of indexes, for converting images to mnist
    """
    logging.debug('Loading pattern file %s', pattern_file)
    try:
        with open(pattern_file) as file:
            sire_ar = int(file.readline())
            data_list = []
            for line in file:
                st_list = line.replace('\n', '').split('\t')
                data_list.append(st_list)
    except:
        logging.info('Error read file %s', pattern_file)
        exit()
    pattern = []
    for el in data_list:
        pattern.append((int(el[1]) - 1) * 28 + int(el[0]))

    return np.array(pattern, dtype='int')

@njit(cache=True, fastmath=True)
def T_pattern_application(data : np.ndarray, pattern : np.ndarray):
    """
    Applying a pattern on one element mnist
        :param data: Array containing image mnist
        :param pattern: Array containing pattern
        :return: Image MNIST converted
    """
    out_data = np.empty_like(data)
    for i in range(data.size):
        out_data[i] = data[pattern[i]-1]
    return out_data

def data_pattern_transform(database, pattern_file :str):
    """
    The function of applying a pattern to the entire image database
        :param database: database containing image mnist
        :param pattern_file: Path to the pattern file
        :return: Prepared database MNIST (0 to 255) + bias
    """
    pattern = T_pattern_load(pattern_file)
    logging.debug('Pattern - %s application', pattern_file)
    database_pattern = np.empty_like(database)
    database_pattern = np.insert(database_pattern, 0, 1, axis=1)
    for index, data in enumerate(database):
        database_pattern[index] = np.insert(
            T_pattern_application(data, pattern), 0, 1)

    return database_pattern

def read_database_covid19(file_name: str, limit=1):
    """
    Covid-19 database reading function
        :param file_name: Path to the SARS-CoV-2-RBV1 file
        :param limit: usage fraction of the selected dataset
        :return: train_data and test_data: Array with data and labels;
                 max_data and min_data: Array of maximum and minimum database values along axis 0.
    """
    try:
        train_data = np.loadtxt(file_name, delimiter='\t')
    except:
        logging.error('File %s database not found', file_name)
        exit()

    # Restriction on reading data
    limit = round(limit * train_data.shape[0])
    train_data = train_data[:limit]

    test_data = train_data[:, -1]
    test_data = np.stack([np.zeros(test_data.__len__()),
                          test_data]).T
    for data in test_data:
        if (data[-1] != 1):
            data[0] = 1

    train_data = np.delete(train_data, -1, 1)
    train_data = np.insert(train_data, 0, 1, axis=1)

    max_data = np.zeros(train_data.shape[1])  # Y
    min_data = np.zeros(train_data.shape[1])  # Y
    min_data[:] = 1000

    for data in train_data:
        max_data = np.maximum(data, max_data)
        min_data = np.minimum(data, min_data)

    return train_data, test_data, max_data, min_data

def read_database_mnist(file_name_image: str,
                        file_name_labels: str,
                        file_pattern : str,
                        limit=1):
    """
    MNIST database reading function
        :param file_name_image: Path to the image mnist
        :param file_name_labels: Path to the label mnist
        :param file_pattern: Path to the pattern
        :param limit: usage fraction of the selected dataset
        :return: train_data, test_data and test_data_array: Array with data and labels;
                max_data and min_data: Array of maximum and minimum database values along axis 0.
    """
    try:
        train_data = read_mnist_images(file_name_image,
                                       limit)
        test_data = read_mnist_labels(file_name_labels,
                                      limit)
    except:
        logging.error('Error read MNIST database')
        exit()
    train_data = data_pattern_transform(train_data, file_pattern)

    test_data_array = np.zeros((test_data.shape[0], 10))
    for index in range(test_data.shape[0]):
        test_data_array[index][test_data[index]] = 1

    max_data = np.zeros(train_data.shape[1])  # Y
    min_data = np.zeros(train_data.shape[1])  # Y
    min_data[:] = 1000
    for data in train_data:
        max_data = np.maximum(data, max_data)
        min_data = np.minimum(data, min_data)

    return train_data, test_data, test_data_array, max_data, min_data

@njit(cache=True, fastmath=True)
def normalization(input, max_data, min_data):
    """
    Single element normalization
        :param input: Database element
        :param max_data: Array of maximum database values
        :param min_data: Array of minimal database values
        :return: Normalized array (0 to 1)
    """
    norm_data = input.copy()
    norm_data[0] = 1
    for i in range(1, input.size):
        if ((max_data[i] - min_data[i]) != 0):
            norm_data[i] = (input[i] - min_data[i]) / \
                        (max_data[i] - min_data[i])
        elif (max_data[i] != 0):
            norm_data[i] = (input[i] - min_data[i]) / max_data[i]
        else:
            norm_data[i] = 0
    return norm_data

# Step 2
def format_norm_database(database, max_data, min_data):
    """
    Database normalization function
        :param database: Array containing database elements
        :param max_data: Array of maximum database values
        :param min_data: Array of minimal database values
        :return: Normalized database (0 to 1)
    """
    norm_database = np.empty_like(database, dtype=np.float64)
    for index, data in enumerate(database):
        norm_database[index] = normalization(np.float64(data),
                                             np.float64(max_data),
                                             np.float64(min_data))
    return norm_database

def main():
    pass

if __name__ == '__main__':
    main()
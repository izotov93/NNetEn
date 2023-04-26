#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:11:35 2023

@author: Izotov Yuriy
@user: izotov93
"""

import logging
# Настройка формата логов программы
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]:%(levelname)s: %(message)s')

import numpy as np
from numba import njit
from datetime import datetime
import os

dir = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(dir, 'Database')

# Reading their MNIST images
def read_mnist_images(file_name : str, limit=1):
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

# Reading MNIST values
def read_mnist_labels(file_name : str, limit=1):
    intType = np.dtype('int32').newbyteorder('>')
    labels = np.fromfile(file_name, dtype='ubyte')[2 * intType.itemsize:]
    # Restriction on reading data
    limit = round(limit * labels.shape[0])
    labels = labels[:limit]

    return labels

# Loading a Pattern
def T_pattern_load(pattern_file : str):
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
    out_data = np.empty_like(data)
    for i in range(data.size):
        out_data[i] = data[pattern[i]-1]
    return out_data

# Применение паттерна навсю базу
def data_pattern_transform(database, pattern_file :str):
    pattern = T_pattern_load(pattern_file)
    logging.debug('Pattern - %s application', pattern_file)
    database_pattern = np.empty_like(database)
    database_pattern = np.insert(database_pattern, 0, 1, axis=1)
    for index, data in enumerate(database):
        database_pattern[index] = np.insert(
            T_pattern_application(data, pattern), 0, 1)

    return database_pattern

# Loading a Time Series
def read_data_xn(path_data_xn):
    logging.debug('Loading file %s', path_data_xn)
    data_list = []
    try:
        with open(path_data_xn) as file:
            for line in file:
                st_list = line.replace(',', '.').split('\t')
                data_list.append(float(st_list[0]))
        data_list = np.array(data_list, dtype='float64')
    except Exception as e:
        logging.info('Error read file %s', path_data_xn)
        exit(0)

    return data_list

# Чтение ковидной базы
def read_database_covid19(file_name: str, limit=1):
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

# Чтение базы MNIST
def read_database_mnist(file_name_image: str,
                        file_name_labels: str,
                        file_pattern : str,
                        limit=1):
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


def write_result_for_txt(NNetEn, time_train, epoch, N, MU,
                         Length_TS, metric):
    result_name = 'log.txt'
    if not os.path.isfile(result_name):
        with open(result_name, mode='a', encoding='utf-8') as file:
            file.write('Timestamp\tNNetEn\tTime\tEpoch\t'
                       'W1 Size\tMU\tLength Time Series\tMetric\n')
    
    with open(result_name, mode='a', encoding='utf-8') as file:
        file.write('[{}]\t{:.4f}\t{:.3f}\t{}\t{}\t{}\t{}\t{}\n'.
                   format(datetime.now(), NNetEn, time_train,
                          epoch, N, MU, Length_TS, metric))

@njit(cache=True, fastmath=True)
# Нормализация одного массива
def normalization(input, max_data, min_data):
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
    norm_database = np.empty_like(database, dtype=np.float64)
    for index, data in enumerate(database):
        norm_database[index] = normalization(np.float64(data),
                                             np.float64(max_data),
                                             np.float64(min_data))
    return norm_database

def load_mnist_database(limit=1):
    # Step 1
    MNIST_TRAIN_IMAGE = os.path.join(DATABASE_PATH,
                                     'train-images.idx3-ubyte')
    MNIST_TRAIN_LABELS = os.path.join(DATABASE_PATH,
                                       'train-labels.idx1-ubyte')
    MNIST_TEST_IMAGE = os.path.join(DATABASE_PATH,
                                     't10k-images.idx3-ubyte')
    MNIST_TEST_LABELS = os.path.join(DATABASE_PATH,
                                      't10k-labels.idx1-ubyte')
    VM_PATTERN_NAME = os.path.join(DATABASE_PATH, 'T-pattern-3.VM')

    # Train database
    dbase, test_db, test_db_array, max_data, min_data = \
        read_database_mnist(MNIST_TRAIN_IMAGE,
                               MNIST_TRAIN_LABELS,
                               VM_PATTERN_NAME, limit)
    # Test database
    dbase_T, test_db_T, test_db_array_T, max_data, min_data = \
        read_database_mnist(MNIST_TEST_IMAGE,
                               MNIST_TEST_LABELS,
                               VM_PATTERN_NAME, limit)
    # Step 2
    logging.debug('Step 2. Formation the normalization database')

    # Train labels
    #FIXME
    norm_data = dbase / 255
    norm_data[:,0] = 1

    #norm_data = format_norm_database(dbase, max_data, min_data)
    # Test labels
    #FIXME
    norm_data_T = dbase_T / 255
    norm_data_T[:,0] = 1
    #norm_data_T = format_norm_database(dbase_T, max_data, min_data)

    return norm_data, test_db_array, norm_data_T, test_db_array_T

def load_covid19_database(limit=1):
    DATABASE_COVID19 = os.path.join(DATABASE_PATH,
                            'Base_0_balance_dot.txt')
    # Step 1
    dbase, test_db, max_data, min_data = \
        read_database_covid19(DATABASE_COVID19, limit)
    # Step 2
    logging.debug('Step 2. Formation the normalization database')
    norm_data = format_norm_database(dbase, max_data, min_data)

    return norm_data, test_db

def main():
    pass

if __name__ == '__main__':
    main()
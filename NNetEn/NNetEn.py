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

from time import time
from .database import read_database_mnist, \
    read_database_covid19, format_norm_database
from .LogNNet import *
from numba.typed import List
import os
from sys import exit

class init_database(object):
    __dir = os.path.dirname(os.path.abspath(__file__))
    __path_database = os.path.join(__dir, 'Database')

    def __init__(self, database='D1', mu=1):
        """
        Class initialization with datasets
            :param database: (default = D1) Select dataset,
                            D1 – MNIST;
                            D2 – SARS-CoV-2-RBV1.
            :param mu: (default = 1) usage fraction of the selected dataset (0.01 .. 1).
            :return: Instance of class init_database
        """

        self.__validation(database=database, mu=mu)

        if database == 'D1':
            self.__read_mnist_database(mu=mu)
        elif database == 'D2':
            self.__read_covid19_database(mu=mu)

    def __validation(self, database : str, mu : float):
        """
        Validation of input data
            :param database: type -> str
            :param mu: type -> float
            :return: If the validation of the type and data was
                not successful, an error message will be displayed
        """
        str_db = {'D1' : 'MNIST',
                  'D2' : 'SARS-CoV-2-RBV1'}.get(database)
        if str_db:
            logging.debug('Loading database {}'
                     ' [Mu - {}]'.format(str_db, mu))
        else:
            logging.error('Invalid parameter - database')
            exit(0)

        if ((isinstance(mu, float) or isinstance(mu, int))
                and (mu >= 0.01 and mu <= 1)):
            self.mu = mu
        else:
            logging.error('Invalid parameter - mu')
            exit(0)

    def __read_mnist_database(self, mu : int):
        """
        Read database mnist and data normalization (0 to 1)
            :param mu: Usage fraction of the selected database (0.01 to 1)
            :return: Arrays of training and test data, as well as their labels.
        """
        # Step 1
        file_pattern = 'T-pattern-3.VM'

        # Train database
        dbase, test_db, self.lab_train, max_data, min_data = \
            read_database_mnist(os.path.join(self.__path_database,
                                         'train-images.idx3-ubyte'),
                                os.path.join(self.__path_database,
                                             'train-labels.idx1-ubyte'),
                                os.path.join(self.__path_database,
                                             file_pattern),
                                limit=mu)
        # Test database
        dbase_T, test_db_T, self.lab_test, max_data, min_data = \
            read_database_mnist(os.path.join(self.__path_database,
                                        't10k-images.idx3-ubyte'),
                                os.path.join(self.__path_database,
                                             't10k-labels.idx1-ubyte'),
                                os.path.join(self.__path_database,
                                             file_pattern),
                                limit=mu)
        # Step 2
        logging.debug('Step 2. Formation the normalization database')
        # Train labels
        # FIXME
        self.norm_db = dbase / 255
        self.norm_db[:, 0] = 1
        # norm_data = format_norm_database(dbase, max_data, min_data)

        # Test labels
        # FIXME
        self.norm_db_T = dbase_T / 255
        self.norm_db_T[:, 0] = 1
        # norm_data_T = format_norm_database(dbase_T, max_data, min_data)

    def __read_covid19_database(self, mu : int):
        """
        Read database SARS-CoV-2-RBV1 and data normalization (0 to 1)
            :param mu: Usage fraction of the selected database (0.01 to 1)
            :return: Arrays of training and test data, as well as their labels.
        """
        # Step 1
        dbase, self.lab_train, max_data, min_data = \
            read_database_covid19(os.path.join(self.__path_database,
                                        'SARS-CoV-2-RBV1.txt'),
                                  limit=mu)
        # Step 2
        logging.debug('Step 2. Formation the normalization database')
        self.norm_db = format_norm_database(dbase, max_data, min_data)

        self.norm_db_T, self.lab_test = \
            self.norm_db, self.lab_train

class NNetEn_entropy(init_database):
    # BASIC PARAMS
    __NORMA = 0.2
    __NEURON_HID_LAYER = 25

    def __init__(self, database='D1', mu=1):
        """
        Initialization of the database by input parameters, formation of the
        basic parameters of the neural network LogNNet
            :param database: (default = D1) Select dataset: D1 – MNIST, D2 – SARS-CoV-2-RBV1.
            :param mu: (default = 1) usage fraction of the selected dataset (0.01 .. 1).
            :return: Instance of class NNetEn_entropy
        """

        super().__init__(database=database, mu=mu)
        self.database = database
        conf_layers = [self.norm_db.shape[1] - 1,
                            self.__NEURON_HID_LAYER,
                            self.lab_test.shape[1]]
        self.conf_layers = List()
        [self.conf_layers.append(x) for x in conf_layers]

        self.W1 = np.zeros([self.conf_layers[1],
                            self.conf_layers[0] + 1], dtype=np.float64)
        self.W2 = np.zeros([self.conf_layers[2],
                            self.conf_layers[1] + 1], dtype=np.float64)

    def __validation_input_params(self, time_series : np.ndarray, method : int,
                                        epoch : int, metric : str, log : bool):
        """
        Validation of input data
            :param time_series: type -> np.ndarray
            :param epoch: type -> int [> 0]
            :param method: type -> int [1 .. 6]
            :param metric: type -> str
            :param log: type -> bool
            :return: If the validation of the type and data was
                not successful, an error message will be displayed
        """

        if not (isinstance(log, bool) and
                isinstance(time_series, np.ndarray)):
            logging.error('Invalid input parameter types')
            exit(0)

        if not (isinstance(method, int) and
            ([x for x in range(1,7)].count(method))):
            logging.error('Invalid parameter - method')
            exit(0)

        if not (isinstance(epoch, int) and epoch > 0):
            logging.error('Invalid parameter - ecpoh')
            exit(0)

        if not (isinstance(metric, str) and
                ['Acc', 'R2E', 'PE'].count(metric)):
            logging.error('Invalid parameter - metric')
            exit(0)

    def __preparation_launch_LogNNet(self, time_series: np.ndarray, method: int):
        """
        Preparatory stage.
        Formation of the weight matrix W1 and W2, calculation of values for normalization
        and calculation of neurons of the 1st layer
            :param time_series: input data with a time series in numpy array format
            :param method: One of 6 methods for forming a reservoir matrix from the time series M1 ... M6
        """

        # Step 3
        logging.debug('Load time series, method - %s', method)
        self.W1 = formation_W1_coef_time_series(time_series,
                                                self.W1, method)

        # Step 4
        logging.debug('Calculating min, max and average coefficients')
        self.Sh_max, self.Sh_min, self.Sh_mean = \
                    calculation_normalization_params(
                        self.norm_db, self.W1)

        # Step 5
        logging.debug('Calculation neurons first layer LogNNet.')
        self.__Sh = calculation_neuron_first_layer(self.norm_db,
                                            self.W1, self.Sh_max,
                                            self.Sh_min, self.Sh_mean)
        self.__Sh = np.nan_to_num(self.__Sh)

        self.W2[:] = 0.5

    def __trainng_LogNNet(self, epoch: int):
        """
        Neural network training stage LogNNet
            :param epoch: The number of training epochs for the LogNNet neural network,
                    with a number greater than 0
            :return: Trained weight matrix W2
        """

        # Step 6
        logging.debug('LogNNet %s network training', self.conf_layers)
        start_all = time()
        for ep in range(epoch):
            start_ep = time()
            for index in range(self.norm_db.shape[0]):
                Sout = back_prop_calc_value(self.__Sh[index], self.W2)
                err2 = err_calc_last_layer(self.lab_train[index], Sout)
                self.W2 = weight_training(self.W2, err2,
                                    self.__Sh[index], self.__NORMA)
            logging.debug('Epoch {}. Time - {:.3f} s.'.
                          format(ep, time() - start_ep))

        logging.debug('Time traning network - {:.3f} s.'.
                      format(time() - start_all))

    def __testing_LogNNet(self, metric: str):
        """
        Neural network testing stage LogNNet and entropy calculation
            :param metric: Options: metric = 'Acc', metric = 'R2E', metric = 'PE'
            :return: NNetEn value
        """

        metric_NNetEn = np.zeros(self.lab_test.shape[0])
        # Step 7
        logging.debug('Testing network')
        for index in range(self.norm_db_T.shape[0]):
            Sh_T = back_prop_calc_first_layer(
                self.norm_db_T[index], self.W1,
                self.Sh_max, self.Sh_min, self.Sh_mean)
            Sh_T = np.nan_to_num(Sh_T)
            self.Sout_T = back_prop_calc_value(Sh_T, self.W2)

            metric_NNetEn[index] = calc_metric(
                                    model=self.lab_test[index],
                                    Sout=self.Sout_T,
                                    metric=metric)

        self.NNetEn = np.mean(metric_NNetEn)

    def calculation(self, time_series, method=3,
                        epoch=20, metric='Acc', log=False):
        """
        Command to calculation a NNetEn parameter
            :param time_series: input data with a time series in numpy array format.
            :param epoch: (default = 20). The number of training epochs for the
                        LogNNet neural network, with a number greater than 0.
            :param method: (default = 3) One of 6 methods for forming a reservoir
                        matrix from the time series M1 ... M6.
            :param metric: (default = 'Acc') 'Acc' - accuracy metric,
                        'R2E' - R2 Efficiency metric,
                        'PE' - Pearson Efficiency metric.
            :param log: (default = False) Parameter for logging the main data used
                        in the calculation. Recording is done in log.txt file.
            :return: NNetEn – the entropy value for the given parameters.
        """

        self.__validation_input_params(time_series=time_series,
                                     method=method, epoch=epoch,
                                     metric=metric, log=log)

        time_work = time()
        self.__preparation_launch_LogNNet(time_series=time_series,
                                            method=method)
        self.__trainng_LogNNet(epoch=epoch)
        self.__testing_LogNNet(metric=metric)

        logging.debug('NNetEn = {}'.format(self.NNetEn))
        logging.debug('Time calculation NNetEn - {:.3f} s.'.
                     format(time() - time_work))

        if (log == True):
            logging.debug('Write results to file - log.txt')
            self.__logging(time() - time_work, epoch, time_series.size, metric)

        return self.NNetEn

    def __logging(self, calc_time, epoch, time_serises_size,
                  metric, log_file='log.txt'):
        """
        Logging of the main parameters used during the operation of the module NNetEn
            :param calc_time: Enropy calculation time NNetEn in seconds
            :param epoch: The number of training epochs of the neural network LogNNet
            :param time_serises_size: Time series array size
            :param metric: Metric for calculating entropy value
            :param log_file: Log file name
            :return: The file is created with the parameters used in the calculations
        """
        dist_log = {
            "NNetEn" : str(self.NNetEn),
            "Calculation Time" : round(calc_time, 4),
            "Epoch" : epoch,
            "Metric" : metric,
            "W1 size" : self.W1.size,
            "Database" : self.database,
            "Coef. mu" : self.mu,
            "Length Time Series" : time_serises_size
        }

        isfile = os.path.isfile(log_file)
        with open(log_file, mode='a', encoding='utf-8') as file:
            if not isfile:
                file.write("\t".join(dist_log.keys()) + "\n")

            file.write("\t".join([str(value) for _, value in
                                  dist_log.items()]) + "\n")

def main():
    pass

if __name__ == '__main__':
    main()
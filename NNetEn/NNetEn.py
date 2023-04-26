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

from time import time
from .database import load_mnist_database, \
    load_covid19_database, write_result_for_txt
from .LogNNet import *
from numba.typed import List

class init_database(object):
    def __init__(self, database='D1', mu=1):
        # Step 1 and Step 2
        # mu - usage fraction of the selected dataset
        if ((isinstance(mu, float) or isinstance(mu, int)) and
                (mu >= 0.01 and mu <= 1)):
            self.mu = mu
        else:
            logging.error('Invalid parameter - mu')
            exit(0)

        str_db = {'D1' : 'MNIST',
                  'D2' : 'COVID-19'}.get(database)
        logging.debug('Loading database {}'
                     ' [Mu - {}]'.format(str_db, mu))

        if database == 'D1':
            self.norm_db, self.lab_train, \
            self.norm_db_T, self.lab_test = \
                load_mnist_database(limit=self.mu)
        elif database == 'D2':
            self.norm_db, self.lab_train = \
                load_covid19_database(limit=self.mu)
            self.norm_db_T, self.lab_test = \
                self.norm_db, self.lab_train
        else:
            logging.error('Invalid parameter - database')
            exit(0)

class NNetEn_entropy(init_database):
    # BASIC PARAMS
    __NORMA = 0.2
    __NEURON_HID_LAYER = 25

    def __init__(self, database='D1', mu=1):
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

    def __validation_input_params(self, time_series, method,
                                        epoch, metric, log):
        # Validation of input data
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

    def __preparation_launch_LogNNet(self, time_series, method):
        # Step 3
        logging.debug('Load time series, method - %s', method)
        self.W1 = formation_W1_coef_time_series(
                time_series, self.conf_layers, self.W1, method)

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

    def __trainng_LogNNet(self, epoch):
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

    def __calc_metric(self, model, Sout, metric):
        if metric == 'Acc':
            if (np.argmax(model) == np.argmax(Sout)):
                return 1
            else:
                return 0
        elif metric == 'R2E':
            R2 = 1 - (np.sum(np.square(model - Sout)) / \
                      np.sum(np.square(model - np.mean(model))))
            return R2
        elif metric == 'PE':
            ro = np.sum((model - np.mean(model)) * (Sout - np.mean(Sout))) / \
                 (np.sqrt(np.sum(np.square(model - np.mean(model)))) *
                  np.sqrt(np.sum(np.square(Sout - np.mean(Sout)))))
            return ro

    def __testing_LogNNet(self, metric):
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

    def calculation(self, time_series, method=5,
                        epoch=20, metric='Acc', log=False):

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
            write_result_for_txt(self.NNetEn, time() - time_work,
                                    epoch, self.W1.size, self.mu,
                                    time_series.size, metric)

        return self.NNetEn

def main():
    pass

if __name__ == '__main__':
    main()
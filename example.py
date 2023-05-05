from NNetEn import NNetEn_entropy
import numpy as np

file_example = 'Time_series_example.txt'

def main():
    # Reading the file containing time series
    time_seties = np.loadtxt(file_example, delimiter=' ')
    print("Read {} time series".format(time_seties.shape[0]))

    # Creating a class with the selected database
    #
    # D1 - MNIST-10, D2 - SARS-CoV-2-RBV1
    # mu - usage fraction of the selected database
    NNetEn = NNetEn_entropy(database='D1', mu=0.1)

    #  Calculation the NNetEn value
    #
    # time_series: input data with a time series in numpy array format.
    # epoch: The number of training epochs for the LogNNet neural network,
    #           with a number greater than 0.
    # method: One of 6 methods for forming a reservoir matrix
    #           from the time series M1 ... M6.
    # metric: 'Acc' - accuracy metric,
    #         'R2E' - R2 Efficiency metric,
    #          'PE' - Pearson Efficiency metric.
    #  log: Parameter for logging the main data used in the calculation.
    #       Recording is done in log.txt file.

    result = np.empty(time_seties.shape[0])
    for num, series in enumerate(time_seties):
        value = NNetEn.calculation(series, epoch=5, method=3,
                                    metric='Acc', log=False)
        print("Time series # {} NNetEn = {}".format(num, value))
        result[num] = value

    file_result = file_example.split('.')[0] + '_result.txt'
    print('Write results to file {}'.format(file_result))
    np.savetxt(file_result, result, fmt='%f')

if __name__ == '__main__':
    main()
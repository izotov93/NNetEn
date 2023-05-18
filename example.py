from NNetEn import NNetEn_entropy
import numpy as np
import re

# Default Configuration
config = {
    'file': 'Time_series_example.txt',
    'ds': 'D1',
    'mu': 0.1,
    'method': 3,
    'metric': 'Acc',
    'epoch':  5
}

def read_file_time_series(input_file : str):
    result = []
    with open(input_file) as file:
        for lines in file.readlines():
            lines = re.sub(r'[ \t]+', " ", lines.strip(), 0)
            result.append(np.array(lines.split(' ')).astype(float))
    return result

def main():
    # Print Configuration
    print('Running calculations with configuration:')
    print('Dataset - {}, mu - {}'.format(
        config['ds'], str(config['mu'])))
    print('Method - {}, Metric - {}, Epoch - {}'.format
          (str(config['method']), config['metric'], str(config['epoch'])))

    # Reading the file containing time series
    time_seties = read_file_time_series(config['file'])
    print("Read {} time series from {}".format(len(time_seties),
                                               config['file']))

    # Creating a class with the selected database
    # D1 - MNIST-10, D2 - SARS-CoV-2-RBV1
    # mu - usage fraction of the selected database
    NNetEn = NNetEn_entropy(database=config['ds'], mu=config['mu'])

    #  Calculation the NNetEn value
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

    result = []
    for series in time_seties:
        value = NNetEn.calculation(series, epoch=config['epoch'],
                                   method=config['method'],
                                   metric=config['metric'],
                                   log=False)
        print("NNetEn = {}".format(value))
        result.append(value)

    # Formation of the result file name
    file_result = re.sub(r'^file', '', '_'.join([str(key) + str(value)
                         for key, value in config.items()])) + '.txt'

    print('Write results to file {}'.format(file_result))
    with open(file_result, 'w') as file:
        file.write("\n".join(str(item) for item in result))

if __name__ == '__main__':
    main()
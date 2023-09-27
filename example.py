from NNetEn import NNetEn_entropy
import numpy as np
import re

'''
    Default Configuration:
        file : Path to the time series file    
        ds : Dataset ('D1' - MNIST-10, 'D2' - SARS-CoV-2-RBV1)
        mu : Usage fraction of the selected database
        method: One of 6 methods for forming a reservoir matrix
                   from the time series M1 ... M6.
        metric: 'Acc' - accuracy metric,
                'R2E' - R2 Efficiency metric,
                'PE' - Pearson Efficiency metric.
        epoch: The number of training epochs for the LogNNet neural network.
        read_mode: Method to read file 
                    ('Rows' reading by rows or 'Columns' reading by columns)
'''

config = {
    'file': 'Time_series_example.txt',
    'ds': 'D1',
    'mu': 1,
    'method': 3,
    'metric': 'Acc',
    'epoch': 5,
    'read_mode': 'Rows'
}


def reading_by_columns(mat: list) -> list:
    matrix = []
    for i in range(len(mat[0])):
        matrix.append(list())
        for j in range(len(mat)):
            try:
                matrix[i].append(mat[j][i])
            except Exception as e:
                print(e)
    return matrix


def read_file_time_series(input_file: str, mode='Rows') -> list:
    result = []
    with open(input_file) as file:
        line_list = file.readlines()
        for lines in line_list:
            lines = re.sub(r'[^(0-9.,\-eE \t)]', '', lines.strip())
            lines = lines.replace(',', '.')
            lines = re.sub(r'[ \t]+', ' ', lines, 0)
            if lines != '':
                result.append(lines.split(' '))

    if mode == 'Columns':
        result = reading_by_columns(result)

    return [np.array(data).astype(float) for data in result]


def main():
    # Print Configuration
    print('Running calculations with configuration:')
    print('Dataset - {}, mu - {}'.format(config['ds'], str(config['mu'])))
    print('Method - {}, Metric - {}, Epoch - {}'.format
          (str(config['method']), config['metric'], str(config['epoch'])))
    print("Reading by {} from {}".format(config['read_mode'], config['file']))

    # Reading the file containing time series
    time_series = read_file_time_series(config['file'], config['read_mode'])
    print("Read {} time series".format(len(time_series)))

    # Creating a class with the selected database
    NNetEn = NNetEn_entropy(database=config['ds'], mu=config['mu'])

    #  Calculation the NNetEn value
    result = []
    for series in time_series:
        value = NNetEn.calculation(series, epoch=config['epoch'], method=config['method'],
                                   metric=config['metric'], log=False)
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

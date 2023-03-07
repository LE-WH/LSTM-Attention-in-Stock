import json
import numpy as np
import time
import os
import pickle

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


from sklearn.preprocessing import MinMaxScaler 

from tqdm import tqdm
from datetime import date, datetime, timedelta

import model


DATA_DIM = 7


def load_config():
    f = open('config.json')
    configs = json.load(f)

    data_path = configs['data_path']
    data_split = configs['data_split']
    model_config = configs['model_config']

    with open(data_path + 'code_list.txt', 'r', encoding='utf-8') as codes:
        codes = codes.readlines()
        code_list = [line[0:6] for line in codes]

    f.close()
    return code_list, data_path, data_split, model_config


class trading_day():
    day_list = []
    day2ind = {}

    @classmethod
    def get_day_list(cls, data_path):
        g = os.walk(data_path + 'timely_data')
        for _, _, file_list in g:
            cls.day_list = [line[:10] for line in file_list]
            cls.day_list.sort()
        for i, day in enumerate(cls.day_list):
            cls.day2ind[day] = i

    def __init__(self, d):
        # d: Year-month-day
        self.day_index = self.day_list.index(d)

    def __add__(self, x):
        self.day_index = self.day_index + x
        if self.day_index >= len(self.day_list):
            raise ValueError('Out of trading day range! Before: {}, Adding: {}'.format(
                self.day_list[self.day_index-x], x))

    def __sub__(self, x):
        self.day_index = self.day_index - x
        if self.day_index < 0:
            raise ValueError('Out of trading day range! Before: {}, Subtracting: {}'.format(
                self.day_list[self.day_index+x], x))


class Dataset():

    def __init__(self, code_list, data_path, data_split, model_config):
        self.code_list = code_list
        self.data_path = data_path

        self.training_start = data_split['training_start']
        self.training_end = data_split['training_end']
        self.validation_start = data_split['validation_start']
        self.validation_end = data_split['validation_end']
        self.test_start = data_split['test_start']
        self.test_end = data_split['test_end']

        self.local_length = model_config['local_length']
        self.global_length = model_config['global_length']
        self.global_width = model_config['global_width']

        # Current_time is the day to be predicted
        self.current_time = trading_day.day2ind[
            self.training_start] + self.local_length

        self.mission = 'training'

        self.stock_data = {}  # a big dict, using codes as keys, np.array as values
        # self.stock_data = np.zeros((self.global_width, self.data_end_ind - self.data_start_ind + 1, DATA_DIM))

        self.time2stocks = {}  # time index: list of stock indexes

        self.code2ind = {}
        self.ind2code = {}

        self.time2targets = {}  # time index: list of targets
        # time index: list of secondary stocks that we'll need to predict the targets.
        self.time2secondary_stocks = {}

    def start(self):
        '''
        set the current_time 
        '''
        if self.mission == 'training':
            self.current_time = trading_day.day2ind[
                self.training_start] + self.local_length
        elif self.mission == 'validation':
            self.current_time = trading_day.day2ind[self.validation_start]
        elif self.mission == 'test':
            self.current_time = trading_day.day2ind[self.test_start]
        else:
            raise ValueError('No such mission as "{}"'.format(self.mission))

    def load(self):
        '''
        load data from csv
        '''
        # load code2ind, ind2code
        for i, code in enumerate(self.code_list):
            self.code2ind[code] = i
            self.ind2code[i] = code

        # load self.stock_data
        print('-'*6, 'Loading stock_data', '-'*6)
        if os.path.exists('temp/stock_data.pkl'):
            print('-'*6, 'Start loading data from temp file', '-'*6)
            st = time.time()
            with open('temp/stock_data.pkl', 'rb') as f:
                self.stock_data = pickle.load(f)
            print('-'*6, 'Loading Done! Using {} seconds'.format(time.time() - st), '-'*6)
        else:
            print('-'*6, 'Start loading data from csv file', '-'*6)
            for i in tqdm(range(self.global_width)):
                code = self.code_list[i]
                with open(self.data_path + 'stock_data/' + code + '.csv', 'r', encoding='utf-8') as f:
                    self.stock_data[code] = np.array([[trading_day.day2ind[line.split(
                        ',')[3]]] + [float(item) for item in line.split(',')[4:]] for line in f.read().splitlines()[1:]])
            with open('temp/stock_data.pkl', 'wb') as f:
                pickle.dump(self.stock_data, f)
            print('-'*6, 'Loading Done!', '-'*6)

        # load self.time2stocks
        print('-'*6, 'Loading time2stocks', '-'*6)
        if os.path.exists('temp/time2stocks.pkl'):
            print('-'*6, 'Start loading data from temp file', '-'*6)
            st = time.time()
            with open('temp/time2stocks.pkl', 'rb') as f:
                self.time2stocks = pickle.load(f)
            print('-'*6, 'Loading Done! Using {} seconds'.format(time.time() - st), '-'*6)
        else:
            print('-'*6, 'Start loading data from csv file', '-'*6)

            g = os.walk(self.data_path + 'timely_data')
            for _, _, file_list in g:
                file_list = [line[:10] for line in file_list]

            file_list.sort()

            for i in tqdm(range(len(file_list))):
                day = file_list[i]
                with open(self.data_path + 'timely_data/' + day + '.csv', 'r', encoding='utf-8') as f:
                    self.time2stocks[trading_day.day2ind[day]] = [
                        self.code2ind[line.split(',')[0]] for line in f.read().splitlines()[1:]]
            with open('temp/time2stocks.pkl', 'wb') as f:
                pickle.dump(self.time2stocks, f)
            print('-'*6, 'Loading Done!', '-'*6)

    def check_targets(self):
        '''
        We'll only try to predict stocks with continued 250 days (local_length) of history
        And will only use stocks with continued 50 days (global_length) of history to help predicting
        '''

        print('-'*6, 'Checking targets', '-'*6)

        if os.path.exists('temp/time2targets.pkl') and os.path.exists('temp/time2secondary_stocks.pkl'):
            print('-'*6, 'Start loading data from temp file', '-'*6)
            st = time.time()
            with open('temp/time2targets.pkl', 'rb') as f:
                self.time2targets = pickle.load(f)
            with open('temp/time2secondary_stocks.pkl', 'rb') as f:
                self.time2secondary_stocks = pickle.load(f)
            print('-'*6, 'Loading Done! Using {} seconds'.format(time.time() - st), '-'*6)
        else:
            for day_ind in tqdm(range(self.local_length, len(trading_day.day_list) - 1)):
                candidates = set(self.time2stocks[day_ind])

                for i in range(self.global_length):
                    candidates = candidates & set(
                        self.time2stocks[day_ind - i - 1])

                
                self.time2secondary_stocks[day_ind] = list(candidates)
                self.time2secondary_stocks[day_ind].sort()

                for i in range(self.global_length, self.local_length):
                    candidates = candidates & set(
                        self.time2stocks[day_ind - i - 1])
                candidates = candidates & set(self.time2stocks[day_ind + 1])
                candidates = list(candidates)
                candidates.sort()
                self.time2targets[day_ind] = candidates
            with open('temp/time2targets.pkl', 'wb') as f:
                pickle.dump(self.time2targets, f)
            with open('temp/time2secondary_stocks.pkl', 'wb') as f:
                pickle.dump(self.time2secondary_stocks, f)
            print('-'*6, 'Checking targets done', '-'*6)

    def conv(self):
        '''
        After setting the mission, repeatedly yield the following triples:

        A: ndarray. Shape = self.global_width, self.local_length, DATA_DIM. The data to feed the model
        targets: ndarray. Shape = self.global_width. The percentage we want to predict
        indexes: dict. {'targets': a list of stock indexes that are the targets.
                       'secondary': a list of stock indexes that we need to help the predicting.}

        '''
        self.start()

        if self.mission == 'training':
            terminate_time = trading_day.day2ind[self.training_end]
        elif self.mission == 'validation':
            terminate_time = trading_day.day2ind[self.validation_end]
        else:  # self.mission == 'test'
            terminate_time = trading_day.day2ind[self.test_end]

        cache_ind1 = {code:0 for code in self.code_list}
        cache_ind2 = {code:0 for code in self.code_list}

        span = terminate_time - self.current_time + 1 

        # while self.current_time <= terminate_time:
        for _ in tqdm(range(span)):

            A = np.zeros((self.global_width, self.local_length, DATA_DIM))
            targets = np.zeros(self.global_width)

            # start_time = self.current_time - self.local_length
            # end_time = self.current_time - 1

            for i, code in enumerate(self.code_list):
                if i in self.time2targets[self.current_time]:
                    if cache_ind1[code] + 1 == self.current_time - self.local_length:
                        cache_ind1[code] = cache_ind1[code] + 1
                        start_ind = cache_ind1[code]
                    else:
                        start_ind = np.where(self.stock_data[code][:, 0] == self.current_time - self.local_length)[0][0]
                        cache_ind1[code] = start_ind

                    tmpArray = self.stock_data[code][start_ind:start_ind + self.local_length, [1, 2, 3, 4, 7, 8, 10]]
                    # normalize it

                    tmpArray[:,0:4] = (tmpArray[:,0:4] - np.min(tmpArray[:,0:4])) / (np.max(tmpArray[:,0:4]) - np.min(tmpArray[:,0:4]))
                    tmpArray[:,4] = (tmpArray[:,4] - np.min(tmpArray[:,4])) / (np.max(tmpArray[:,4]) - np.min(tmpArray[:,4]))
                    tmpArray[:,5] = (tmpArray[:,5] - np.min(tmpArray[:,5])) / (np.max(tmpArray[:,5]) - np.min(tmpArray[:,5]))
                    tmpArray[:,6] = (tmpArray[:,6] - np.min(tmpArray[:,6])) / (np.max(tmpArray[:,6]) - np.min(tmpArray[:,6]))

                    A[i, :, :] = tmpArray

                    opening1 = self.stock_data[code][start_ind + self.local_length, 1]
                    opening2 = self.stock_data[code][start_ind + self.local_length + 1, 1]

                    # percentage = (opening2 - opening1) / opening1 * 100
                    percentage = opening2 - opening1
                    if percentage < 0:
                        targets[i] = 0
                    else:
                        targets[i] = 1

                    # print(code, opening2, opening1, targets[i])


                elif i in self.time2secondary_stocks[self.current_time]:

                    if cache_ind2[code] + 1 == self.current_time - self.global_length:
                        cache_ind2[code] = cache_ind2[code] + 1
                        start_ind = cache_ind2[code]
                    else:
                        start_ind = np.where(self.stock_data[code][:, 0] == self.current_time - self.global_length)[0][0]
                        cache_ind2[code] = start_ind

                    tmpArray = self.stock_data[code][start_ind:start_ind + self.global_length, [1, 2, 3, 4, 7, 8, 10]]

                    tmpArray[:,0:4] = (tmpArray[:,0:4] - np.min(tmpArray[:,0:4])) / (np.max(tmpArray[:,0:4]) - np.min(tmpArray[:,0:4]))
                    tmpArray[:,4] = (tmpArray[:,4] - np.min(tmpArray[:,4])) / (np.max(tmpArray[:,4]) - np.min(tmpArray[:,4]))
                    tmpArray[:,5] = (tmpArray[:,5] - np.min(tmpArray[:,5])) / (np.max(tmpArray[:,5]) - np.min(tmpArray[:,5]))
                    tmpArray[:,6] = (tmpArray[:,6] - np.min(tmpArray[:,6])) / (np.max(tmpArray[:,6]) - np.min(tmpArray[:,6]))

                    A[i, -self.global_length:, :] = tmpArray


            indexes = {'targets': self.time2targets[self.current_time],
                       'secondary': self.time2secondary_stocks[self.current_time]}



            A = torch.tensor(A, requires_grad=False).to(torch.float32)
            # targets = F.one_hot(torch.tensor(targets[self.time2targets[self.current_time]], dtype=int), 4)
            targets = torch.tensor(targets[self.time2targets[self.current_time]], dtype=int)

            # print(A)
            # print(targets)
            # print(indexes)
            # time.sleep(100)


            yield  A, targets, indexes

            self.current_time = self.current_time + 1

def Initialize():
    code_list, data_path, data_split, model_config = load_config()
    trading_day.get_day_list(data_path)

    D = Dataset(code_list, data_path, data_split, model_config)
    D.load()
    D.check_targets()

    P = model.Predictor(model_config)
    return D, P

    # def load(self):
    #     # load all data into cache
    #     print('-'*6, 'Start loading data', '-'*6)
    #     for i in tqdm(range(self.global_width)):
    #         code = self.code_list[i]
    #         with open(self.data_path + code + '.csv', 'r', encoding='utf-8') as f:
    #             self.stock_data[code] = [line.split(',')[3:] for line in f.read().splitlines()[1:]]
    #             for line in self.stock_data[code]:
    #                 line[0] = trading_day.day_list.index(line[0])
    #     print('-'*6, 'Loading Done!', '-'*6)

    # def load(self):
    #     # load all data into cache
    #     print('-'*6, 'Start loading data', '-'*6)
    #     for i in tqdm(range(self.global_width)):
    #         code = self.code_list[i]
    #         with open(self.data_path + code + '.csv', 'r', encoding='utf-8') as f:
    #             tmp_list = [line.split(',')[3:] for line in f.read().splitlines()[1:]]
    #             for line in tmp_list:
    #                 line[0] = trading_day.day_list.index(line[0])
    #                 line[1:] = [float(a) for a in line[1:]]
    #             p = 0
    #             for j in range(self.data_end_ind - self.data_start_ind + 1):
    #                 #在tmp_list: ind, 开盘价,收盘价,最高价,最低价,涨跌幅,涨跌额,成交量,成交额,振幅,换手率
    #                 #在stock_data: 开盘价,收盘价,最高价,最低价,成交量,成交额,换手率
    #                 if tmp_list[p][0] == j :
    #                     self.stock_data[i,j,0:4] = np.array(tmp_list[p][1:5])
    #                     self.stock_data[i,j,4:6] = np.array(tmp_list[p][7:9])
    #                     self.stock_data[i,j,6] = tmp_list[p][10]
    #                     p = p + 1
    #                 # else we'll use a linear function to estimate
    #                 elif j == 0:
    #                     self.stock_data[i,j,0:4] = (np.array(tmp_list[p][1:5]) * tmp_list[p+1][0] - tmp_list[p][0] * np.array(tmp_list[p+1][1:5])) / (tmp_list[p+1][0] - tmp_list[p][0])
    #                     self.stock_data[i,j,4:6] = (np.array(tmp_list[p][7:9]) * tmp_list[p+1][0] - tmp_list[p][0] * np.array(tmp_list[p+1][7:9])) / (tmp_list[p+1][0] - tmp_list[p][0])
    #                     self.stock_data[i,j,6] = (tmp_list[p][10] * tmp_list[p+1][0] - tmp_list[p][0] * tmp_list[p+1][10]) / (tmp_list[p+1][0] - tmp_list[p][0])
    #                     if tmp_list[p+1][0] > 10:
    #                         print("i: ", i)
    #                         print("j: ", j)
    #                         print("code: ", code)
    #                         print(self.stock_data[i,j,:])
    #                         time.sleep(10)
    #                 elif j == self.data_end_ind - self.data_start_ind:
    #                     self.stock_data[i,j,0:4] = (np.array(tmp_list[p][1:5]) * (j - tmp_list[p-1][0]) - (j - tmp_list[p][0]) * np.array(tmp_list[p-1][1:5])) / (tmp_list[p][0] - tmp_list[p-1][0])
    #                     self.stock_data[i,j,4:6] = (np.array(tmp_list[p][7:9]) * (j - tmp_list[p-1][0]) - (j - tmp_list[p][0]) * np.array(tmp_list[p-1][7:9])) / (tmp_list[p][0] - tmp_list[p-1][0])
    #                     self.stock_data[i,j,6] = (tmp_list[p][10] * (j - tmp_list[p-1][0]) - (j - tmp_list[p][0]) * tmp_list[p-1][10]) / (tmp_list[p][0] - tmp_list[p-1][0])
    #                 else:
    #                     self.stock_data[i,j,0:4] = (np.array(tmp_list[p][1:5]) + (tmp_list[p][0] - j) * self.stock_data[i, j-1 ,0:4]) / (1 + tmp_list[p][0] - j)
    #                     self.stock_data[i,j,4:6] = (np.array(tmp_list[p][7:9]) + (tmp_list[p][0] - j) * self.stock_data[i, j-1 ,4:6]) / (1 + tmp_list[p][0] - j)
    #                     self.stock_data[i,j,6] = (tmp_list[p][10] + (tmp_list[p][0] - j) * self.stock_data[i, j-1 ,6]) / (1 + tmp_list[p][0] - j)
    #     print('-'*6, 'Loading Done!', '-'*6)


# code_list, data_path, data_split, model_config = load_config()
# trading_day.get_day_list(data_path)

# dataset = Dataset(code_list, data_path, data_split, model_config)
# dataset.load()
# dataset.check_targets()
# print(Dataset.stock_data['900957'])

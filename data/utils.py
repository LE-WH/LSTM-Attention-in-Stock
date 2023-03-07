import os
import time

def get_all_dates():
    # because some stocks have days off, we need to get a list of trading days
    g = os.walk('sieved_data')

    # Firstly output the list of sieved stocks for later use
    for _, _, file_list in g:  
        file_list = [line[:6] for line in file_list]
        with open('code_list_sieved.txt', 'w') as f:
            for code in file_list:
                f.write(code + '\n')
    
        # Then find all trading days

        all_days = set()

        for code in file_list:
            with open('sieved_data\{}.csv'.format(code), encoding='utf-8') as f:
                f = f.read().splitlines()
                a = set([line.split(',')[3] for line in f[1:]])
                all_days = all_days.union(a)

        all_days = list(all_days)
        all_days.sort()

        with open('all_trading_days.txt', 'w') as f:
            for day in all_days:
                f.write(day + '\n')

            
def rearrange():
    # rearrange the data according to time order
    g = os.walk('stock_data')

    for _, _, file_list in g:  
        file_list = [line[:6] for line in file_list]
        min_list = []
        code_ind = {}
        code_mode = {}
        mode = 1
        for code in file_list:
            with open('stock_data\{}.csv'.format(code),'r' , encoding='utf-8') as f:
                f = f.read().splitlines()[1:]
                min_list.append(f[0].split(',')[3])
            code_ind[code] = 0
            code_mode[code] = 1
        cur_date = min(min_list)

        while(mode):
            tmp_list = []
            min_set = set()
            for code in file_list:
                with open('stock_data\{}.csv'.format(code),'r' , encoding='utf-8') as f:
                    if not code_mode[code]:
                        continue
                    f = f.read().splitlines()[1:]

                    if f[code_ind[code]].split(',')[3] == cur_date:
                        tmp_list.append(','.join(f[code_ind[code]].split(',')[1:]))
                        code_ind[code] = code_ind[code] + 1

                    if code_ind[code] >= len(f):
                        code_mode[code] = 0
                    else:
                        min_set = min_set.union({f[code_ind[code]].split(',')[3]})
            mode = 0
            for code in file_list:
                if code_mode[code] == 1:
                    mode = 1
                    break
            with open('timely_data\{}.csv'.format(cur_date), 'w', encoding='utf-8') as f:
                f.write('股票代码,股票名称,时间,开盘价,收盘价,最高价,最低价,涨跌幅,涨跌额,成交量,成交额,振幅,换手率\n')
                for line in tmp_list:
                    f.write(line + '\n')
            cur_date = min(min_set)


# def fix1():
#     g = os.walk('timely_data')

#     for _, _, file_list in g:  
#         file_list = [line for line in file_list]
#         for code in file_list:
#             with open('timely_data\{}'.format(code),'r' , encoding='utf-8') as f:
#                 f = f.read().splitlines()[1:]
#                 t = f[0].split(',')[2]
#             os.rename('timely_data\{}'.format(code), 'timely_data\{}.csv'.format(t))

# def fix2():
#     g = os.walk('stock_data')

#     for _, _, file_list in g:  
#         file_list = [line[:6] for line in file_list]

#         tmp_list = []
#         for code in file_list:
#             with open('stock_data\{}.csv'.format(code),'r' , encoding='utf-8') as f:
#                 f = f.read().splitlines()[1:]
#                 if f[-1].split(',')[3] == '2023-01-20':
#                     tmp_list.append(f[-1])

                

#         with open('timely_data\\2023-01-20.csv', 'w', encoding='utf-8') as f:
#             f.write('股票代码,股票名称,时间,开盘价,收盘价,最高价,最低价,涨跌幅,涨跌额,成交量,成交额,振幅,换手率\n')
#             for line in tmp_list:
#                 f.write(line + '\n')

def fix3():
    tmp_list = []
    with open('data\\timely_data\\2023-01-20.csv', 'r', encoding='utf-8') as f:
        tmp_list = [','.join(line.split(',')[1:]) for line in f.read().splitlines()[1:]]

    with open('data\\timely_data\\2023-01-20.csv', 'w', encoding='utf-8') as f:
        f.write('股票代码,股票名称,时间,开盘价,收盘价,最高价,最低价,涨跌幅,涨跌额,成交量,成交额,振幅,换手率\n')
        for line in tmp_list:
            f.write(line + '\n')

fix3()















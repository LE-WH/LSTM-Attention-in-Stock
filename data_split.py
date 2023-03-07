import os
import datetime
import pandas as pd


def data_funnel(code_list, begin, end, black_list):
    # find stocks that have the full data from the beginning to the end.
    # begin: year-month-day
    # end: year-month-day

    begin_time = datetime.datetime.strptime(begin,'%Y-%m-%d').date()
    end_time = datetime.datetime.strptime(end,'%Y-%m-%d').date()

    if os.path.isdir('sieved_data'):
        raise RuntimeError('Have already sieved the data')
    os.makedirs('sieved_data')

    empty_set = []
    for code in code_list:
        if code in black_list:
            continue
        table = pd.read_csv('stock_data/{}.csv'.format(code), index_col=0)
        try:
            table_start = datetime.datetime.strptime(table.loc[0]['时间'],'%Y-%m-%d').date()
        except:
            # raise RuntimeError('Empty data with code {}'.format(code))
            empty_set.append(code)
            continue
        table_end = datetime.datetime.strptime(table.loc[len(table)-1]['时间'],'%Y-%m-%d').date()

        start_ind = None
        end_ind = None

        if table_start > begin_time or table_end < end_time:
            continue

        for i in range(len(table)):
            line_time = datetime.datetime.strptime(table.loc[i]['时间'],'%Y-%m-%d').date()
            if line_time >= begin_time and not start_ind:
                start_ind = i
                if code == '000005':
                    print(start_ind)
            if line_time >= end_time and not end_ind:
                end_ind = i
        new_table = table.loc[start_ind:end_ind]
        new_table.reset_index(drop=True, inplace=True)
        new_table.to_csv('sieved_data/{}.csv'.format(code))
    print('these are empty csv:', empty_set)
            



if __name__ == "__main__":
    with open('code_list.txt', 'r', encoding='utf-8') as codes:
        codes = codes.readlines()
        code_list = [line[0:6] for line in codes]
        black_list = ['000005']
        begin = '2015-01-01'
        end = '2023-01-20'
        data_funnel(code_list, begin, end, black_list)
import time
import math
import random
import requests
import json
import os
import pandas as pd
import tqdm

def crawl_stock(code_list, begin='0', end='20500101'):
    # 使用之前可以先用浏览器打开以下网址，（随便以一只股票为例），更新一下Cookie
    # https://quote.eastmoney.com/concept/SZ300258.html?from=data#fschart-r
    # 最好验证一下ut是否是对的，因为我还不清楚它是什么参数

    url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?'
    header = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        'Cookie': 'qgqp_b_id=4c85db87fc3bb7b8b2cf2492d7e13838; st_si=31979787216229; st_asi=delete; em-quote-version=topspeed; HAList=a-sz-300258-%u7CBE%u953B%u79D1%u6280%2Ca-sz-001331-%u80DC%u901A%u80FD%u6E90%2Ca-sz-300986-%u5FD7%u7279%u65B0%u6750%2Ca-sh-900943-%u5F00%u5F00%uFF22%u80A1%2Ca-sh-900945-%u6D77%u63A7B%u80A1%2Ca-sh-688800-%u745E%u53EF%u8FBE%2Ca-sz-000933-%u795E%u706B%u80A1%u4EFD%2Cty-1-600395-%u76D8%u6C5F%u80A1%u4EFD%2Cty-0-002403-%u7231%u4ED5%u8FBE%2Cty-0-301112-%u4FE1%u90A6%u667A%u80FD%2Cty-1-603636-%u5357%u5A01%u8F6F%u4EF6%2Cty-1-601728-%u4E2D%u56FD%u7535%u4FE1%2Cty-1-600908-%u65E0%u9521%u94F6%u884C; st_pvi=22786667246730; st_sp=2023-01-21%2014%3A48%3A54; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=139; st_psi=20230123223032378-113200301202-4422128636',
        'Referer': 'http://quote.eastmoney.com/',
        'Host': 'push2his.eastmoney.com'
    }
    code_list_len = len(code_list)
    for i in tqdm.tqdm(range(code_list_len)):
        code = code_list[i]
        if os.path.isfile('stock_data/{}.csv'.format(code)):
            raise ValueError(
                'We\'ve already crawled the data for {}.'.format(code))

        c = 1 if code[0] == '6' or code[0] == '9' else 0
        params = {'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                  'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                  'beg': begin,
                  'end': end,
                  'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                  'rtntype': '6',
                  'secid': '{}.{}'.format(c, code),
                  'klt': '101',
                  'fqt': '2',  # 1: backward adjusted; 2: forward adjusted
                  'cb': 'jsonp' + str(round(time.time() + 1000001 * random.random()))
                  }

        stock_df = pd.DataFrame(columns=['股票代码', '股票名称', "时间", '开盘价', '收盘价', '最高价', '最低价', "涨跌幅", '涨跌额',
                                         "成交量", "成交额", "振幅", "换手率"])
        res = requests.get(url, headers=header, params=params)
        res.encoding = "utf-8"

        html = res.text.lstrip(params['cb']+'(')
        html = html.rstrip(');')
        js_html = json.loads(html)
        js_data = js_html['data']
        js_klines = js_data['klines']

        day_num = len(js_klines)
        for num in range(day_num):
            stock_df.loc[num] = [str(js_data['code']),
                                 js_data['name'],
                                 js_klines[num].split(",")[0],
                                 js_klines[num].split(",")[1],
                                 js_klines[num].split(",")[2], js_klines[num].split(",")[
                3], js_klines[num].split(",")[4],
                js_klines[num].split(",")[8], js_klines[num].split(",")[
                9], js_klines[num].split(",")[5],
                js_klines[num].split(",")[6], js_klines[num].split(",")[
                7], js_klines[num].split(",")[10]
            ]
        stock_df.to_csv('stock_data/{}.csv'.format(code))

        time.sleep(random.random()*5)


def get_list():
    # 爬取所有股票的代码及名称
    # 写入同目录下的code_list.txt

    output = open('code_list.txt', 'a', encoding='utf-8')

    url = 'https://push2.eastmoney.com/api/qt/clist/get?'
    header = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        'Cookie': 'qgqp_b_id=4c85db87fc3bb7b8b2cf2492d7e13838; st_si=31979787216229; st_asi=delete; em-quote-version=topspeed; HAList=a-sh-900943-%u5F00%u5F00%uFF22%u80A1%2Ca-sh-900945-%u6D77%u63A7B%u80A1%2Ca-sh-688800-%u745E%u53EF%u8FBE%2Ca-sz-000933-%u795E%u706B%u80A1%u4EFD%2Ca-sz-300258-%u7CBE%u953B%u79D1%u6280%2Cty-1-600395-%u76D8%u6C5F%u80A1%u4EFD%2Cty-0-002403-%u7231%u4ED5%u8FBE%2Cty-0-301112-%u4FE1%u90A6%u667A%u80FD%2Cty-1-603636-%u5357%u5A01%u8F6F%u4EF6%2Cty-1-601728-%u4E2D%u56FD%u7535%u4FE1%2Cty-1-600908-%u65E0%u9521%u94F6%u884C; st_pvi=22786667246730; st_sp=2023-01-21%2014%3A48%3A54; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=112; st_psi=20230123025238528-113200301202-0715885204',
        'Referer': 'https://data.eastmoney.com/',
        'Host': 'push2.eastmoney.com'
    }
    for page in tqdm.tqdm(range(101)):
        params = {'cb': 'jQuery112307786838608998163_1674412359630',
                  'fid': 'f12',
                  'po': '0',
                  'pz': '50',
                  'pn': str(page + 1),
                  'np': '1',
                  'fltt': '2',
                  'invt': '2',
                  'ut': 'b2884a393a59ad64002292a3e90d46a5',
                  'fs': 'm:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2',
                  'fields': 'f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124,f1,f13'
                  }

        res = requests.get(url, headers=header, params=params)
        res.encoding = "utf-8"

        html = res.text.lstrip(params['cb']+'(')
        html = html.rstrip(');')
        js_html = json.loads(html)
        js_data = js_html['data']
        js_diff = js_data['diff']

        for line in js_diff:
            output.write(line['f12'] + ' ' + line['f14'] + '\n')

        time.sleep(random.random()*5)

    output.close()

# if __name__ == "__main__":
#     get_list()


if __name__ == "__main__":
    with open('code_list.txt', 'r', encoding='utf-8') as codes:
        codes = codes.readlines()
        code_list = [line[0:6] for line in codes[4600:]]
        crawl_stock(code_list)
import time
from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from copy import deepcopy

import utils

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'


def train(model, dataset, optimizer):
    """
    Train for one epoch
    return loss and accuracy
    """
    model = model.to(device)
    model.train()

    dataset.mission='training'

    crit = nn.CrossEntropyLoss()

    loss_list = []
    acc_list = []
    for data, targets, indexes in dataset.conv():
        optimizer.zero_grad()

        data = data.to(device)
        targets = targets.to(device)
        output = model(data, targets, indexes)

        loss = crit(output, targets)
        loss_list.append(loss.item())
        loss.backward()
        
        # ze = dataset.global_width - len(indexes['targets'])
        t = torch.argmax(output, dim=1) == targets
        acc = sum(t) / len(targets)
        acc_list.append(acc.item())

        optimizer.step()

    print('Average loss:', sum(loss_list)/len(loss_list))
    print('Average accuracy: {}%'.format(sum(acc_list)/len(acc_list) * 100))

    return sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)






def main_train(model, dataset):

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    now = datetime.now()
    # with open('log/train_log.txt','a') as f:
    #     f.write('Time:'+ now.strftime("%m/%d/%Y, %H:%M:%S")+'\n')

    best_model = deepcopy(model.state_dict())
    best_acc = 0

    for i in range(10):

        print('-'*6+'Epoch:{}'.format(i+1)+'-'*6+'\n')
        loss_list = []
        acc_list = []
        for data, targets, indexes in dataset.conv():
            optimizer.zero_grad()

            data = data.to(device)
            targets = targets.to(device)
            output = model(data, targets, indexes)

            loss = crit(output, targets)
            loss_list.append(loss.item())
            loss.backward()
            
            # ze = dataset.global_width - len(indexes['targets'])
            t = torch.argmax(output, dim=1) == targets
            acc = sum(t) / len(targets)
            acc_list.append(acc)

            optimizer.step()

            # print(model.query)
            
            
        # print('Time used:', time.time() - t1, 'seconds')
        print('Average loss:', sum(loss_list)/len(loss_list))
        print('Average accuracy: {}%'.format(sum(acc_list)/len(acc_list) * 100))

    return sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)

    #     with open('log/train_log.txt','a') as f:
    #         f.write('Epoch:{}'.format(i+1))
    #         f.write('Average loss: {}\n'.format(sum(loss_list)/len(loss_list)))
    #         f.write('Average accuracy: {}%\n\n'.format(sum(acc_list)/len(acc_list) * 100))
        
    #     if best_acc >= sum(acc_list)/len(acc_list) and i >= 3:
    #         break
    #     else:
    #         best_acc = sum(acc_list)/len(acc_list)
    #         best_model = deepcopy(model.state_dict())
            

    # torch.save(best_model, 'save/class_net_3.pkl')




def validate(model, dataset):
    model = model.to(device)
    torch.cuda.empty_cache()
    model.eval()
    dataset.mission='validation'

    # model.load_state_dict(torch.load('save/class_net_3.pkl'))


    with torch.no_grad():

        acc_list = []
        for data, targets, indexes in dataset.conv():

            data = data.to(device)
            targets = targets.to(device)

            # print(torch.cuda.memory_summary())

            output = model(data, targets, indexes)

            values, ind = torch.topk(output[:,1], 20)
            a = targets[ind] >=1
            a = a.cpu()
            a = sum(a)/len(a)
            acc_list.append(a.item())


            # t = torch.argmax(output, dim=1) == targets
            # acc = sum(t) / len(targets)
            # acc_list.append(acc)
        print('Average accuracy: {}%'.format(sum(acc_list)/len(acc_list) * 100))

    return sum(acc_list)/len(acc_list)


def draw_fig(model, dataset):
    model = model.to(device)
    model.eval()

    acc_list = []

    dataset.mission='training'
    for data, targets, indexes in dataset.conv():
        data = data.to(device)
        targets = targets.to(device)
        output = model(data, targets, indexes)

        values, ind = torch.topk(output[:,1], 20)
        a = targets[ind] >=1
        a = sum(a)/len(a)
        a = a.cpu().item()


        print(utils.trading_day.day_list[dataset.current_time])

        acc_list.append(sum(a)/len(a))


    dataset.mission='validation'



    dataset.mission='test'


        

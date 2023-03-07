import time
import torch
import utils
import train
import json

from datetime import datetime
from torch import optim
from copy import deepcopy

def main_loop(model, dataset):

    f = open('config.json')
    configs = json.load(f)['train_setting']
    patience = configs['patience']

    acc_list = []
    temper = 0
    best_acc = 0

    now = datetime.now()
    with open('log/train_log.txt','a') as f:
        f.write('Time:'+ now.strftime("%m/%d/%Y, %H:%M:%S")+'\n')

    optimizer = optim.Adam(model.parameters(), lr=configs['learning_rate'])

    for i in range(configs['max_epochs']):
        train_loss, train_acc = train.train(model, dataset, optimizer)
        acc = train.validate(model, dataset)

        with open('log/train_log.txt','a') as f:
            f.write('Epoch:{}\n'.format(i+1))
            f.write('Train loss: {}\n'.format(train_loss))
            f.write('Train accuracy: {}%\n'.format(train_acc * 100))
            f.write('Validation accuracy: {}%\n\n'.format(acc * 100))


        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(model.state_dict())

        if len(acc_list) > 0:
            if acc < acc_list[-1]:
                temper = temper + 1
            else:
                temper = 0
        acc_list.append(acc)

        if temper > patience:
            break

        # if best_acc >= sum(acc_list)/len(acc_list) and i >= 3:
        #     break
        # else:
        #     best_acc = sum(acc_list)/len(acc_list)
        #     best_model = deepcopy(model.state_dict())
            

    torch.save(best_model, 'save/class_net_4.pkl')

    with open('log/train_log.txt','a') as f:
        f.write('Best Validation Acc:{}\n\n'.format(best_acc))




if __name__ == "__main__":
    dataset, model = utils.Initialize()
    
    # train.main_train(model, dataset)
    # model.load_state_dict(torch.load('save/class_net_3.pkl'))

    main_loop(model, dataset)
    


    

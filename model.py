import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import json


DATA_DIM = 7


def load_config():
    f = open('config.json')
    configs = json.load(f)

    model_config = configs['model_config']

    return model_config


class Predictor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config

        lstm_config = model_config['lstm']

        self.rnn = nn.LSTM(DATA_DIM, lstm_config['hidden_size'], lstm_config['num_layers'],
                            batch_first=True, dropout=lstm_config['dropout'])

        # self.pos_embed = nn.Parameter(torch.randn(model_config['local_length'], DATA_DIM))

        # self.query01 = nn.Parameter(torch.randn(model_config['global_width'], model_config['local_length'], DATA_DIM))
        # self.key01 = nn.Parameter(torch.randn(model_config['global_width'], model_config['local_length'], DATA_DIM))

        # self.att0 = Attention_layer(DATA_DIM, 1, DATA_DIM)

        # self.fc01 = nn.Linear(model_config['local_length'] * DATA_DIM, model_config['first_dim'])
        # self.fc02 = nn.Linear(model_config['first_dim'], model_config['first_dim'])




        self.query1 = nn.Parameter(torch.randn(model_config['global_width'], model_config['qk_size'])[None, :, :])
        self.key1 = nn.Parameter(torch.randn(model_config['global_width'], model_config['qk_size'])[None, :, :])

        self.att1 = Attention_layer(model_config['global_length'], model_config['num_heads'], model_config['qk_size'])
        

        self.query2 = nn.Parameter(torch.randn(model_config['global_width'], model_config['qk_size'])[None, :, :])
        self.key2 = nn.Parameter(torch.randn(model_config['global_width'], model_config['qk_size'])[None, :, :])

        self.att2 = Attention_layer(model_config['global_length'], model_config['num_heads'], model_config['qk_size'])  

        self.query3 = nn.Parameter(torch.randn(model_config['global_width'], model_config['qk_size'])[None, :, :])
        self.key3 = nn.Parameter(torch.randn(model_config['global_width'], model_config['qk_size'])[None, :, :])

        self.att3 = Attention_layer(model_config['global_length'], model_config['num_heads'], model_config['qk_size'])  

        self.query4 = nn.Parameter(torch.randn(model_config['global_width'], model_config['qk_size'])[None, :, :])
        self.key4 = nn.Parameter(torch.randn(model_config['global_width'], model_config['qk_size'])[None, :, :])

        self.att4 = Attention_layer(model_config['global_length'], model_config['num_heads'], model_config['qk_size'])  


        self.fc1 = nn.Linear(model_config['global_length'] + model_config['first_dim'], model_config['fc_hidden'])
        self.relu = nn.ReLU()
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(model_config['fc_hidden'], 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, targets, indexes):
        '''
        inputs: shape = global_width, local_length, DATA_DIM
        '''

        output1, (_, _) = self.rnn(inputs)
        output1 = output1[:, -1, :].squeeze()

        # inputs = inputs + self.pos_embed
        # output1 = self.att0(self.query01, self.key01, inputs)
        # # print(torch.flatten(output1, start_dim=1).shape)
        # output1 = self.fc01(torch.flatten(output1, start_dim=1))
        # output1 = self.relu(output1)
        # output1 = self.fc02(output1)

        # output1 = torch.zeros(self.model_config['global_width'], self.model_config['first_dim']).to(inputs.get_device())



        value = inputs[:, -self.model_config['global_length']:, 0].squeeze()
        value = value[None, :, :]
        
        output2 = self.att1(self.query1, self.key1, value)
        output2 = self.att2(self.query2, self.key2, output2)
        output2 = output2 + value
        output2 = self.att3(self.query3, self.key3, output2)
        output2 = self.att4(self.query4, self.key4, output2)
        output2 = output2.squeeze()


        concated = torch.cat([output1, output2], dim=1)  
        concated = self.fc1(concated)
        concated = self.relu(concated)
        concated = self.dp1(concated)
        output = self.fc2(concated)
        output = self.softmax(output)

        # for i in range(self.model_config['global_width']):
        #     if i not in indexes['targets']:
        #         output[i] = 0
    
        # mask = (inputs[:, 0, 0].squeeze() != 0).to(torch.int32)
        output = output[indexes['targets'],:]

        return output

class Attention_layer(nn.Module):
    def __init__(self, global_length, num_heads, qk_size):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(qk_size, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(global_length)

        self.fc1 = nn.Linear(global_length, global_length)
        self.fc2 = nn.Linear(global_length, global_length)
        self.layer_norm2 = nn.LayerNorm(global_length)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, query, key, value):
        output, attn_output_weights = self.multihead_attn(query, key, value)
        output = output + value
        output = self.layer_norm1(output)

        output = self.fc2(self.relu(self.fc1(output))) + output
        output = self.layer_norm2(output)

        return output


        

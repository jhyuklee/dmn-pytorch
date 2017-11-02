import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import sys
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class NETS(nn.Module):
    def __init__(self, config, widx2vec, iter=None):
        super(NETS, self).__init__()
        self.config = config

        # embedding layers
        self.word_embed = nn.Embedding(config.word_vocab_size, config.word_embed_dim,
                padding_idx=1)
        
        # dimensions according to settings
        self.rnn_idim = config.word_embed_dim

        # rnn layers
        self.t_rnn = nn.GRU(self.rnn_idim, config.rnn_hdim, config.rnn_ln,
                dropout=config.rnn_dr, batch_first=True, bidirectional=False)

        # linear layers
        self.output_fc1 = nn.Linear(config.fc1_dim, config.output_dim)

        # initialization
        self.init_word_embed(widx2vec)
        params = self.model_params(debug=False)
        self.optimizer = optim.Adam(params, lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()

    def init_word_embed(self, widx2vec):
        self.word_embed.weight.data.copy_(torch.from_numpy(np.array(widx2vec)))
        self.word_embed.weight.requires_grad = False

    def model_params(self, debug=True):
        print('model parameters: ', end='')
        params = []
        total_size = 0
        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
        print('%s\n' % '{:,}'.format(total_size))
        return params
    
    def init_rnn_h(self, batch_size):
        return (Variable(torch.zeros(
            self.config.t_rnn_ln*1, batch_size, self.config.t_rnn_hdim)).cuda(),
                Variable(torch.zeros(
            self.config.t_rnn_ln*1, batch_size, self.config.t_rnn_hdim)).cuda())

    def rnn_layer(self, tc):

        # word embedding for title, then concat
        word_embed = self.word_embed(tw.view(-1, self.config.max_sentlen))
        lstm_input = word_embed

        # typical rnn
        init_rnn_h = self.init_t_rnn_h(lstm_input.size(0))
        lstm_out, _ = self.t_rnn(lstm_input, init_t_rnn_h)
        lstm_out = lstm_out.contiguous().view(-1, self.config.t_rnn_hdim)
        lstm_out = lstm_out.cpu()
        fw_tl = (torch.arange(0, tl.size(0)).type(torch.LongTensor) *
                self.config.max_sentlen + tl.data.cpu() - 1)
        selected = lstm_out[fw_tl,:].view(-1, self.config.t_rnn_hdim).cuda()

        return selected 

    def matching_layer(self, intention, snapshot_mf, grid):
        day_fc1 = F.relu(self.day_fc1(output))
        slot_fc1 = F.relu(self.slot_fc1(output))
        output = self.output_fc1(torch.cat((day_fc1, slot_fc1), 1))
        return output

    def forward(self, inputs):
        output = inputs

        return output

    def get_regloss(self, weight_decay=None):
        if weight_decay is None:
            weight_decay = self.config.wd
        reg_loss = 0
        params = [] # add params here
        for param in params:
            reg_loss += torch.norm(param.weight, 2)
        return reg_loss * weight_decay

    def decay_lr(self, lr_decay=None):
        if lr_decay is None:
            lr_decay = self.config.lr_decay
        self.config.lr /= lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr

        print('\tlearning rate decay to %.3f' % self.config.lr)

    def get_metrics(self, outputs, targets):
        max_idx = torch.max(outputs, 1)[1].data.cpu().numpy()
        outputs_topk = torch.topk(outputs, topk)[1].data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        outputs = outputs.data.cpu().numpy()

        acc = np.mean([float(k == tk[0]) for (k, tk)
            in zip(targets, outputs_topk)]) * 100

        return acc
 
    def save_checkpoint(self, state, filename=None):
        if filename is None:
            filename = self.config.checkpoint_dir + self.config.model_name + '.pth'
        else:
            filename = self.config.checkpoint_dir + filename
        print('\t=> save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = self.config.checkpoint_dir + self.config.model_name + '.pth'
        else:
            filename = self.config.checkpoint_dir + filename
        print('\t=> load checkpoint %s' % filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.config = checkpoint['config']


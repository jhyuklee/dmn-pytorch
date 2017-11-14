import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import sys
from torch.autograd import Variable


class DMN(nn.Module):
    def __init__(self, config, idx2vec):
        super(DMN, self).__init__()
        self.config = config

        # embedding layers
        self.word_embed = nn.Embedding(config.word_vocab_size, config.word_embed_dim,
                padding_idx=0)
        
        # dimensions according to settings
        self.s_rnn_idim = config.word_embed_dim
        self.q_rnn_idim = config.word_embed_dim
        self.e_rnn_idim = config.s_rnn_hdim
        self.a_rnn_idim = config.q_rnn_hdim + config.word_vocab_size
        self.m_cell_idim = config.e_rnn_hdim
        self.z_dim = config.s_rnn_hdim * 7 + 2 # simplified (all hdim equal)

        # rnn layers
        self.s_rnn = nn.GRU(self.s_rnn_idim, config.s_rnn_hdim, config.s_rnn_ln,
                dropout=config.s_rnn_dr, batch_first=True, bidirectional=False)
        self.q_rnn = nn.GRU(self.q_rnn_idim, config.q_rnn_hdim, config.q_rnn_ln,
                dropout=config.q_rnn_dr, batch_first=True, bidirectional=False)
        self.e_cell = nn.GRUCell(self.e_rnn_idim, config.e_rnn_hdim)
        self.m_cell = nn.GRUCell(self.m_cell_idim, config.m_cell_hdim)
        self.a_cell = nn.GRUCell(self.a_rnn_idim, config.a_rnn_hdim)

        # linear layers
        self.z_sq = nn.Linear(config.s_rnn_hdim, config.q_rnn_hdim, bias=False)
        # self.z_sm = nn.Linear(config.s_rnn_hdim, config.m_cell_hdim, bias=False)
        self.out = nn.Linear(config.m_cell_hdim, config.word_vocab_size, bias=False)
        self.g1 = nn.Linear(self.z_dim, config.g1_dim)
        self.g2 = nn.Linear(config.g1_dim, 1)

        # initialization
        self.init_word_embed(idx2vec)
        params = self.model_params(debug=False)
        self.optimizer = optim.Adam(params, lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()

    def init_word_embed(self, idx2vec):
        self.word_embed.weight.data.copy_(torch.from_numpy(np.array(idx2vec)))
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
        return Variable(torch.zeros(
            self.config.s_rnn_ln*1, batch_size, self.config.s_rnn_hdim)).cuda()

    def init_cell_h(self, batch_size):
        return Variable(torch.zeros(
            batch_size, self.config.s_rnn_hdim)).cuda()

    def input_module(self, stories, s_lens):
        word_embed = self.word_embed(stories)
        init_s_rnn_h = self.init_rnn_h(stories.size(0))
        gru_out, _ = self.s_rnn(word_embed, init_s_rnn_h)
        gru_out = gru_out.contiguous().view(-1, self.config.s_rnn_hdim).cpu()
        s_lens_offset = (torch.arange(0, stories.size(0)).type(torch.LongTensor)
                * self.config.max_slen).unsqueeze(1)
        s_lens = (s_lens + s_lens_offset).view(-1)
        selected = gru_out[s_lens,:].view(-1, self.config.max_sentnum, 
                self.config.s_rnn_hdim).cuda()
        return selected 

    def question_module(self, questions, q_lens):
        word_embed = self.word_embed(questions)
        init_q_rnn_h = self.init_rnn_h(questions.size(0))
        gru_out, _ = self.q_rnn(word_embed, init_q_rnn_h)
        gru_out = gru_out.contiguous().view(-1, self.config.q_rnn_hdim).cpu()
        q_lens = (torch.arange(0, questions.size(0)).type(torch.LongTensor)
                * self.config.max_qlen + q_lens)
        selected = gru_out[q_lens,:].view(-1, self.config.q_rnn_hdim).cuda() 
        return selected

    def episodic_memory_module(self, s_rep, q_rep, e_lens, memory):
        q_rep = q_rep.unsqueeze(1).expand_as(s_rep)
        memory = memory.unsqueeze(1).expand_as(s_rep)
        sw = self.z_sq(s_rep.view(-1, self.config.s_rnn_hdim)).view(
                q_rep.size())
        swq = torch.sum(sw * q_rep, 2, keepdim=True)
        swm = torch.sum(sw * memory, 2, keepdim=True)
        Z = torch.cat([s_rep, memory, q_rep, s_rep*q_rep, s_rep*memory,
            torch.abs(s_rep-q_rep), torch.abs(s_rep-memory), swq, swm], 2)
        # print('z', Z.size())
        G = F.sigmoid(self.g2(F.tanh(self.g1(Z.view(-1, self.z_dim)))))
        G = G.view(-1, self.config.max_sentnum).unsqueeze(2)
        G = torch.transpose(G, 0, 1).contiguous()
        s_rep = torch.transpose(s_rep, 0, 1).contiguous()
        # print('g', G.size())

        e_rnn_h = self.init_cell_h(s_rep.size(1))
        # print('input', s_rep.size())
        # print('hidden', e_rnn_h.size())
        hiddens = []
        for step, (gg, ss) in enumerate(zip(G, s_rep)):
            e_rnn_h = gg * self.e_cell(ss, e_rnn_h) + (1 - gg) * e_rnn_h
            hiddens.append(e_rnn_h)
        hiddens = torch.transpose(torch.stack(hiddens), 0, 1).contiguous().view(
                -1, self.config.e_rnn_hdim).cpu()
        e_lens = (torch.arange(0, s_rep.size(1)).type(torch.LongTensor)
                * self.config.max_sentnum + e_lens)
        selected = hiddens[e_lens,:].view(-1, self.config.e_rnn_hdim).cuda() 
        # print('out', selected.size())
        return selected, torch.transpose(G, 0, 1).contiguous()

    def answer_module(self, q_rep, memory):
        y = F.softmax(self.out(memory))
        a_rnn_h = memory
        ys = []
        #print('q_rep', q_rep[0,:])
        for step in range(2):
            a_rnn_h = self.a_cell(torch.cat((y, q_rep), 1), a_rnn_h)
            y = F.softmax(self.out(a_rnn_h))
            ys.append(y)
        ys = torch.transpose(torch.stack(ys), 0, 1).contiguous()
        return ys

    def forward(self, stories, questions, s_lens, q_lens, e_lens):
        s_rep = self.input_module(stories, s_lens)
        # print('stories', s_rep.size())
        q_rep = self.question_module(questions, q_lens)
        # print('questions', q_rep.size())
        
        memory = q_rep # initial memory
        gates = []
        for episode in range(self.config.max_episode):
            e_rep, gate = self.episodic_memory_module(s_rep, q_rep, e_lens, memory)
            gates.append(gate)
            memory = self.m_cell(e_rep, memory)
        # print('memory', memory.size())
        gates = torch.transpose(torch.stack(gates), 0, 1).contiguous().squeeze(3)
        
        outputs = self.answer_module(q_rep, memory)
        # print('outputs', outputs.size())

        return outputs, gates

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

    def get_metrics(self, outputs, targets, multiple=False):
        if not multiple:
            outputs = outputs[:,0,:]
            targets = targets[:,0]

            max_idx = torch.max(outputs, 1)[1].data.cpu().numpy()
            outputs_topk = torch.topk(outputs, 3)[1].data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()

            acc = np.mean([float(k == tk[0]) for (k, tk)
                in zip(targets, outputs_topk)]) * 100
        else:
            # TODO: multiple outputs
            acc = 0

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


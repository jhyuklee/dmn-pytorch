import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
from utils import progress 
from torch.autograd import Variable
from datetime import datetime


def run_epoch(m, d, ep, mode='tr', set_num=1, is_train=True):
    total_metrics = np.zeros(2)
    total_step = 0.0
    print_step = m.config.print_step
    start_time = datetime.now()
    d.shuffle_data(seed=None, mode='tr')

    while True:
        m.optimizer.zero_grad()
        stories, questions, answers, sup_facts, s_lens, q_lens, e_lens= \
                d.get_next_batch(mode, set_num)
        #d.decode_data(stories[0], questions[0], answers[0], sup_facts[0], s_lens[0])
        wrap_tensor = lambda x: torch.LongTensor(np.array(x))
        wrap_var = lambda x: Variable(wrap_tensor(x)).cuda()
        stories = wrap_var(stories)
        questions = wrap_var(questions)
        answers = wrap_var(answers)
        sup_facts = wrap_var(sup_facts) - 1
        s_lens = wrap_tensor(s_lens)
        q_lens = wrap_tensor(q_lens)
        e_lens = wrap_tensor(e_lens)

        if is_train: m.train()
        else: m.eval()
        outputs, gates = m(stories, questions, s_lens, q_lens, e_lens)
        a_loss = m.criterion(outputs[:,0,:], answers[:,0])
        if answers.size(1) > 1: # multiple answer
            for ans_idx in range(m.config.max_alen):
                a_loss += m.criterion(outputs[:,ans_idx,:], answers[:,ans_idx])
        for episode in range(5):
            if episode == 0:
                g_loss = m.criterion(gates[:,episode,:], sup_facts[:,episode]) 
            else:
                g_loss += m.criterion(gates[:,episode,:], sup_facts[:,episode])
        beta = 0 if ep < m.config.beta_cnt and mode == 'tr' else 1
        alpha = 1
        metrics = m.get_metrics(outputs, answers, multiple=answers.size(1)>1)
        total_loss = alpha * g_loss + beta * a_loss

        if is_train:
            total_loss.backward()
            nn.utils.clip_grad_norm(m.parameters(), m.config.grad_max_norm)
            m.optimizer.step()

        total_metrics[0] += total_loss.data[0]
        total_metrics[1] += metrics
        total_step += 1.0
        
        # print step
        if d.get_batch_ptr(mode) % print_step == 0 or total_step == 1:
            et = int((datetime.now() - start_time).total_seconds())
            _progress = progress(
                    d.get_batch_ptr(mode) / d.get_dataset_len(mode, set_num))
            if d.get_batch_ptr(mode) == 0:
                _progress = progress(1)
            _progress += '[%s] time: %s' % (
                    '\t'.join(['{:.2f}'.format(k) 
                    for k in total_metrics / total_step]),
                    '{:2d}:{:2d}:{:2d}'.format(et//3600, et%3600//60, et%60))
            sys.stdout.write(_progress)
            sys.stdout.flush()

            # end of an epoch
            if d.get_batch_ptr(mode) == 0:
                et = (datetime.now() - start_time).total_seconds()
                print('\n\ttotal metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                    for k in total_metrics / total_step]))) 
                break

    return total_metrics / total_step

